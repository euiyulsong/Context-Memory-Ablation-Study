#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
memory_ablation_3exp.py

Three clean LongMemEval memory ablations using an OpenAI-compatible endpoint.

EXPERIMENT A — Recursive rewrite vs Python append
  recursive_typed_no_limit_full
  python_append_typed

EXPERIMENT B — Prompt length instruction vs no prompt length instruction
  recursive_typed_prompt_limit_full
  recursive_typed_no_limit_full

EXPERIMENT C — Post-generation truncation vs no truncation
  recursive_typed_no_limit_truncate
  recursive_typed_no_limit_full

Important controls
------------------
* Same dataset / model / QA prompt / temperature.
* Memory writer never sees the evaluation question or gold answer.
* max summary/update calls per example defaults to 2.
* Prompt-length ablation uses the SAME API max_tokens ceiling in both arms.
* Truncation ablation generates ONE identical no-limit recursive memory and only
  changes what is passed to QA. This avoids regeneration noise.
* Strict normalized Exact Match is PRIMARY; token F1 and relaxed EM are diagnostics.

Expected dataset next to this script:
  longmemeval_s_cleaned.json

Dependencies:
  pip install openai tqdm

OpenRouter example:
  export VLLM_BASE_URL="https://openrouter.ai/api/v1"
  export VLLM_API_KEY="sk-or-v1-..."
  export VLLM_MODEL="qwen/qwen3.5-35b-a3b"

Recommended pilot:
  python3 memory_ablation_3exp.py --limit 50 --workers 8 --max-summary-updates 2
"""

from __future__ import annotations

import argparse
import collections
import concurrent.futures
import csv
import json
import math
import os
import re
import statistics
import sys
import threading
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from tqdm import tqdm
from openai import OpenAI


METHODS = [
    "recursive_typed_no_limit_full",
    "python_append_typed",
    "recursive_typed_prompt_limit_full",
    "recursive_typed_no_limit_truncate",
]

DATA_CANDIDATES = [
    "longmemeval_s_cleaned.json",
    "longmemeval_s.json",
    "longmemeval_oracle.json",
]

SUMMARY_SYSTEM = """You construct long-term memory from conversations.
Use only information explicitly supported by the conversation.
Never infer unstated facts and never use outside knowledge.
Preserve names, entities, dates, quantities, preferences, plans, relationships,
and changes over time. Do not answer a future question; construct memory only."""

TYPED_EXTRACT_PROMPT = """Extract typed atomic memories from the NEW CONVERSATION.

Output ONLY one memory per line in exactly this form:
[TYPE] fact

Allowed TYPE values:
IDENTITY, PREFERENCE, WORK, RELATION, LOCATION, ACTIVITY, PLAN, EVENT, TEMPORAL, OTHER

Rules:
- One explicit fact per line.
- Preserve exact names, entities, dates, quantities, and relationships.
- Preserve state changes and their time when stated.
- Do not speculate.
- Do not refer to any future question.

NEW CONVERSATION:
{chunk}

TYPED MEMORY:"""

RECURSIVE_TYPED_NO_LIMIT_PROMPT = """Rewrite the EXISTING MEMORY together with the NEW CONVERSATION as complete typed atomic memory.

Output ONLY one memory per line in exactly this form:
[TYPE] fact

Allowed TYPE values:
IDENTITY, PREFERENCE, WORK, RELATION, LOCATION, ACTIVITY, PLAN, EVENT, TEMPORAL, OTHER

Rules:
- Retain useful factual information already present in EXISTING MEMORY.
- Integrate all useful factual information from NEW CONVERSATION.
- Prefer newer information for the current state when facts change.
- Preserve earlier state and time when needed to answer historical questions later.
- Deduplicate semantically equivalent facts.
- Preserve exact names, entities, dates, quantities, and relationships.
- Never speculate.
- Do not intentionally shorten the memory merely for brevity.

EXISTING MEMORY:
{memory}

NEW CONVERSATION:
{chunk}

UPDATED TYPED MEMORY:"""

RECURSIVE_TYPED_LIMIT_PROMPT = """Rewrite the EXISTING MEMORY together with the NEW CONVERSATION as typed atomic memory.

Output ONLY one memory per line in exactly this form:
[TYPE] fact

Allowed TYPE values:
IDENTITY, PREFERENCE, WORK, RELATION, LOCATION, ACTIVITY, PLAN, EVENT, TEMPORAL, OTHER

Rules:
- Retain useful factual information already present in EXISTING MEMORY.
- Integrate useful factual information from NEW CONVERSATION.
- Prefer newer information for the current state when facts change.
- Preserve earlier state and time when needed to answer historical questions later.
- Deduplicate semantically equivalent facts.
- Preserve exact names, entities, dates, quantities, and relationships.
- Never speculate.
- IMPORTANT LENGTH CONSTRAINT: keep the COMPLETE updated memory at or below {word_limit} words.
- If the limit forces compression, prioritize concrete facts, dates, entities, quantities,
  preferences, relationships, plans, and state changes over conversational detail.

EXISTING MEMORY:
{memory}

NEW CONVERSATION:
{chunk}

UPDATED TYPED MEMORY:"""

QA_SYSTEM = """Answer factual questions using only the supplied memory/context.
Return ONLY the shortest answer span that directly answers the question.
Do not explain reasoning. Do not write 'Answer:'.
Preserve names, dates, and quantities exactly when possible.
If the answer cannot be determined from the memory, output exactly: I don't know."""

QA_USER = """MEMORY / CONTEXT:
{memory}

QUESTION:
{question}

SHORTEST ANSWER:"""


def normalize_answer(s: Any) -> str:
    if s is None:
        return ""
    s = unicodedata.normalize("NFKC", str(s)).lower().strip()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(ch if ch.isalnum() or ch.isspace() else " " for ch in s)
    return " ".join(s.split())


def exact_match(pred: str, gold: Any) -> float:
    golds = gold if isinstance(gold, list) else [gold]
    p = normalize_answer(pred)
    return float(any(p == normalize_answer(g) for g in golds))


def token_f1_single(pred: str, gold: str) -> float:
    p = normalize_answer(pred).split()
    g = normalize_answer(gold).split()
    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0
    common = collections.Counter(p) & collections.Counter(g)
    n = sum(common.values())
    if not n:
        return 0.0
    precision = n / len(p)
    recall = n / len(g)
    return 2 * precision * recall / (precision + recall)


def token_f1(pred: str, gold: Any) -> float:
    golds = gold if isinstance(gold, list) else [gold]
    return max(token_f1_single(pred, str(g)) for g in golds)


def relaxed_em(pred: str, gold: Any) -> float:
    golds = gold if isinstance(gold, list) else [gold]
    p = normalize_answer(pred)
    for g in golds:
        gg = normalize_answer(g)
        if p == gg or (p and gg and (p in gg or gg in p)):
            return 1.0
    return 0.0


def clean_answer(x: str) -> str:
    x = (x or "").strip()
    x = re.sub(r"^(final answer|answer)\s*:\s*", "", x, flags=re.I).strip()
    lines = [z.strip() for z in x.splitlines() if z.strip()]
    if lines:
        x = lines[0]
    return x.strip().strip('"').strip("'")


def approx_tokens(text: str) -> int:
    return 0 if not text else max(1, math.ceil(len(text) / 4.0))


def truncate_head_tail(text: str, max_tokens: int) -> Tuple[str, bool]:
    """Deterministic post-generation truncation. Keeps 35% head / 65% tail."""
    if approx_tokens(text) <= max_tokens:
        return text, False
    max_chars = max_tokens * 4
    marker = "\n...[POST_TRUNCATED]...\n"
    available = max(1, max_chars - len(marker))
    head = int(available * 0.35)
    tail = available - head
    return text[:head] + marker + text[-tail:], True


def dedupe_lines(lines: Sequence[str]) -> List[str]:
    seen = set()
    out = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        key = normalize_answer(line)
        if key and key not in seen:
            seen.add(key)
            out.append(line)
    return out


def find_dataset(script_dir: Path, explicit: Optional[str]) -> Path:
    if explicit:
        p = Path(explicit)
        if p.exists():
            return p
        raise FileNotFoundError(explicit)
    for name in DATA_CANDIDATES:
        p = script_dir / name
        if p.exists():
            return p
    raise FileNotFoundError(
        "Put longmemeval_s_cleaned.json next to this script or pass --data PATH"
    )


def load_dataset(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, list):
        return obj
    for key in ("data", "examples", "items"):
        if isinstance(obj.get(key), list):
            return obj[key]
    raise ValueError("Unsupported dataset JSON")


def turn_to_text(turn: Dict[str, Any]) -> str:
    return f"{str(turn.get('role', 'unknown')).upper()}: {turn.get('content', '')}"


def session_chunks(item: Dict[str, Any]) -> List[str]:
    sessions = item.get("haystack_sessions") or []
    dates = list(item.get("haystack_dates") or [])
    if len(dates) < len(sessions):
        dates.extend([None] * (len(sessions) - len(dates)))
    out = []
    for i, sess in enumerate(sessions):
        body = "\n".join(turn_to_text(t) for t in sess)
        date = dates[i] if i < len(dates) else None
        out.append((f"[DATE: {date}]\n" if date else "") + body)
    return out


def merge_contiguous_chunks(chunks: Sequence[str], max_chunks: int) -> List[str]:
    """Use all history, but merge it into <= max_chunks chronological super-chunks."""
    chunks = list(chunks)
    if max_chunks <= 0 or len(chunks) <= max_chunks:
        return chunks
    n = len(chunks)
    out = []
    for i in range(max_chunks):
        start = round(i * n / max_chunks)
        end = round((i + 1) * n / max_chunks)
        block = chunks[start:end]
        if block:
            out.append("\n\n--- NEXT SESSION ---\n\n".join(block))
    return out


@dataclass
class LLMConfig:
    base_url: str
    api_key: str
    model: str
    timeout: float
    max_retries: int


class LLM:
    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg
        self._local = threading.local()

    def client(self) -> OpenAI:
        c = getattr(self._local, "client", None)
        if c is None:
            c = OpenAI(
                base_url=self.cfg.base_url,
                api_key=self.cfg.api_key,
                timeout=self.cfg.timeout,
                max_retries=self.cfg.max_retries,
            )
            self._local.client = c
        return c

    def chat(self, system: str, user: str, max_tokens: int) -> str:
        kwargs = dict(
            model=self.cfg.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.0,
            max_tokens=max_tokens,
        )
        # OpenRouter/Qwen reasoning can make simple memory writing much slower.
        # OpenAI SDK forwards extra_body to OpenRouter-compatible providers.
        if "openrouter.ai" in self.cfg.base_url:
            kwargs["extra_body"] = {"reasoning": {"enabled": False}}
        r = self.client().chat.completions.create(**kwargs)
        return (r.choices[0].message.content or "").strip()


def build_recursive(
    llm: LLM,
    chunks: Sequence[str],
    summary_max_tokens: int,
    max_updates: int,
    prompt_word_limit: Optional[int],
) -> Tuple[str, int]:
    merged = merge_contiguous_chunks(chunks, max_updates)
    memory = "(empty)"
    calls = 0
    for chunk in merged:
        if prompt_word_limit is None:
            prompt = RECURSIVE_TYPED_NO_LIMIT_PROMPT.format(memory=memory, chunk=chunk)
        else:
            prompt = RECURSIVE_TYPED_LIMIT_PROMPT.format(
                memory=memory,
                chunk=chunk,
                word_limit=prompt_word_limit,
            )
        memory = llm.chat(SUMMARY_SYSTEM, prompt, max_tokens=summary_max_tokens)
        calls += 1
    return ("" if memory == "(empty)" else memory), calls


def build_python_append(
    llm: LLM,
    chunks: Sequence[str],
    summary_max_tokens: int,
    max_updates: int,
) -> Tuple[str, int]:
    """
    Crucial ablation:
    - New history is split into the SAME <= max_updates super-chunks as recursive.
    - LLM sees ONLY the new chunk and extracts typed facts.
    - Python appends those lines.
    - Previous memory is NEVER sent back to the LLM and NEVER rewritten.
    """
    merged = merge_contiguous_chunks(chunks, max_updates)
    lines: List[str] = []
    calls = 0
    for chunk in merged:
        prompt = TYPED_EXTRACT_PROMPT.format(chunk=chunk)
        x = llm.chat(SUMMARY_SYSTEM, prompt, max_tokens=summary_max_tokens)
        lines.extend(x.splitlines())
        calls += 1
    return "\n".join(dedupe_lines(lines)), calls


def answer(llm: LLM, memory: str, question: str, qa_max_tokens: int) -> str:
    return clean_answer(
        llm.chat(
            QA_SYSTEM,
            QA_USER.format(memory=memory, question=question),
            max_tokens=qa_max_tokens,
        )
    )


def evaluate_prediction(pred: str, gold: Any) -> Tuple[float, float, float]:
    return exact_match(pred, gold), token_f1(pred, gold), relaxed_em(pred, gold)


def process_example(item: Dict[str, Any], llm: LLM, args: argparse.Namespace) -> Dict[str, Dict[str, Any]]:
    """Build unique memories once, then derive all requested conditions."""
    chunks = session_chunks(item)
    question = str(item["question"])
    gold = item["answer"]
    qid = str(item["question_id"])
    qtype = str(item.get("question_type", "unknown"))

    need_no_limit = any(m in args.methods for m in [
        "recursive_typed_no_limit_full",
        "recursive_typed_no_limit_truncate",
    ])
    need_prompt_limit = "recursive_typed_prompt_limit_full" in args.methods
    need_append = "python_append_typed" in args.methods

    built: Dict[str, Tuple[str, int, float]] = {}

    if need_no_limit:
        t0 = time.perf_counter()
        mem, calls = build_recursive(
            llm, chunks, args.summary_max_tokens,
            args.max_summary_updates, prompt_word_limit=None,
        )
        built["recursive_no_limit"] = (mem, calls, time.perf_counter() - t0)

    if need_prompt_limit:
        t0 = time.perf_counter()
        mem, calls = build_recursive(
            llm, chunks, args.summary_max_tokens,
            args.max_summary_updates, prompt_word_limit=args.prompt_limit_words,
        )
        built["recursive_prompt_limit"] = (mem, calls, time.perf_counter() - t0)

    if need_append:
        t0 = time.perf_counter()
        mem, calls = build_python_append(
            llm, chunks, args.summary_max_tokens, args.max_summary_updates,
        )
        built["python_append"] = (mem, calls, time.perf_counter() - t0)

    rows: Dict[str, Dict[str, Any]] = {}
    for method in args.methods:
        if method == "recursive_typed_no_limit_full":
            full_memory, write_calls, write_sec = built["recursive_no_limit"]
            qa_memory = full_memory
            was_truncated = False

        elif method == "recursive_typed_no_limit_truncate":
            full_memory, write_calls, write_sec = built["recursive_no_limit"]
            qa_memory, was_truncated = truncate_head_tail(full_memory, args.truncate_tokens)

        elif method == "recursive_typed_prompt_limit_full":
            full_memory, write_calls, write_sec = built["recursive_prompt_limit"]
            qa_memory = full_memory
            was_truncated = False

        elif method == "python_append_typed":
            full_memory, write_calls, write_sec = built["python_append"]
            qa_memory = full_memory
            was_truncated = False

        else:
            raise ValueError(method)

        # Optional emergency QA context ceiling; default 0 = disabled.
        # This is intentionally separate from Experiment C.
        emergency_truncated = False
        if args.qa_context_cap_tokens > 0:
            qa_memory, emergency_truncated = truncate_head_tail(
                qa_memory, args.qa_context_cap_tokens
            )

        tqa = time.perf_counter()
        pred = answer(llm, qa_memory, question, args.qa_max_tokens)
        qa_sec = time.perf_counter() - tqa
        em, f1, rem = evaluate_prediction(pred, gold)

        rows[method] = {
            "method": method,
            "question_id": qid,
            "question_type": qtype,
            "question": question,
            "gold": gold,
            "prediction": pred,
            "em": em,
            "f1": f1,
            "relaxed_em": rem,
            "generated_memory_tokens": approx_tokens(full_memory),
            "qa_memory_tokens": approx_tokens(qa_memory),
            "post_truncated": bool(was_truncated),
            "emergency_truncated": bool(emergency_truncated),
            "write_calls": write_calls,
            "write_sec": write_sec,
            "qa_sec": qa_sec,
            "elapsed_sec": write_sec + qa_sec,
            # Saved for debugging. Can make files large, but this experiment needs it.
            "generated_memory": full_memory,
            "qa_memory": qa_memory,
            "error": None,
        }

    return rows


def load_completed(path: Path) -> Dict[str, Dict[str, Any]]:
    out = {}
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
                out[str(r["question_id"])] = r
            except Exception:
                pass
    return out


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def aggregate(method: str, rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    def mean(key: str) -> float:
        vals = [float(r[key]) for r in rows if r.get(key) is not None]
        return statistics.mean(vals) if vals else float("nan")

    return {
        "method": method,
        "n": len(rows),
        "EM": mean("em"),
        "F1": mean("f1"),
        "Relaxed_EM": mean("relaxed_em"),
        "Avg_generated_memory_tokens": mean("generated_memory_tokens"),
        "Avg_QA_memory_tokens": mean("qa_memory_tokens"),
        "Truncation_rate": mean("post_truncated"),
        "Avg_write_calls": mean("write_calls"),
        "Avg_write_sec": mean("write_sec"),
        "Avg_QA_sec": mean("qa_sec"),
        "errors": sum(bool(r.get("error")) for r in rows),
    }


def by_type(method: str, rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups = collections.defaultdict(list)
    for r in rows:
        groups[r["question_type"]].append(r)
    out = []
    for qt, rs in sorted(groups.items()):
        out.append({
            "method": method,
            "question_type": qt,
            "n": len(rs),
            "EM": statistics.mean(r["em"] for r in rs),
            "F1": statistics.mean(r["f1"] for r in rs),
            "Relaxed_EM": statistics.mean(r["relaxed_em"] for r in rs),
        })
    return out


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def paired_stats(name: str, a_name: str, b_name: str,
                 all_rows: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    a = {r["question_id"]: r for r in all_rows[a_name]}
    b = {r["question_id"]: r for r in all_rows[b_name]}
    ids = sorted(set(a) & set(b))
    if not ids:
        return {}

    a_only_correct = sum(a[i]["em"] == 1 and b[i]["em"] == 0 for i in ids)
    b_only_correct = sum(a[i]["em"] == 0 and b[i]["em"] == 1 for i in ids)
    both_correct = sum(a[i]["em"] == 1 and b[i]["em"] == 1 for i in ids)
    both_wrong = sum(a[i]["em"] == 0 and b[i]["em"] == 0 for i in ids)

    return {
        "experiment": name,
        "A": a_name,
        "B": b_name,
        "n_paired": len(ids),
        "A_EM": statistics.mean(a[i]["em"] for i in ids),
        "B_EM": statistics.mean(b[i]["em"] for i in ids),
        "A_F1": statistics.mean(a[i]["f1"] for i in ids),
        "B_F1": statistics.mean(b[i]["f1"] for i in ids),
        "A_only_correct": a_only_correct,
        "B_only_correct": b_only_correct,
        "both_correct": both_correct,
        "both_wrong": both_wrong,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data", default=None)
    p.add_argument("--output-dir", default="results_3exp")
    p.add_argument("--limit", type=int, default=50, help="0 = all")
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--overwrite", action="store_true")

    p.add_argument(
        "--methods",
        default=",".join(METHODS),
        help="Comma-separated subset of methods",
    )

    p.add_argument("--base-url", default=os.getenv("VLLM_BASE_URL", "http://127.0.0.1:8000/v1"))
    p.add_argument("--api-key", default=os.getenv("VLLM_API_KEY", "EMPTY"))
    p.add_argument("--model", default=os.getenv("VLLM_MODEL", ""))
    p.add_argument("--timeout", type=float, default=180.0)
    p.add_argument("--max-retries", type=int, default=3)

    p.add_argument("--max-summary-updates", type=int, default=2)
    p.add_argument(
        "--summary-max-tokens", type=int, default=768,
        help="Same hard generation ceiling for ALL writer methods. Keep high so Exp B measures prompt instruction, not API cap.",
    )
    p.add_argument(
        "--prompt-limit-words", type=int, default=200,
        help="Explicit natural-language memory length constraint used ONLY in prompt-limit arm.",
    )
    p.add_argument(
        "--truncate-tokens", type=int, default=128,
        help="Post-generation approximate token budget used ONLY in truncation arm.",
    )
    p.add_argument(
        "--qa-context-cap-tokens", type=int, default=0,
        help="Emergency common context cap. 0 disables it. Leave 0 for clean Exp C.",
    )
    p.add_argument("--qa-max-tokens", type=int, default=32)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    bad = [m for m in args.methods if m not in METHODS]
    if bad:
        raise SystemExit(f"Unknown methods: {bad}; allowed={METHODS}")
    if not args.model:
        raise SystemExit("Set VLLM_MODEL or pass --model")

    script_dir = Path(__file__).resolve().parent
    data_path = find_dataset(script_dir, args.data)
    data = load_dataset(data_path)
    data = data[args.offset:]
    if args.limit > 0:
        data = data[:args.limit]

    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = script_dir / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    llm = LLM(LLMConfig(
        base_url=args.base_url,
        api_key=args.api_key,
        model=args.model,
        timeout=args.timeout,
        max_retries=args.max_retries,
    ))

    paths = {m: out_dir / f"{m}.jsonl" for m in args.methods}
    if args.overwrite:
        for p in paths.values():
            if p.exists():
                p.unlink()

    completed = {m: load_completed(paths[m]) for m in args.methods}
    wanted_ids = {str(x["question_id"]) for x in data}

    # Skip an example only if every requested method is already complete.
    todo = [
        item for item in data
        if not all(str(item["question_id"]) in completed[m] for m in args.methods)
    ]

    print("=" * 118)
    print("LONGMEMEVAL — THREE MEMORY ABLATIONS")
    print("=" * 118)
    print(f"dataset              : {data_path}")
    print(f"run examples         : {len(data)} (todo={len(todo)}, offset={args.offset})")
    print(f"methods              : {args.methods}")
    print(f"endpoint             : {args.base_url}")
    print(f"model                : {args.model}")
    print(f"workers              : {args.workers}")
    print(f"summary update cap   : {args.max_summary_updates}")
    print(f"writer hard max      : {args.summary_max_tokens} tokens (SAME for all writer arms)")
    print(f"prompt length limit  : {args.prompt_limit_words} words (prompt-limit arm only)")
    print(f"post truncate budget : ~{args.truncate_tokens} tokens (truncate arm only)")
    print("PRIMARY metric       : strict normalized EM")
    print()
    print("A: recursive_typed_no_limit_full  vs python_append_typed")
    print("B: recursive_typed_prompt_limit_full vs recursive_typed_no_limit_full")
    print("C: recursive_typed_no_limit_truncate vs recursive_typed_no_limit_full")
    print()

    if todo:
        bar = tqdm(total=len(todo), desc="examples")
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
            fut_to_item = {ex.submit(process_example, item, llm, args): item for item in todo}
            for fut in concurrent.futures.as_completed(fut_to_item):
                item = fut_to_item[fut]
                qid = str(item["question_id"])
                try:
                    result = fut.result()
                    for m, row in result.items():
                        # If one arm was already done from a previous partial run, don't duplicate.
                        if qid not in completed[m]:
                            append_jsonl(paths[m], row)
                            completed[m][qid] = row
                except Exception as e:
                    for m in args.methods:
                        if qid in completed[m]:
                            continue
                        row = {
                            "method": m,
                            "question_id": qid,
                            "question_type": str(item.get("question_type", "unknown")),
                            "question": item.get("question"),
                            "gold": item.get("answer"),
                            "prediction": "",
                            "em": 0.0, "f1": 0.0, "relaxed_em": 0.0,
                            "generated_memory_tokens": 0,
                            "qa_memory_tokens": 0,
                            "post_truncated": False,
                            "emergency_truncated": False,
                            "write_calls": None,
                            "write_sec": None,
                            "qa_sec": None,
                            "elapsed_sec": None,
                            "generated_memory": "",
                            "qa_memory": "",
                            "error": f"{type(e).__name__}: {e}",
                        }
                        append_jsonl(paths[m], row)
                        completed[m][qid] = row
                bar.update(1)
        bar.close()

    all_rows: Dict[str, List[Dict[str, Any]]] = {}
    summary_rows = []
    type_rows = []
    for m in args.methods:
        rows = [r for qid, r in completed[m].items() if qid in wanted_ids]
        all_rows[m] = rows
        summary_rows.append(aggregate(m, rows))
        type_rows.extend(by_type(m, rows))

    write_csv(out_dir / "summary.csv", summary_rows)
    write_csv(out_dir / "by_question_type.csv", type_rows)

    pair_rows = []
    if "recursive_typed_no_limit_full" in all_rows and "python_append_typed" in all_rows:
        pair_rows.append(paired_stats(
            "A_recursive_vs_python_append",
            "recursive_typed_no_limit_full", "python_append_typed", all_rows,
        ))
    if "recursive_typed_prompt_limit_full" in all_rows and "recursive_typed_no_limit_full" in all_rows:
        pair_rows.append(paired_stats(
            "B_prompt_length_limit",
            "recursive_typed_prompt_limit_full", "recursive_typed_no_limit_full", all_rows,
        ))
    if "recursive_typed_no_limit_truncate" in all_rows and "recursive_typed_no_limit_full" in all_rows:
        pair_rows.append(paired_stats(
            "C_post_generation_truncation",
            "recursive_typed_no_limit_truncate", "recursive_typed_no_limit_full", all_rows,
        ))
    pair_rows = [r for r in pair_rows if r]
    write_csv(out_dir / "paired_comparisons.csv", pair_rows)

    with (out_dir / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 118)
    print("RESULT")
    print("=" * 118)
    hdr = f"{'method':40s} {'n':>5s} {'EM':>8s} {'F1':>8s} {'RelEM':>8s} {'gen_tok':>9s} {'qa_tok':>9s} {'trunc%':>8s}"
    print(hdr)
    print("-" * len(hdr))
    for r in sorted(summary_rows, key=lambda x: x["EM"], reverse=True):
        print(
            f"{r['method'][:40]:40s} {r['n']:5d} {r['EM']:8.4f} {r['F1']:8.4f} "
            f"{r['Relaxed_EM']:8.4f} {r['Avg_generated_memory_tokens']:9.1f} "
            f"{r['Avg_QA_memory_tokens']:9.1f} {100*r['Truncation_rate']:7.1f}%"
        )

    print("\nPAIRED COMPARISONS")
    for r in pair_rows:
        print(
            f"- {r['experiment']}: A_EM={r['A_EM']:.4f} vs B_EM={r['B_EM']:.4f}; "
            f"A_only={r['A_only_correct']} B_only={r['B_only_correct']} "
            f"both_correct={r['both_correct']}"
        )

    print(f"\nSaved under: {out_dir}")
    print("  summary.csv")
    print("  paired_comparisons.csv")
    print("  by_question_type.csv")
    print("  <method>.jsonl  # includes generated_memory and qa_memory for inspection")


if __name__ == "__main__":
    main()
