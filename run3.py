#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
memory_squad_chunk_ablation.py

Goal
----
Build long-term memory from LongMemEval conversations, then suddenly ask an
unrelated SQuAD question WITH its own current context.

This measures whether different memory construction strategies interfere with
new-context QA, and whether grouping the old conversation into 2 vs 5 large
chunks changes the result.

Arms
----
0. squad_only                    : no long-term memory (clean baseline)
1. python_append_typed_c2        : typed extraction from 2 big chunks + Python append
2. python_append_typed_c5        : typed extraction from 5 big chunks + Python append
3. recursive_typed_limit_c2      : recursive typed summary over 2 big chunks
4. recursive_typed_limit_c5      : recursive typed summary over 5 big chunks

Primary metric
--------------
SQuAD normalized Exact Match.
Also reports token F1 and paired deltas against squad_only.

Important controls
------------------
- SQuAD question/context/gold are NEVER shown to the memory writer.
- The exact same SQuAD example is used for all 5 arms for a given pair.
- Current SQuAD context is always included at QA time.
- Recursive prompt length limit is identical for c2 and c5.
- Writer hard max_tokens is identical for append and recursive arms.
- append means: each chunk -> LLM typed fact extraction, then Python newline append.
- recursive means: previous typed memory + next chunk -> LLM rewrites complete memory.

Expected files in same directory
--------------------------------
longmemeval_s_cleaned.json
dev-v1.1.json

Dependencies
------------
pip install openai tqdm

OpenRouter example
------------------
export VLLM_BASE_URL="https://openrouter.ai/api/v1"
export VLLM_API_KEY="sk-or-v1-..."
export VLLM_MODEL="qwen/qwen3.5-35b-a3b"

Example
-------
python3 memory_squad_chunk_ablation.py \
  --limit 50 \
  --workers 20 \
  --prompt-limit-words 200
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
    "squad_only",
    "python_append_typed_c2",
    "python_append_typed_c5",
    "recursive_typed_limit_c2",
    "recursive_typed_limit_c5",
]

SUMMARY_SYSTEM = """You construct long-term memory from an OLD conversation.
Use only information explicitly supported by that old conversation.
Never infer unstated facts and never use outside knowledge.
Preserve names, entities, dates, quantities, preferences, plans, relationships,
events, and changes over time.
Do not answer any future question. Construct memory only."""

TYPED_EXTRACT_PROMPT = """Extract typed atomic memories from the OLD CONVERSATION CHUNK.

Output ONLY one memory per line in exactly this form:
[TYPE] fact

Allowed TYPE values:
IDENTITY, PREFERENCE, WORK, RELATION, LOCATION, ACTIVITY, PLAN, EVENT, TEMPORAL, OTHER

Rules:
- One explicit fact per line.
- Preserve exact names, entities, dates, quantities, and relationships.
- Preserve state changes and their time when stated.
- Do not speculate.
- Do not answer a future question.

OLD CONVERSATION CHUNK:
{chunk}

TYPED MEMORY:"""

RECURSIVE_TYPED_LIMIT_PROMPT = """Rewrite the EXISTING LONG-TERM MEMORY together with the next OLD CONVERSATION CHUNK as typed atomic memory.

Output ONLY one memory per line in exactly this form:
[TYPE] fact

Allowed TYPE values:
IDENTITY, PREFERENCE, WORK, RELATION, LOCATION, ACTIVITY, PLAN, EVENT, TEMPORAL, OTHER

Rules:
- Retain useful factual information already present in EXISTING LONG-TERM MEMORY.
- Integrate useful factual information from the new OLD CONVERSATION CHUNK.
- Prefer newer information for current state when facts change.
- Preserve historical state/time when it could matter later.
- Deduplicate semantically equivalent facts.
- Preserve exact names, entities, dates, quantities, and relationships.
- Never speculate.
- Keep the COMPLETE updated memory at or below {limit_words} words.
- Do not answer a future question.

EXISTING LONG-TERM MEMORY:
{memory}

NEXT OLD CONVERSATION CHUNK:
{chunk}

UPDATED TYPED MEMORY:"""

QA_SYSTEM = """You answer an extractive reading-comprehension question.

There are two information sources:
1. OLD LONG-TERM MEMORY: background from an unrelated earlier conversation.
2. CURRENT PASSAGE: the passage relevant to the current question.

For the current question, use the CURRENT PASSAGE as the authoritative source.
Do not let unrelated old memory override or distract from the current passage.

Return ONLY the shortest answer span supported by the CURRENT PASSAGE.
Do not explain.
Do not write "Answer:".
If the answer cannot be determined from the CURRENT PASSAGE, output exactly: I don't know."""

QA_USER = """OLD LONG-TERM MEMORY:
{memory}

CURRENT PASSAGE:
{context}

CURRENT QUESTION:
{question}

SHORTEST ANSWER SPAN:"""


# --------------------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------------------

def normalize_answer(s: Any) -> str:
    if s is None:
        return ""
    s = unicodedata.normalize("NFKC", str(s)).lower().strip()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(ch if ch.isalnum() or ch.isspace() else " " for ch in s)
    return " ".join(s.split())


def exact_match(pred: str, golds: Sequence[str]) -> float:
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
    if n == 0:
        return 0.0
    precision = n / len(p)
    recall = n / len(g)
    return 2 * precision * recall / (precision + recall)


def token_f1(pred: str, golds: Sequence[str]) -> float:
    return max(token_f1_single(pred, g) for g in golds)


def clean_answer(x: str) -> str:
    x = (x or "").strip()
    x = re.sub(r"^(final answer|answer)\s*:\s*", "", x, flags=re.I).strip()
    x = x.strip('"').strip("'").strip()
    lines = [z.strip() for z in x.splitlines() if z.strip()]
    return lines[0] if lines else ""


def approx_tokens(text: str) -> int:
    return max(1, math.ceil(len(text) / 4)) if text else 0


# --------------------------------------------------------------------------------------
# OpenAI-compatible client
# --------------------------------------------------------------------------------------

@dataclass
class LLMConfig:
    base_url: str
    api_key: str
    model: str
    timeout: float
    max_retries: int


_thread_local = threading.local()


class VLLM:
    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg

    def _client(self) -> OpenAI:
        if not hasattr(_thread_local, "client"):
            _thread_local.client = OpenAI(
                base_url=self.cfg.base_url,
                api_key=self.cfg.api_key,
                timeout=self.cfg.timeout,
                max_retries=self.cfg.max_retries,
            )
        return _thread_local.client

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

        # OpenRouter + reasoning-capable models: disable unnecessary thinking when supported.
        if "openrouter.ai" in self.cfg.base_url:
            kwargs["extra_body"] = {"reasoning": {"enabled": False}}

        r = self._client().chat.completions.create(**kwargs)
        return (r.choices[0].message.content or "").strip()


# --------------------------------------------------------------------------------------
# Dataset loading
# --------------------------------------------------------------------------------------

def find_file(script_dir: Path, explicit: Optional[str], candidates: Sequence[str], label: str) -> Path:
    if explicit:
        p = Path(explicit)
        if p.exists():
            return p
        raise FileNotFoundError(f"{label} not found: {p}")
    for name in candidates:
        p = script_dir / name
        if p.exists():
            return p
    raise FileNotFoundError(
        f"{label} not found. Put one of these next to this script:\n"
        + "\n".join(f"  - {x}" for x in candidates)
    )


def load_longmemeval(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, list):
        return obj
    for k in ["data", "examples", "items"]:
        if isinstance(obj.get(k), list):
            return obj[k]
    raise ValueError("Unsupported LongMemEval JSON structure")


def load_squad_v11(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)

    out = []
    for article in obj["data"]:
        title = article.get("title", "")
        for para in article["paragraphs"]:
            context = para["context"]
            for qa in para["qas"]:
                answers = [a["text"] for a in qa.get("answers", [])]
                if not answers:
                    continue
                out.append({
                    "id": qa["id"],
                    "title": title,
                    "context": context,
                    "question": qa["question"],
                    "answers": answers,
                })
    return out


def turn_to_text(turn: Dict[str, Any]) -> str:
    role = str(turn.get("role", "unknown")).upper()
    return f"{role}: {turn.get('content', '')}"


def longmem_history(item: Dict[str, Any]) -> str:
    sessions = item.get("haystack_sessions") or []
    dates = item.get("haystack_dates") or [None] * len(sessions)
    blocks = []
    for i, sess in enumerate(sessions):
        date = dates[i] if i < len(dates) else None
        head = f"[SESSION {i+1}"
        if date:
            head += f" | DATE: {date}"
        head += "]"
        body = "\n".join(turn_to_text(t) for t in sess)
        blocks.append(head + "\n" + body)
    return "\n\n".join(blocks)


# --------------------------------------------------------------------------------------
# Chunking
# --------------------------------------------------------------------------------------

def split_text_into_n_big_chunks(text: str, n: int) -> List[str]:
    """
    Deterministic contiguous chunking.
    We split by session blocks when possible, preserving chronology.
    If there are fewer blocks than n, we use as many non-empty chunks as available.
    """
    blocks = [x for x in re.split(r"(?=\[SESSION \d+)", text) if x.strip()]
    if not blocks:
        blocks = [text]

    n = max(1, min(n, len(blocks)))
    groups: List[List[str]] = [[] for _ in range(n)]

    # contiguous, approximately equal number of session blocks
    for i, block in enumerate(blocks):
        g = min(n - 1, (i * n) // len(blocks))
        groups[g].append(block)

    return ["\n\n".join(g) for g in groups if g]


# --------------------------------------------------------------------------------------
# Memory construction
# --------------------------------------------------------------------------------------

def dedupe_exact_lines(lines: Sequence[str]) -> List[str]:
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


def build_python_append(
    llm: VLLM,
    history: str,
    n_chunks: int,
    writer_max_tokens: int,
) -> Tuple[str, int]:
    chunks = split_text_into_n_big_chunks(history, n_chunks)
    all_lines: List[str] = []

    # Independent extraction from each chunk. No previous memory is shown.
    for chunk in chunks:
        x = llm.chat(
            SUMMARY_SYSTEM,
            TYPED_EXTRACT_PROMPT.format(chunk=chunk),
            writer_max_tokens,
        )
        all_lines.extend(x.splitlines())

    memory = "\n".join(dedupe_exact_lines(all_lines))
    return memory, len(chunks)


def build_recursive_limited(
    llm: VLLM,
    history: str,
    n_chunks: int,
    writer_max_tokens: int,
    prompt_limit_words: int,
) -> Tuple[str, int]:
    chunks = split_text_into_n_big_chunks(history, n_chunks)
    memory = "(empty)"

    for chunk in chunks:
        memory = llm.chat(
            SUMMARY_SYSTEM,
            RECURSIVE_TYPED_LIMIT_PROMPT.format(
                memory=memory,
                chunk=chunk,
                limit_words=prompt_limit_words,
            ),
            writer_max_tokens,
        )

    return memory, len(chunks)


# --------------------------------------------------------------------------------------
# QA
# --------------------------------------------------------------------------------------

def answer_squad(
    llm: VLLM,
    memory: str,
    context: str,
    question: str,
    qa_max_tokens: int,
) -> str:
    return clean_answer(
        llm.chat(
            QA_SYSTEM,
            QA_USER.format(
                memory=memory or "(none)",
                context=context,
                question=question,
            ),
            qa_max_tokens,
        )
    )


# --------------------------------------------------------------------------------------
# One paired example
# --------------------------------------------------------------------------------------

def run_one(
    pair_index: int,
    long_item: Dict[str, Any],
    squad_item: Dict[str, Any],
    llm: VLLM,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    t0 = time.perf_counter()
    history = longmem_history(long_item)

    # Build 4 memories. SQuAD content is NOT available here.
    append2, calls_a2 = build_python_append(
        llm, history, 2, args.writer_max_tokens
    )
    append5, calls_a5 = build_python_append(
        llm, history, 5, args.writer_max_tokens
    )
    rec2, calls_r2 = build_recursive_limited(
        llm, history, 2, args.writer_max_tokens, args.prompt_limit_words
    )
    rec5, calls_r5 = build_recursive_limited(
        llm, history, 5, args.writer_max_tokens, args.prompt_limit_words
    )

    memories = {
        "squad_only": "",
        "python_append_typed_c2": append2,
        "python_append_typed_c5": append5,
        "recursive_typed_limit_c2": rec2,
        "recursive_typed_limit_c5": rec5,
    }
    write_calls = {
        "squad_only": 0,
        "python_append_typed_c2": calls_a2,
        "python_append_typed_c5": calls_a5,
        "recursive_typed_limit_c2": calls_r2,
        "recursive_typed_limit_c5": calls_r5,
    }

    method_rows = {}
    for method in METHODS:
        pred = answer_squad(
            llm,
            memories[method],
            squad_item["context"],
            squad_item["question"],
            args.qa_max_tokens,
        )
        method_rows[method] = {
            "prediction": pred,
            "em": exact_match(pred, squad_item["answers"]),
            "f1": token_f1(pred, squad_item["answers"]),
            "memory_tokens": approx_tokens(memories[method]),
            "write_calls": write_calls[method],
            "memory": memories[method] if args.save_memories else None,
        }

    return {
        "pair_index": pair_index,
        "longmemeval_question_id": str(long_item.get("question_id", pair_index)),
        "squad_id": squad_item["id"],
        "squad_title": squad_item["title"],
        "squad_question": squad_item["question"],
        "squad_gold": squad_item["answers"],
        "context_tokens_approx": approx_tokens(squad_item["context"]),
        "methods": method_rows,
        "elapsed_sec": time.perf_counter() - t0,
    }


# --------------------------------------------------------------------------------------
# Aggregation / outputs
# --------------------------------------------------------------------------------------

def mean(xs: Sequence[float]) -> float:
    return statistics.mean(xs) if xs else float("nan")


def flatten_method_rows(results: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for x in results:
        for method, m in x["methods"].items():
            out.append({
                "pair_index": x["pair_index"],
                "longmemeval_question_id": x["longmemeval_question_id"],
                "squad_id": x["squad_id"],
                "squad_title": x["squad_title"],
                "squad_question": x["squad_question"],
                "squad_gold": json.dumps(x["squad_gold"], ensure_ascii=False),
                "method": method,
                "prediction": m["prediction"],
                "em": m["em"],
                "f1": m["f1"],
                "memory_tokens": m["memory_tokens"],
                "write_calls": m["write_calls"],
                "context_tokens_approx": x["context_tokens_approx"],
                "elapsed_sec_pair": x["elapsed_sec"],
            })
    return out


def aggregate(results: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    baseline_em = mean([x["methods"]["squad_only"]["em"] for x in results])
    baseline_f1 = mean([x["methods"]["squad_only"]["f1"] for x in results])

    rows = []
    for method in METHODS:
        ms = [x["methods"][method] for x in results]
        em = mean([m["em"] for m in ms])
        f1 = mean([m["f1"] for m in ms])
        rows.append({
            "method": method,
            "n": len(ms),
            "EM": em,
            "F1": f1,
            "EM_delta_vs_squad_only": em - baseline_em,
            "F1_delta_vs_squad_only": f1 - baseline_f1,
            "Avg_memory_tokens": mean([m["memory_tokens"] for m in ms]),
            "Avg_write_calls": mean([m["write_calls"] for m in ms]),
        })
    return rows


def paired_compare(
    results: Sequence[Dict[str, Any]],
    a: str,
    b: str,
    name: str,
) -> Dict[str, Any]:
    a_em = [x["methods"][a]["em"] for x in results]
    b_em = [x["methods"][b]["em"] for x in results]

    return {
        "comparison": name,
        "A": a,
        "B": b,
        "A_EM": mean(a_em),
        "B_EM": mean(b_em),
        "A_only_correct": sum(x == 1 and y == 0 for x, y in zip(a_em, b_em)),
        "B_only_correct": sum(x == 0 and y == 1 for x, y in zip(a_em, b_em)),
        "both_correct": sum(x == 1 and y == 1 for x, y in zip(a_em, b_em)),
        "both_wrong": sum(x == 0 and y == 0 for x, y in zip(a_em, b_em)),
    }


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def print_summary(rows: Sequence[Dict[str, Any]]) -> None:
    print("\n" + "=" * 124)
    print("SQuAD NEW-QUESTION RESULT")
    print("=" * 124)
    print(
        f"{'method':34s} {'n':>5s} {'EM':>8s} {'F1':>8s} "
        f"{'dEM':>8s} {'dF1':>8s} {'memtok':>10s} {'writes':>8s}"
    )
    print("-" * 124)
    for r in rows:
        print(
            f"{r['method']:34s} {r['n']:5d} "
            f"{r['EM']:8.4f} {r['F1']:8.4f} "
            f"{r['EM_delta_vs_squad_only']:8.4f} "
            f"{r['F1_delta_vs_squad_only']:8.4f} "
            f"{r['Avg_memory_tokens']:10.1f} "
            f"{r['Avg_write_calls']:8.1f}"
        )


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    p.add_argument("--longmemeval", type=str, default=None)
    p.add_argument("--squad", type=str, default=None)
    p.add_argument("--limit", type=int, default=50, help="0 means all possible paired examples")
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--output-dir", type=str, default="results_squad_chunk")

    p.add_argument("--base-url", type=str, default=os.getenv("VLLM_BASE_URL", "http://127.0.0.1:8000/v1"))
    p.add_argument("--api-key", type=str, default=os.getenv("VLLM_API_KEY", "EMPTY"))
    p.add_argument("--model", type=str, default=os.getenv("VLLM_MODEL", ""))
    p.add_argument("--timeout", type=float, default=240.0)
    p.add_argument("--max-retries", type=int, default=3)

    p.add_argument("--writer-max-tokens", type=int, default=768)
    p.add_argument("--prompt-limit-words", type=int, default=200)
    p.add_argument("--qa-max-tokens", type=int, default=32)

    p.add_argument("--save-memories", action="store_true",
                   help="Store full constructed memories in per-example JSONL (large file).")

    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.model:
        raise SystemExit("Set VLLM_MODEL or pass --model")

    script_dir = Path(__file__).resolve().parent
    lm_path = find_file(
        script_dir,
        args.longmemeval,
        ["longmemeval_s_cleaned.json", "longmemeval_s.json"],
        "LongMemEval",
    )
    sq_path = find_file(
        script_dir,
        args.squad,
        ["dev-v1.1.json", "train-v1.1.json"],
        "SQuAD",
    )

    long_data = load_longmemeval(lm_path)
    squad_data = load_squad_v11(sq_path)

    max_n = min(len(long_data) - args.offset, len(squad_data) - args.offset)
    n = max_n if args.limit == 0 else min(args.limit, max_n)

    pairs = [
        (i, long_data[args.offset + i], squad_data[args.offset + i])
        for i in range(n)
    ]

    cfg = LLMConfig(
        base_url=args.base_url,
        api_key=args.api_key,
        model=args.model,
        timeout=args.timeout,
        max_retries=args.max_retries,
    )
    llm = VLLM(cfg)

    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = script_dir / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 124)
    print("LONG-TERM MEMORY -> SUDDEN SQuAD QUESTION / 2-vs-5 CHUNK ABLATION")
    print("=" * 124)
    print(f"LongMemEval       : {lm_path}")
    print(f"SQuAD             : {sq_path}")
    print(f"paired examples   : {n}")
    print(f"model             : {args.model}")
    print(f"endpoint          : {args.base_url}")
    print(f"workers           : {args.workers}")
    print(f"writer hard max   : {args.writer_max_tokens}")
    print(f"recursive limit   : {args.prompt_limit_words} words")
    print()
    print("Arms:")
    print("  squad_only")
    print("  python_append_typed_c2")
    print("  python_append_typed_c5")
    print("  recursive_typed_limit_c2")
    print("  recursive_typed_limit_c5")
    print()
    print("SQuAD CURRENT PASSAGE is included in every QA arm.")
    print("The old LongMemEval memory is intentionally unrelated distractor/background.")
    print("PRIMARY metric: SQuAD normalized Exact Match.")
    print()

    results: List[Dict[str, Any]] = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [
            ex.submit(run_one, idx, lm, sq, llm, args)
            for idx, lm, sq in pairs
        ]
        for fut in tqdm(
            concurrent.futures.as_completed(futs),
            total=len(futs),
            desc="paired examples",
        ):
            try:
                results.append(fut.result())
            except Exception as e:
                print(f"\n[ERROR] {type(e).__name__}: {e}", file=sys.stderr)

    results.sort(key=lambda x: x["pair_index"])

    # Full per-example JSONL
    jsonl_path = out_dir / "examples.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for x in results:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")

    detailed = flatten_method_rows(results)
    summary = aggregate(results)

    comparisons = [
        paired_compare(
            results,
            "python_append_typed_c2",
            "recursive_typed_limit_c2",
            "append_vs_recursive_at_2_chunks",
        ),
        paired_compare(
            results,
            "python_append_typed_c5",
            "recursive_typed_limit_c5",
            "append_vs_recursive_at_5_chunks",
        ),
        paired_compare(
            results,
            "python_append_typed_c5",
            "python_append_typed_c2",
            "append_5_chunks_vs_2_chunks",
        ),
        paired_compare(
            results,
            "recursive_typed_limit_c5",
            "recursive_typed_limit_c2",
            "recursive_5_chunks_vs_2_chunks",
        ),
        paired_compare(
            results,
            "python_append_typed_c2",
            "squad_only",
            "append_2_vs_squad_only",
        ),
        paired_compare(
            results,
            "recursive_typed_limit_c2",
            "squad_only",
            "recursive_2_vs_squad_only",
        ),
    ]

    write_csv(out_dir / "summary.csv", summary)
    write_csv(out_dir / "detailed.csv", detailed)
    write_csv(out_dir / "paired_comparisons.csv", comparisons)

    print_summary(summary)
    print("\nPAIRED COMPARISONS")
    for c in comparisons:
        print(
            f"- {c['comparison']}: "
            f"A_EM={c['A_EM']:.4f} vs B_EM={c['B_EM']:.4f}; "
            f"A_only={c['A_only_correct']} "
            f"B_only={c['B_only_correct']} "
            f"both={c['both_correct']}"
        )

    print(f"\nSaved under: {out_dir}")
    print("  summary.csv")
    print("  detailed.csv")
    print("  paired_comparisons.csv")
    print("  examples.jsonl")


if __name__ == "__main__":
    main()
