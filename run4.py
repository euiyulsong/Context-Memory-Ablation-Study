#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
append_c2_vs_c5_longmemeval.py

Compare:
- python_append_typed_c2
- python_append_typed_c5

on the ORIGINAL LongMemEval QA task.

This isolates chunk granularity for append-only typed memory:
history -> N contiguous big chunks -> LLM typed fact extraction per chunk
-> Python append -> answer original LongMemEval question.

Primary metric: normalized Exact Match
Secondary: token F1, relaxed containment EM

Expected local file:
  longmemeval_s_cleaned.json

Dependencies:
  pip install openai tqdm

OpenRouter:
  export VLLM_BASE_URL="https://openrouter.ai/api/v1"
  export VLLM_API_KEY="sk-or-v1-..."
  export VLLM_MODEL="qwen/qwen3.5-35b-a3b"

Run:
  python3 append_c2_vs_c5_longmemeval.py --limit 50 --workers 20

For final:
  python3 append_c2_vs_c5_longmemeval.py --limit 200 --workers 20
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


METHODS = ["python_append_typed_c2", "python_append_typed_c5"]

SUMMARY_SYSTEM = """You construct long-term memory from a conversation.
Use only information explicitly supported by the conversation.
Never infer unstated facts and never use outside knowledge.
Preserve names, entities, dates, quantities, preferences, plans, relationships,
events, and changes over time.
Do not answer any future question. Construct memory only."""

TYPED_EXTRACT_PROMPT = """Extract typed atomic memories from the CONVERSATION CHUNK.

Output ONLY one memory per line in exactly this form:
[TYPE] fact

Allowed TYPE values:
IDENTITY, PREFERENCE, WORK, RELATION, LOCATION, ACTIVITY, PLAN, EVENT, TEMPORAL, OTHER

Rules:
- One explicit fact per line.
- Preserve exact names, entities, dates, quantities, and relationships.
- Preserve historical facts, not only the latest state.
- If a state changes, preserve both the previous and new state with timing when available.
- Do not merge unrelated facts.
- Do not speculate.
- Do not answer a future question.

CONVERSATION CHUNK:
{chunk}

TYPED MEMORY:"""

QA_SYSTEM = """You answer a factual question using only the supplied long-term memory.

Return ONLY the shortest answer that directly answers the question.
Do not explain your reasoning.
Do not write "Answer:".
Preserve names, dates, quantities, and short phrases exactly when possible.
If the answer cannot be determined from the memory, output exactly: I don't know."""

QA_USER = """LONG-TERM MEMORY:
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


def relaxed_em(pred: str, gold: Any) -> float:
    golds = gold if isinstance(gold, list) else [gold]
    p = normalize_answer(pred)
    for g in golds:
        gg = normalize_answer(g)
        if p == gg:
            return 1.0
        if p and gg and (p in gg or gg in p):
            return 1.0
    return 0.0


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


def token_f1(pred: str, gold: Any) -> float:
    golds = gold if isinstance(gold, list) else [gold]
    return max(token_f1_single(pred, str(g)) for g in golds)


def clean_answer(x: str) -> str:
    x = (x or "").strip()
    x = re.sub(r"^(final answer|answer)\s*:\s*", "", x, flags=re.I).strip()
    x = x.strip('"').strip("'").strip()
    lines = [z.strip() for z in x.splitlines() if z.strip()]
    return lines[0] if lines else ""


def approx_tokens(text: str) -> int:
    return max(1, math.ceil(len(text) / 4)) if text else 0


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
        if "openrouter.ai" in self.cfg.base_url:
            kwargs["extra_body"] = {"reasoning": {"enabled": False}}

        r = self._client().chat.completions.create(**kwargs)
        return (r.choices[0].message.content or "").strip()


def load_longmemeval(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, list):
        return obj
    for k in ["data", "examples", "items"]:
        if isinstance(obj.get(k), list):
            return obj[k]
    raise ValueError("Unsupported LongMemEval JSON structure")


def turn_to_text(turn: Dict[str, Any]) -> str:
    role = str(turn.get("role", "unknown")).upper()
    return f"{role}: {turn.get('content', '')}"


def session_blocks(item: Dict[str, Any]) -> List[str]:
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
    return blocks


def split_blocks_into_n_chunks(blocks: Sequence[str], n: int) -> List[str]:
    if not blocks:
        return [""]
    n = max(1, min(n, len(blocks)))
    groups: List[List[str]] = [[] for _ in range(n)]

    # contiguous, roughly equal number of sessions per chunk
    for i, block in enumerate(blocks):
        g = min(n - 1, (i * n) // len(blocks))
        groups[g].append(block)

    return ["\n\n".join(g) for g in groups if g]


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
    blocks: Sequence[str],
    n_chunks: int,
    writer_max_tokens: int,
) -> Tuple[str, int]:
    chunks = split_blocks_into_n_chunks(blocks, n_chunks)
    all_lines: List[str] = []

    for chunk in chunks:
        x = llm.chat(
            SUMMARY_SYSTEM,
            TYPED_EXTRACT_PROMPT.format(chunk=chunk),
            writer_max_tokens,
        )
        all_lines.extend(x.splitlines())

    memory = "\n".join(dedupe_exact_lines(all_lines))
    return memory, len(chunks)


def answer_question(
    llm: VLLM,
    memory: str,
    question: str,
    qa_max_tokens: int,
) -> str:
    raw = llm.chat(
        QA_SYSTEM,
        QA_USER.format(memory=memory, question=question),
        qa_max_tokens,
    )
    return clean_answer(raw)


def run_one(
    idx: int,
    item: Dict[str, Any],
    llm: VLLM,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    t0 = time.perf_counter()
    blocks = session_blocks(item)

    mem2, calls2 = build_python_append(
        llm, blocks, 2, args.writer_max_tokens
    )
    mem5, calls5 = build_python_append(
        llm, blocks, 5, args.writer_max_tokens
    )

    pred2 = answer_question(
        llm, mem2, str(item["question"]), args.qa_max_tokens
    )
    pred5 = answer_question(
        llm, mem5, str(item["question"]), args.qa_max_tokens
    )

    return {
        "index": idx,
        "question_id": str(item.get("question_id", idx)),
        "question_type": str(item.get("question_type", "unknown")),
        "question": item["question"],
        "gold": item["answer"],
        "elapsed_sec": time.perf_counter() - t0,
        "methods": {
            "python_append_typed_c2": {
                "prediction": pred2,
                "em": exact_match(pred2, item["answer"]),
                "f1": token_f1(pred2, item["answer"]),
                "relaxed_em": relaxed_em(pred2, item["answer"]),
                "memory_tokens": approx_tokens(mem2),
                "write_calls": calls2,
                "memory": mem2 if args.save_memories else None,
            },
            "python_append_typed_c5": {
                "prediction": pred5,
                "em": exact_match(pred5, item["answer"]),
                "f1": token_f1(pred5, item["answer"]),
                "relaxed_em": relaxed_em(pred5, item["answer"]),
                "memory_tokens": approx_tokens(mem5),
                "write_calls": calls5,
                "memory": mem5 if args.save_memories else None,
            },
        },
    }


def mean(xs):
    xs = list(xs)
    return statistics.mean(xs) if xs else float("nan")


def aggregate(results: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for method in METHODS:
        ms = [x["methods"][method] for x in results]
        out.append({
            "method": method,
            "n": len(ms),
            "EM": mean(m["em"] for m in ms),
            "F1": mean(m["f1"] for m in ms),
            "Relaxed_EM": mean(m["relaxed_em"] for m in ms),
            "Avg_memory_tokens": mean(m["memory_tokens"] for m in ms),
            "Avg_write_calls": mean(m["write_calls"] for m in ms),
        })
    return out


def aggregate_by_type(results: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups = collections.defaultdict(list)
    for x in results:
        groups[x["question_type"]].append(x)

    rows = []
    for qt, xs in sorted(groups.items()):
        for method in METHODS:
            ms = [x["methods"][method] for x in xs]
            rows.append({
                "question_type": qt,
                "method": method,
                "n": len(ms),
                "EM": mean(m["em"] for m in ms),
                "F1": mean(m["f1"] for m in ms),
                "Relaxed_EM": mean(m["relaxed_em"] for m in ms),
                "Avg_memory_tokens": mean(m["memory_tokens"] for m in ms),
            })
    return rows


def paired_compare(results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    a = "python_append_typed_c5"
    b = "python_append_typed_c2"
    aem = [x["methods"][a]["em"] for x in results]
    bem = [x["methods"][b]["em"] for x in results]

    return {
        "comparison": "append_c5_vs_c2",
        "A": a,
        "B": b,
        "A_EM": mean(aem),
        "B_EM": mean(bem),
        "A_only_correct": sum(x == 1 and y == 0 for x, y in zip(aem, bem)),
        "B_only_correct": sum(x == 0 and y == 1 for x, y in zip(aem, bem)),
        "both_correct": sum(x == 1 and y == 1 for x, y in zip(aem, bem)),
        "both_wrong": sum(x == 0 and y == 0 for x, y in zip(aem, bem)),
    }


def flatten_details(results: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for x in results:
        for method in METHODS:
            m = x["methods"][method]
            rows.append({
                "index": x["index"],
                "question_id": x["question_id"],
                "question_type": x["question_type"],
                "question": x["question"],
                "gold": json.dumps(x["gold"], ensure_ascii=False),
                "method": method,
                "prediction": m["prediction"],
                "em": m["em"],
                "f1": m["f1"],
                "relaxed_em": m["relaxed_em"],
                "memory_tokens": m["memory_tokens"],
                "write_calls": m["write_calls"],
                "elapsed_sec_pair": x["elapsed_sec"],
            })
    return rows


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data", type=str, default=None)
    p.add_argument("--limit", type=int, default=50, help="0 = all 500")
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--workers", type=int, default=20)
    p.add_argument("--output-dir", type=str, default="results_append_c2_c5")

    p.add_argument("--base-url", type=str, default=os.getenv("VLLM_BASE_URL", "http://127.0.0.1:8000/v1"))
    p.add_argument("--api-key", type=str, default=os.getenv("VLLM_API_KEY", "EMPTY"))
    p.add_argument("--model", type=str, default=os.getenv("VLLM_MODEL", ""))
    p.add_argument("--timeout", type=float, default=240.0)
    p.add_argument("--max-retries", type=int, default=3)

    p.add_argument("--writer-max-tokens", type=int, default=768)
    p.add_argument("--qa-max-tokens", type=int, default=32)
    p.add_argument("--save-memories", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    if not args.model:
        raise SystemExit("Set VLLM_MODEL or pass --model")

    script_dir = Path(__file__).resolve().parent
    data_path = Path(args.data) if args.data else script_dir / "longmemeval_s_cleaned.json"
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    data = load_longmemeval(data_path)
    subset = data[args.offset:]
    if args.limit > 0:
        subset = subset[:args.limit]

    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = script_dir / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    llm = VLLM(LLMConfig(
        base_url=args.base_url,
        api_key=args.api_key,
        model=args.model,
        timeout=args.timeout,
        max_retries=args.max_retries,
    ))

    print("=" * 116)
    print("LONGMEMEVAL — PYTHON APPEND TYPED: 2 CHUNKS vs 5 CHUNKS")
    print("=" * 116)
    print(f"dataset          : {data_path}")
    print(f"examples         : {len(subset)}")
    print(f"model            : {args.model}")
    print(f"endpoint         : {args.base_url}")
    print(f"workers          : {args.workers}")
    print(f"writer hard max  : {args.writer_max_tokens}")
    print(f"PRIMARY metric   : strict normalized EM")
    print()
    print("c2: history -> 2 large contiguous chunks -> typed extract each -> Python append")
    print("c5: history -> 5 large contiguous chunks -> typed extract each -> Python append")
    print("Original LongMemEval question/answer is used for evaluation.")
    print()

    results = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [
            ex.submit(run_one, args.offset + i, item, llm, args)
            for i, item in enumerate(subset)
        ]
        for fut in tqdm(concurrent.futures.as_completed(futs), total=len(futs), desc="examples"):
            try:
                results.append(fut.result())
            except Exception as e:
                print(f"\n[ERROR] {type(e).__name__}: {e}", file=sys.stderr)

    results.sort(key=lambda x: x["index"])

    summary = aggregate(results)
    by_type = aggregate_by_type(results)
    paired = paired_compare(results)
    details = flatten_details(results)

    with (out_dir / "examples.jsonl").open("w", encoding="utf-8") as f:
        for x in results:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")

    write_csv(out_dir / "summary.csv", summary)
    write_csv(out_dir / "by_question_type.csv", by_type)
    write_csv(out_dir / "paired_comparison.csv", [paired])
    write_csv(out_dir / "details.csv", details)

    print("\n" + "=" * 116)
    print("RESULT")
    print("=" * 116)
    print(f"{'method':30s} {'n':>5s} {'EM':>8s} {'F1':>8s} {'RelEM':>8s} {'memtok':>10s} {'writes':>8s}")
    print("-" * 116)
    for r in summary:
        print(
            f"{r['method']:30s} {r['n']:5d} "
            f"{r['EM']:8.4f} {r['F1']:8.4f} {r['Relaxed_EM']:8.4f} "
            f"{r['Avg_memory_tokens']:10.1f} {r['Avg_write_calls']:8.1f}"
        )

    print("\nPAIRED")
    print(
        f"c5_EM={paired['A_EM']:.4f} vs c2_EM={paired['B_EM']:.4f}; "
        f"c5_only={paired['A_only_correct']} "
        f"c2_only={paired['B_only_correct']} "
        f"both={paired['both_correct']} "
        f"both_wrong={paired['both_wrong']}"
    )

    print(f"\nSaved under: {out_dir}")
    print("  summary.csv")
    print("  by_question_type.csv")
    print("  paired_comparison.csv")
    print("  details.csv")
    print("  examples.jsonl")


if __name__ == "__main__":
    main()
