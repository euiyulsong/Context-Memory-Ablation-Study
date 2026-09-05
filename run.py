#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
memory_ablation.py

LongMemEval-S memory representation / update ablation using an OpenAI-compatible vLLM endpoint.

Main goals
----------
1) Compare memory representation:
   - current_only
   - last_k
   - full_context
   - recursive_paragraph
   - recursive_list
   - recursive_typed_list
   - append_typed
   - update_typed
   - turn_update_typed
   - session_update_typed
   - atomic_fact
   - segment_summary
   - hybrid_fact_segment

2) Compare summary format:
   paragraph vs flat list vs typed list

3) Compare update strategy:
   recursive rewrite vs append-only vs update-in-place

4) Evaluate with normalized Exact Match (primary) + token F1 + relaxed containment diagnostic.

Important
---------
- Memory construction prompts NEVER see the evaluation question or gold answer.
- Only final retrieval/QA sees the question.
- QA prompt forces a short answer with no explanation, making EM more meaningful.
- All methods share the same vLLM endpoint/model/temperature and a common final memory budget.
- The official LongMemEval evaluator is LLM-judge based; this script intentionally adds strict EM/F1
  for controlled ablations.

Expected local dataset file
---------------------------
Place ONE of these in the same directory as this script:
  longmemeval_s_cleaned.json   (recommended)
  longmemeval_s.json
  longmemeval_oracle.json

Dependencies
------------
pip install openai tqdm

Example
-------
export VLLM_BASE_URL=http://127.0.0.1:8000/v1
export VLLM_API_KEY=EMPTY
export VLLM_MODEL=Qwen/Qwen3-8B

python memory_ablation.py --limit 50 \
  --methods current_only,last_k,recursive_paragraph,recursive_list,recursive_typed_list,append_typed,update_typed

Resume-safe:
- Per-example outputs are appended to results/<method>.jsonl
- rerunning skips completed question_ids unless --overwrite is used.
"""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import math
import os
import re
import statistics
import sys
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from tqdm import tqdm

try:
    from openai import OpenAI
except ImportError:
    print("Missing dependency: openai. Run: pip install openai tqdm", file=sys.stderr)
    raise


# ======================================================================================
# Configuration
# ======================================================================================

ALL_METHODS = [
    "current_only",
    "last_k",
    "full_context",
    "recursive_paragraph",
    "recursive_list",
    "recursive_typed_list",
    "append_typed",
    "update_typed",
    "turn_update_typed",
    "session_update_typed",
    "atomic_fact",
    "segment_summary",
    "hybrid_fact_segment",
]

DATA_CANDIDATES = [
    "longmemeval_s_cleaned.json",
    "longmemeval_s.json",
    "longmemeval_oracle.json",
]

TYPED_CATEGORIES = [
    "IDENTITY",
    "PREFERENCE",
    "WORK",
    "RELATION",
    "LOCATION",
    "ACTIVITY",
    "PLAN",
    "EVENT",
    "TEMPORAL",
    "OTHER",
]

DEFAULT_MEMORY_TOKEN_BUDGET = 1024
DEFAULT_QA_MAX_TOKENS = 64
DEFAULT_SUMMARY_MAX_TOKENS = 512
DEFAULT_LAST_K_TURNS = 12
DEFAULT_RETRIEVAL_TOP_K = 8


# ======================================================================================
# Text / metric utilities
# ======================================================================================

def normalize_answer(s: Any) -> str:
    """
    SQuAD-style normalization:
    lowercase -> unicode normalize -> punctuation removal -> article removal -> whitespace fix.
    """
    if s is None:
        return ""
    s = str(s)
    s = unicodedata.normalize("NFKC", s).lower().strip()

    # English articles because LongMemEval is English.
    s = re.sub(r"\b(a|an|the)\b", " ", s)

    # punctuation -> spaces (keeps alphanumeric)
    s = "".join(ch if ch.isalnum() or ch.isspace() else " " for ch in s)
    s = " ".join(s.split())
    return s


def exact_match(pred: str, gold: Any) -> float:
    golds = gold if isinstance(gold, list) else [gold]
    npred = normalize_answer(pred)
    return float(any(npred == normalize_answer(g) for g in golds))


def relaxed_containment_em(pred: str, gold: Any) -> float:
    """
    Diagnostic only, NOT the primary metric.
    Returns 1 when normalized gold is contained in pred or vice versa.
    Useful to see how much strict EM is punished by formatting.
    """
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
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(p)
    recall = num_same / len(g)
    return 2 * precision * recall / (precision + recall)


def token_f1(pred: str, gold: Any) -> float:
    golds = gold if isinstance(gold, list) else [gold]
    return max(token_f1_single(pred, str(g)) for g in golds)


def clean_model_answer(text: str) -> str:
    """
    Keep model outputs EM-friendly without changing semantic content.
    """
    if text is None:
        return ""
    x = text.strip()

    # Strip common answer wrappers only.
    x = re.sub(r"^(final answer|answer)\s*:\s*", "", x, flags=re.I)
    x = x.strip().strip('"').strip("'").strip()

    # If model ignored instruction and emitted multiple lines, keep first non-empty line.
    lines = [z.strip() for z in x.splitlines() if z.strip()]
    if lines:
        x = lines[0]

    return x.strip()


def approx_tokens(text: str) -> int:
    """
    Fast tokenizer-free approximation.
    English chat: ~4 chars/token is a rough estimate.
    """
    if not text:
        return 0
    return max(1, math.ceil(len(text) / 4.0))


def truncate_to_approx_tokens(text: str, max_tokens: int) -> str:
    if approx_tokens(text) <= max_tokens:
        return text
    max_chars = max_tokens * 4
    # preserve tail + head because knowledge updates often occur late
    head = int(max_chars * 0.35)
    tail = max_chars - head
    return text[:head] + "\n...[TRUNCATED]...\n" + text[-tail:]


# ======================================================================================
# BM25-lite retrieval (stdlib only)
# ======================================================================================

_WORD_RE = re.compile(r"[A-Za-z0-9]+")


def lexical_tokens(text: str) -> List[str]:
    return _WORD_RE.findall(text.lower())


def bm25_scores(query: str, docs: Sequence[str], k1: float = 1.5, b: float = 0.75) -> List[float]:
    """
    Minimal BM25 implementation so company/offline machines need no sklearn/rank_bm25.
    """
    q_terms = lexical_tokens(query)
    tokenized = [lexical_tokens(d) for d in docs]
    n = len(docs)
    if n == 0:
        return []

    avgdl = sum(len(d) for d in tokenized) / max(n, 1)
    df = collections.Counter()
    for d in tokenized:
        for term in set(d):
            df[term] += 1

    scores = []
    for d in tokenized:
        tf = collections.Counter(d)
        dl = len(d)
        s = 0.0
        for term in q_terms:
            if term not in tf:
                continue
            freq = tf[term]
            n_q = df[term]
            idf = math.log(1.0 + (n - n_q + 0.5) / (n_q + 0.5))
            denom = freq + k1 * (1 - b + b * dl / max(avgdl, 1e-9))
            s += idf * (freq * (k1 + 1)) / max(denom, 1e-9)
        scores.append(s)
    return scores


def retrieve_units(query: str, units: Sequence[str], top_k: int, token_budget: int) -> Tuple[str, List[int]]:
    """
    BM25-rank memory units, then greedily fit into a shared budget.
    """
    if not units:
        return "", []

    scores = bm25_scores(query, units)
    ranked = sorted(range(len(units)), key=lambda i: scores[i], reverse=True)

    selected = []
    used = 0
    for i in ranked[: max(top_k * 3, top_k)]:
        t = approx_tokens(units[i])
        if selected and used + t > token_budget:
            continue
        selected.append(i)
        used += t
        if len(selected) >= top_k or used >= token_budget:
            break

    if not selected:
        selected = [ranked[0]]

    # Reorder chronologically after retrieval when ids correspond to chronology.
    selected_ordered = sorted(selected)
    text = "\n\n".join(units[i] for i in selected_ordered)
    return truncate_to_approx_tokens(text, token_budget), selected_ordered


# ======================================================================================
# vLLM/OpenAI-compatible client
# ======================================================================================

@dataclass
class LLMConfig:
    base_url: str
    api_key: str
    model: str
    temperature: float
    timeout: float
    max_retries: int


class VLLM:
    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg
        self.client = OpenAI(
            base_url=cfg.base_url,
            api_key=cfg.api_key,
            timeout=cfg.timeout,
            max_retries=cfg.max_retries,
        )

    def chat(
        self,
        system: str,
        user: str,
        max_tokens: int,
        temperature: Optional[float] = None,
    ) -> str:
        temp = self.cfg.temperature if temperature is None else temperature
        r = self.client.chat.completions.create(
            model=self.cfg.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temp,
            max_tokens=max_tokens,
        )
        return (r.choices[0].message.content or "").strip()


# ======================================================================================
# LongMemEval data adapters
# ======================================================================================

def find_dataset(script_dir: Path, explicit: Optional[str]) -> Path:
    if explicit:
        p = Path(explicit)
        if p.exists():
            return p
        raise FileNotFoundError(f"--data not found: {p}")

    for name in DATA_CANDIDATES:
        p = script_dir / name
        if p.exists():
            return p

    raise FileNotFoundError(
        "Dataset not found. Put one of these next to memory_ablation.py:\n"
        + "\n".join(f"  - {x}" for x in DATA_CANDIDATES)
    )


def load_dataset(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)

    if isinstance(obj, list):
        data = obj
    elif isinstance(obj, dict):
        # Be permissive for wrappers.
        for key in ["data", "examples", "items"]:
            if isinstance(obj.get(key), list):
                data = obj[key]
                break
        else:
            raise ValueError(f"Unsupported dataset wrapper keys: {list(obj)[:20]}")
    else:
        raise ValueError("Dataset JSON must be a list or object containing a list.")

    required = ["question_id", "question", "answer", "haystack_sessions"]
    for k in required:
        if data and k not in data[0]:
            raise ValueError(f"Dataset seems incompatible; missing field: {k}")
    return data


def turn_to_text(turn: Dict[str, Any]) -> str:
    role = str(turn.get("role", "unknown")).upper()
    content = str(turn.get("content", ""))
    return f"{role}: {content}"


def session_to_text(session: Sequence[Dict[str, Any]], date: Optional[str] = None) -> str:
    body = "\n".join(turn_to_text(t) for t in session)
    return f"[DATE: {date}]\n{body}" if date else body


def get_sessions(item: Dict[str, Any]) -> List[Tuple[Optional[str], List[Dict[str, Any]]]]:
    sessions = item.get("haystack_sessions") or []
    dates = item.get("haystack_dates") or [None] * len(sessions)
    if len(dates) < len(sessions):
        dates = list(dates) + [None] * (len(sessions) - len(dates))
    return [(dates[i], sessions[i]) for i in range(len(sessions))]


def flatten_turns(item: Dict[str, Any]) -> List[str]:
    out = []
    for date, sess in get_sessions(item):
        for t in sess:
            prefix = f"[DATE: {date}] " if date else ""
            out.append(prefix + turn_to_text(t))
    return out


# ======================================================================================
# Prompt templates
# ======================================================================================

SUMMARY_SYSTEM = """You are a memory compressor for a long-term conversational assistant.
Your job is to preserve only information explicitly supported by the conversation.
Never infer unstated facts. Never use knowledge outside the conversation.
Preserve names, entities, dates, quantities, preferences, plans, relationships, and changes over time.
When newer information supersedes older information, preserve the latest state and, when useful, the prior state with its time.
Do not answer any future question; you are only constructing memory."""

PARAGRAPH_PROMPT = """Compress the NEW CONVERSATION into a concise factual paragraph for future memory.
Preserve details that could matter in a later factual question.
No headings, no bullets, no speculation.

NEW CONVERSATION:
{chunk}

PARAGRAPH MEMORY:"""

LIST_PROMPT = """Extract concise atomic memory items from the NEW CONVERSATION.
Use one factual item per line beginning with "- ".
Preserve exact names, dates, quantities and relations.
Do not merge unrelated facts. Do not speculate.

NEW CONVERSATION:
{chunk}

MEMORY LIST:"""

TYPED_PROMPT = """Extract atomic long-term memories from the NEW CONVERSATION.
Output ONLY lines in exactly this format:
[TYPE] fact

Allowed TYPE values:
IDENTITY, PREFERENCE, WORK, RELATION, LOCATION, ACTIVITY, PLAN, EVENT, TEMPORAL, OTHER

Rules:
- One fact per line.
- Preserve exact names, dates, quantities and relationships.
- If the content describes a change, state both the new value and time when available.
- No explanations and no markdown bullets.

NEW CONVERSATION:
{chunk}

TYPED MEMORY:"""

SEGMENT_PROMPT = """Summarize this conversation session as a compact, self-contained memory segment.
Keep the chronology and relationships between facts. Preserve exact entities, dates, quantities,
preferences, decisions, plans, and state changes. Omit greetings and low-value chit-chat.
Do not infer anything not stated.

SESSION:
{chunk}

SEGMENT MEMORY:"""

RECURSIVE_PARAGRAPH_PROMPT = """Rewrite the EXISTING MEMORY together with the NEW CONVERSATION into one compact factual paragraph.

Requirements:
- Retain all still-useful factual details from EXISTING MEMORY.
- Integrate new information.
- When a fact is updated or contradicted by newer information, prefer the newest state but preserve prior state/time if relevant.
- Do not invent.
- No bullets or headings.
- Keep the result concise.

EXISTING MEMORY:
{memory}

NEW CONVERSATION:
{chunk}

UPDATED PARAGRAPH MEMORY:"""

RECURSIVE_LIST_PROMPT = """Rewrite the EXISTING MEMORY together with the NEW CONVERSATION as a compact atomic list.

Requirements:
- One fact per line beginning with "- ".
- Retain useful existing facts.
- Integrate new facts.
- Resolve updates in favor of the latest state; keep old state/time only when relevant.
- Deduplicate semantically equivalent facts.
- Preserve names, dates and quantities.
- No speculation.

EXISTING MEMORY:
{memory}

NEW CONVERSATION:
{chunk}

UPDATED MEMORY LIST:"""

RECURSIVE_TYPED_PROMPT = """Rewrite the EXISTING MEMORY together with the NEW CONVERSATION as typed atomic memory.

Output ONLY:
[TYPE] fact

Allowed TYPE:
IDENTITY, PREFERENCE, WORK, RELATION, LOCATION, ACTIVITY, PLAN, EVENT, TEMPORAL, OTHER

Requirements:
- Retain useful existing memories.
- Integrate new information.
- Resolve changes in favor of the newest state.
- Preserve earlier state/time if it is needed to understand a change.
- Deduplicate.
- Preserve exact entities, dates and quantities.
- Never speculate.

EXISTING MEMORY:
{memory}

NEW CONVERSATION:
{chunk}

UPDATED TYPED MEMORY:"""

UPDATE_TYPED_PROMPT = """Update the existing typed memory using the new conversation.

Return ONLY the complete updated memory, using:
[TYPE] fact

Allowed TYPE:
IDENTITY, PREFERENCE, WORK, RELATION, LOCATION, ACTIVITY, PLAN, EVENT, TEMPORAL, OTHER

Rules:
1. ADD new durable facts.
2. UPDATE a memory when newer information changes the same attribute/state.
3. REMOVE stale duplicates.
4. Keep historical states only when their timing is explicitly useful.
5. Preserve exact names, dates, quantities and relationships.
6. Never infer unstated information.
7. Keep unrelated existing memories unchanged.
8. No commentary.

EXISTING MEMORY:
{memory}

NEW CONVERSATION:
{chunk}

UPDATED MEMORY:"""

QA_SYSTEM = """You answer factual questions using only the supplied memory/context.
Return ONLY the shortest answer span that directly answers the question.
Do not explain your reasoning.
Do not write 'Answer:'.
Do not use a full sentence unless a full sentence is strictly necessary.
Preserve names, dates and quantities exactly when possible.
If the information is not present or cannot be determined from the memory, output exactly: I don't know."""

QA_USER = """MEMORY / CONTEXT:
{memory}

QUESTION:
{question}

SHORTEST ANSWER:"""


# ======================================================================================
# Memory builders
# ======================================================================================

def llm_extract(llm: VLLM, prompt: str, chunk: str, max_tokens: int) -> str:
    return llm.chat(
        SUMMARY_SYSTEM,
        prompt.format(chunk=chunk),
        max_tokens=max_tokens,
        temperature=0.0,
    ).strip()


def build_recursive(
    llm: VLLM,
    chunks: Sequence[str],
    prompt: str,
    step_max_tokens: int,
    running_budget: int,
) -> str:
    memory = "(empty)"
    for chunk in chunks:
        user = prompt.format(memory=memory, chunk=chunk)
        memory = llm.chat(SUMMARY_SYSTEM, user, max_tokens=step_max_tokens, temperature=0.0).strip()
        memory = truncate_to_approx_tokens(memory, running_budget)
    return memory if memory != "(empty)" else ""


def build_append_typed(
    llm: VLLM,
    chunks: Sequence[str],
    step_max_tokens: int,
) -> List[str]:
    units: List[str] = []
    for chunk in chunks:
        x = llm_extract(llm, TYPED_PROMPT, chunk, step_max_tokens)
        units.extend([line.strip() for line in x.splitlines() if line.strip()])
    return dedupe_lines(units)


def dedupe_lines(lines: Sequence[str]) -> List[str]:
    seen = set()
    out = []
    for line in lines:
        key = normalize_answer(line)
        if key and key not in seen:
            seen.add(key)
            out.append(line)
    return out


def build_update_typed(
    llm: VLLM,
    chunks: Sequence[str],
    step_max_tokens: int,
    running_budget: int,
) -> str:
    memory = "(empty)"
    for chunk in chunks:
        user = UPDATE_TYPED_PROMPT.format(memory=memory, chunk=chunk)
        memory = llm.chat(SUMMARY_SYSTEM, user, max_tokens=step_max_tokens, temperature=0.0).strip()
        memory = truncate_to_approx_tokens(memory, running_budget)
    return memory if memory != "(empty)" else ""


def sessions_as_chunks(item: Dict[str, Any]) -> List[str]:
    return [session_to_text(sess, date) for date, sess in get_sessions(item)]


def turns_as_chunks(item: Dict[str, Any]) -> List[str]:
    return flatten_turns(item)


def build_memory_units(
    method: str,
    item: Dict[str, Any],
    llm: VLLM,
    step_max_tokens: int,
    running_budget: int,
) -> Tuple[List[str], Dict[str, Any]]:
    """
    Returns memory units + metadata.
    For monolithic methods, units contains one memory string.
    Crucially, item['question'] and item['answer'] are never passed into any construction prompt.
    """
    sessions = sessions_as_chunks(item)
    turns = turns_as_chunks(item)

    meta: Dict[str, Any] = {"write_calls_estimate": 0}

    if method == "current_only":
        return [], meta

    if method == "last_k":
        return turns, meta

    if method == "full_context":
        return sessions, meta

    if method == "recursive_paragraph":
        meta["write_calls_estimate"] = len(sessions)
        mem = build_recursive(llm, sessions, RECURSIVE_PARAGRAPH_PROMPT, step_max_tokens, running_budget)
        return [mem], meta

    if method == "recursive_list":
        meta["write_calls_estimate"] = len(sessions)
        mem = build_recursive(llm, sessions, RECURSIVE_LIST_PROMPT, step_max_tokens, running_budget)
        return [mem], meta

    if method == "recursive_typed_list":
        meta["write_calls_estimate"] = len(sessions)
        mem = build_recursive(llm, sessions, RECURSIVE_TYPED_PROMPT, step_max_tokens, running_budget)
        return [mem], meta

    if method == "append_typed":
        meta["write_calls_estimate"] = len(sessions)
        return build_append_typed(llm, sessions, step_max_tokens), meta

    if method == "update_typed":
        meta["write_calls_estimate"] = len(sessions)
        mem = build_update_typed(llm, sessions, step_max_tokens, running_budget)
        return [mem], meta

    if method == "session_update_typed":
        meta["write_calls_estimate"] = len(sessions)
        mem = build_update_typed(llm, sessions, step_max_tokens, running_budget)
        return [mem], meta

    if method == "turn_update_typed":
        meta["write_calls_estimate"] = len(turns)
        mem = build_update_typed(llm, turns, step_max_tokens, running_budget)
        return [mem], meta

    if method == "atomic_fact":
        meta["write_calls_estimate"] = len(sessions)
        return build_append_typed(llm, sessions, step_max_tokens), meta

    if method == "segment_summary":
        meta["write_calls_estimate"] = len(sessions)
        units = []
        for s in sessions:
            units.append(llm_extract(llm, SEGMENT_PROMPT, s, step_max_tokens))
        return units, meta

    if method == "hybrid_fact_segment":
        # One typed extraction + one segment summary per session.
        meta["write_calls_estimate"] = len(sessions) * 2
        units = []
        for s in sessions:
            fact = llm_extract(llm, TYPED_PROMPT, s, step_max_tokens)
            seg = llm_extract(llm, SEGMENT_PROMPT, s, step_max_tokens)
            units.append("[ATOMIC FACTS]\n" + fact + "\n[SEGMENT]\n" + seg)
        return units, meta

    raise ValueError(f"Unknown method: {method}")


def select_final_memory(
    method: str,
    units: Sequence[str],
    question: str,
    token_budget: int,
    retrieval_top_k: int,
    last_k_turns: int,
) -> Tuple[str, List[int]]:
    if method == "current_only":
        return "(no previous memory)", []

    if method == "last_k":
        chosen = list(units[-last_k_turns:])
        txt = "\n".join(chosen)
        return truncate_to_approx_tokens(txt, token_budget), list(range(max(0, len(units)-len(chosen)), len(units)))

    if method == "full_context":
        # Deliberately no retrieval: this is a budgeted full-history baseline.
        # If history is larger than budget, head+tail truncation is used.
        txt = "\n\n".join(units)
        return truncate_to_approx_tokens(txt, token_budget), list(range(len(units)))

    if method in {
        "recursive_paragraph",
        "recursive_list",
        "recursive_typed_list",
        "update_typed",
        "turn_update_typed",
        "session_update_typed",
    }:
        txt = "\n\n".join(units)
        return truncate_to_approx_tokens(txt, token_budget), list(range(len(units)))

    # Retrieval-based representations.
    return retrieve_units(question, units, retrieval_top_k, token_budget)


# ======================================================================================
# Evaluation
# ======================================================================================

def answer_question(llm: VLLM, memory: str, question: str, max_tokens: int) -> str:
    raw = llm.chat(
        QA_SYSTEM,
        QA_USER.format(memory=memory, question=question),
        max_tokens=max_tokens,
        temperature=0.0,
    )
    return clean_model_answer(raw)


def qtype(item: Dict[str, Any]) -> str:
    return str(item.get("question_type", "unknown"))


def stable_subset(data: List[Dict[str, Any]], limit: Optional[int], offset: int) -> List[Dict[str, Any]]:
    data = data[offset:]
    if limit is None or limit <= 0:
        return data
    return data[:limit]


def load_completed(path: Path) -> Dict[str, Dict[str, Any]]:
    done = {}
    if not path.exists():
        return done
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                done[str(row["question_id"])] = row
            except Exception:
                continue
    return done


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def run_method(
    method: str,
    data: Sequence[Dict[str, Any]],
    llm: VLLM,
    out_dir: Path,
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    out_path = out_dir / f"{method}.jsonl"

    if args.overwrite and out_path.exists():
        out_path.unlink()

    completed = load_completed(out_path)
    rows: List[Dict[str, Any]] = list(completed.values())

    bar = tqdm(data, desc=method)
    for item in bar:
        qid = str(item["question_id"])
        if qid in completed:
            continue

        t0 = time.perf_counter()
        error = None

        try:
            units, meta = build_memory_units(
                method=method,
                item=item,
                llm=llm,
                step_max_tokens=args.summary_max_tokens,
                running_budget=args.running_memory_tokens,
            )

            memory, selected_ids = select_final_memory(
                method=method,
                units=units,
                question=str(item["question"]),
                token_budget=args.memory_tokens,
                retrieval_top_k=args.retrieval_top_k,
                last_k_turns=args.last_k_turns,
            )

            pred = answer_question(
                llm=llm,
                memory=memory,
                question=str(item["question"]),
                max_tokens=args.qa_max_tokens,
            )

            em = exact_match(pred, item["answer"])
            f1 = token_f1(pred, item["answer"])
            rem = relaxed_containment_em(pred, item["answer"])

        except Exception as e:
            memory = ""
            pred = ""
            em = f1 = rem = 0.0
            selected_ids = []
            meta = {"write_calls_estimate": None}
            error = f"{type(e).__name__}: {e}"

        elapsed = time.perf_counter() - t0

        row = {
            "method": method,
            "question_id": qid,
            "question_type": qtype(item),
            "question": item["question"],
            "gold": item["answer"],
            "prediction": pred,
            "em": em,
            "f1": f1,
            "relaxed_em": rem,
            "memory_approx_tokens": approx_tokens(memory),
            "selected_unit_ids": selected_ids,
            "write_calls_estimate": meta.get("write_calls_estimate"),
            "elapsed_sec": elapsed,
            "error": error,
        }
        append_jsonl(out_path, row)
        completed[qid] = row
        rows.append(row)

        bar.set_postfix(
            EM=f"{statistics.mean([x['em'] for x in completed.values()]):.3f}",
            n=len(completed),
        )

    # Keep only current requested question set for summary.
    wanted = {str(x["question_id"]) for x in data}
    return [r for q, r in completed.items() if q in wanted]


def mean_or_nan(xs: Sequence[float]) -> float:
    vals = [float(x) for x in xs if x is not None]
    return statistics.mean(vals) if vals else float("nan")


def aggregate_rows(method: str, rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    ok = [r for r in rows if not r.get("error")]
    return {
        "method": method,
        "n": len(rows),
        "n_ok": len(ok),
        "EM": mean_or_nan([r["em"] for r in rows]),
        "F1": mean_or_nan([r["f1"] for r in rows]),
        "Relaxed_EM": mean_or_nan([r["relaxed_em"] for r in rows]),
        "Avg_memory_tokens": mean_or_nan([r["memory_approx_tokens"] for r in rows]),
        "Avg_elapsed_sec": mean_or_nan([r["elapsed_sec"] for r in rows]),
        "Avg_write_calls": mean_or_nan([
            r["write_calls_estimate"] for r in rows if r.get("write_calls_estimate") is not None
        ]),
        "errors": sum(bool(r.get("error")) for r in rows),
    }


def aggregate_by_type(method: str, rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
    for r in rows:
        grouped[r["question_type"]].append(r)

    out = []
    for qt, rs in sorted(grouped.items()):
        out.append({
            "method": method,
            "question_type": qt,
            "n": len(rs),
            "EM": mean_or_nan([x["em"] for x in rs]),
            "F1": mean_or_nan([x["f1"] for x in rs]),
            "Relaxed_EM": mean_or_nan([x["relaxed_em"] for x in rs]),
        })
    return out


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def print_summary(summary: Sequence[Dict[str, Any]]) -> None:
    print("\n" + "=" * 118)
    print("MEMORY ABLATION RESULT")
    print("=" * 118)
    header = f"{'method':28s} {'n':>5s} {'EM':>8s} {'F1':>8s} {'RelEM':>8s} {'memtok':>9s} {'sec/q':>9s} {'writes':>9s} {'err':>5s}"
    print(header)
    print("-" * len(header))
    for r in sorted(summary, key=lambda x: x["EM"], reverse=True):
        print(
            f"{r['method'][:28]:28s} "
            f"{r['n']:5d} "
            f"{r['EM']:8.4f} "
            f"{r['F1']:8.4f} "
            f"{r['Relaxed_EM']:8.4f} "
            f"{r['Avg_memory_tokens']:9.1f} "
            f"{r['Avg_elapsed_sec']:9.2f} "
            f"{r['Avg_write_calls']:9.1f} "
            f"{r['errors']:5d}"
        )


# ======================================================================================
# CLI
# ======================================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    p.add_argument("--data", type=str, default=None, help="Path to LongMemEval JSON. Auto-detects same dir if omitted.")
    p.add_argument("--output-dir", type=str, default="results")
    p.add_argument("--methods", type=str, default="current_only,last_k,recursive_paragraph,recursive_list,recursive_typed_list,append_typed,update_typed")
    p.add_argument("--limit", type=int, default=50, help="0 means all examples.")
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--overwrite", action="store_true")

    p.add_argument("--base-url", type=str, default=os.getenv("VLLM_BASE_URL", "http://127.0.0.1:8000/v1"))
    p.add_argument("--api-key", type=str, default=os.getenv("VLLM_API_KEY", "EMPTY"))
    p.add_argument("--model", type=str, default=os.getenv("VLLM_MODEL", ""))
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--timeout", type=float, default=180.0)
    p.add_argument("--max-retries", type=int, default=3)

    p.add_argument("--memory-tokens", type=int, default=DEFAULT_MEMORY_TOKEN_BUDGET,
                   help="Common FINAL memory/context budget for fair comparison.")
    p.add_argument("--running-memory-tokens", type=int, default=1536,
                   help="Safety cap while recursively rewriting/updating memory.")
    p.add_argument("--summary-max-tokens", type=int, default=DEFAULT_SUMMARY_MAX_TOKENS,
                   help="Per memory-write generation cap.")
    p.add_argument("--qa-max-tokens", type=int, default=DEFAULT_QA_MAX_TOKENS)
    p.add_argument("--last-k-turns", type=int, default=DEFAULT_LAST_K_TURNS)
    p.add_argument("--retrieval-top-k", type=int, default=DEFAULT_RETRIEVAL_TOP_K)

    return p.parse_args()


def validate_methods(s: str) -> List[str]:
    methods = [x.strip() for x in s.split(",") if x.strip()]
    bad = [x for x in methods if x not in ALL_METHODS]
    if bad:
        raise ValueError(f"Unknown methods: {bad}\nAvailable: {ALL_METHODS}")
    return methods


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    data_path = find_dataset(script_dir, args.data)

    if not args.model:
        raise SystemExit(
            "Model name missing.\n"
            "Set VLLM_MODEL or pass --model, e.g.\n"
            "  export VLLM_MODEL=Qwen/Qwen3-8B"
        )

    methods = validate_methods(args.methods)
    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = script_dir / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_dataset(data_path)
    subset = stable_subset(data, None if args.limit == 0 else args.limit, args.offset)

    cfg = LLMConfig(
        base_url=args.base_url,
        api_key=args.api_key,
        model=args.model,
        temperature=args.temperature,
        timeout=args.timeout,
        max_retries=args.max_retries,
    )
    llm = VLLM(cfg)

    print("=" * 118)
    print("LONGMEMEVAL MEMORY ABLATION")
    print("=" * 118)
    print(f"dataset       : {data_path}")
    print(f"dataset size  : {len(data)}")
    print(f"run examples  : {len(subset)} (offset={args.offset})")
    print(f"methods       : {methods}")
    print(f"vLLM endpoint : {args.base_url}")
    print(f"model         : {args.model}")
    print(f"memory budget : {args.memory_tokens} approx tokens")
    print(f"temperature   : {args.temperature}")
    print()
    print("NOTE: strict normalized EM is the PRIMARY metric.")
    print("      F1 and Relaxed_EM are diagnostics.")
    print()

    all_summary = []
    all_type_rows = []

    for method in methods:
        rows = run_method(method, subset, llm, out_dir, args)
        all_summary.append(aggregate_rows(method, rows))
        all_type_rows.extend(aggregate_by_type(method, rows))

    write_csv(out_dir / "summary.csv", all_summary)
    write_csv(out_dir / "by_question_type.csv", all_type_rows)

    config_path = out_dir / "run_config.json"
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset": str(data_path),
                "dataset_size": len(data),
                "run_examples": len(subset),
                "offset": args.offset,
                "methods": methods,
                "base_url": args.base_url,
                "model": args.model,
                "temperature": args.temperature,
                "memory_tokens": args.memory_tokens,
                "running_memory_tokens": args.running_memory_tokens,
                "summary_max_tokens": args.summary_max_tokens,
                "qa_max_tokens": args.qa_max_tokens,
                "last_k_turns": args.last_k_turns,
                "retrieval_top_k": args.retrieval_top_k,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print_summary(all_summary)
    print(f"\nSaved:")
    print(f"  {out_dir / 'summary.csv'}")
    print(f"  {out_dir / 'by_question_type.csv'}")
    print(f"  {out_dir / 'run_config.json'}")
    print(f"  {out_dir / '<method>.jsonl'}")


if __name__ == "__main__":
    main()
