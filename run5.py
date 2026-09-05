#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LongMemEval: FIXED-size grouping vs semantic SEGMENT detection.

Goal
----
Compare ONLY memory-construction granularity:

  fixed_1:
      every 1 user-assistant exchange -> one memory extraction unit

  fixed_5:
      every 5 user-assistant exchanges -> one memory extraction unit

  You can pass arbitrary fixed sizes with --fixed-sizes.

  segment:
      a cheap LLM detects topic/event boundaries first
      -> each semantic segment -> one memory extraction unit

Everything else is held constant:
- same 50 LongMemEval oracle examples
- same untyped memory extraction prompt
- same final QA prompt/model
- same memory output budget per memory unit
- reasoning disabled
- structured JSON for segmentation + memory extraction
- multithreaded
- cache enabled
- tqdm shows EM + correct/wrong/error counts for BOTH methods

Cost-saving defaults
--------------------
segment model: qwen/qwen3.5-9b
memory model : qwen/qwen3.5-9b
QA model     : qwen/qwen3.5-35b-a3b
memory tokens: 256 / unit
segment tokens: 256 / session
QA tokens    : 16

Install
-------
pip install -U openai huggingface_hub tqdm

Run
---
export OPENROUTER_API_KEY="sk-or-v1-..."

python3 fixed_vs_segment.py \
  --limit 50 \
  --workers 32 \
  --fixed-sizes 1 5

If your endpoint/model dislikes json_schema, use:
  --json-object-fallback

Debug:
  --debug-api

Outputs
-------
results_fixed_vs_segment/
  details.jsonl
  summary.csv
  paired.csv
  by_question_type.csv
  run_config.json
"""

import argparse
import csv
import json
import os
import random
import re
import string
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from huggingface_hub import hf_hub_download
from openai import OpenAI
from tqdm import tqdm


# =============================================================================
# CONFIG
# =============================================================================

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

DATA_REPO = "xiaowu0162/longmemeval-cleaned"
DATA_FILE = "longmemeval_oracle.json"

BASE_METHODS = ("segment",)

def fixed_method_name(size: int) -> str:
    return f"fixed_{size}"


_thread_local = threading.local()
_print_lock = threading.Lock()


# =============================================================================
# STRUCTURED OUTPUT SCHEMAS
# =============================================================================

MEMORY_SCHEMA = {
    "name": "memory_extraction",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "memories": {
                "type": "array",
                "items": {"type": "string"},
            }
        },
        "required": ["memories"],
        "additionalProperties": False,
    },
}

SEGMENT_SCHEMA = {
    "name": "conversation_segmentation",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "segments": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "start": {"type": "integer"},
                        "end": {"type": "integer"},
                    },
                    "required": ["start", "end"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["segments"],
        "additionalProperties": False,
    },
}


# =============================================================================
# CLIENT / LLM
# =============================================================================

@dataclass
class LLMResult:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    finish_reason: Optional[str] = None


def get_client() -> OpenAI:
    if not hasattr(_thread_local, "client"):
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY is not set.\n"
                'export OPENROUTER_API_KEY="sk-or-v1-..."\n'
            )
        _thread_local.client = OpenAI(
            base_url=OPENROUTER_BASE_URL,
            api_key=api_key,
            timeout=120.0,
            max_retries=0,
        )
    return _thread_local.client


def safe_dump(obj: Any, limit: int = 3000) -> str:
    try:
        if hasattr(obj, "model_dump"):
            obj = obj.model_dump()
        s = json.dumps(obj, ensure_ascii=False, default=str)
    except Exception:
        s = repr(obj)
    return s[:limit]


def message_text(message: Any) -> str:
    content = getattr(message, "content", None)

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        out = []
        for part in content:
            if isinstance(part, str):
                out.append(part)
            elif isinstance(part, dict):
                if part.get("text"):
                    out.append(str(part["text"]))
            else:
                text = getattr(part, "text", None)
                if text:
                    out.append(str(text))
        return "\n".join(out).strip()

    return ""


def usage_numbers(response: Any) -> Tuple[int, int, int]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return 0, 0, 0

    def g(name: str) -> int:
        x = getattr(usage, name, 0)
        try:
            return int(x or 0)
        except Exception:
            return 0

    return (
        g("prompt_tokens"),
        g("completion_tokens"),
        g("total_tokens"),
    )


def extract_json_object(text: str) -> Dict[str, Any]:
    text = "" if text is None else str(text).strip()
    if not text:
        raise ValueError("empty content")

    text = re.sub(r"^\s*```(?:json)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*```\s*$", "", text).strip()

    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    for m in re.finditer(r"\{", text):
        try:
            obj, _ = decoder.raw_decode(text[m.start():])
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue

    raise ValueError(f"invalid JSON: {text[:1000]!r}")


def call_llm(
    *,
    model: str,
    system: str,
    user: str,
    max_tokens: int,
    json_schema: Optional[Dict[str, Any]] = None,
    retries: int = 5,
    debug_api: bool = False,
    json_object_fallback: bool = False,
) -> LLMResult:
    """
    OpenRouter/OpenAI-compatible call.

    Important:
    - reasoning explicitly disabled
    - empty content is retryable
    - malformed JSON is retryable
    - structured JSON is used when requested
    """
    client = get_client()
    last_error = None

    for attempt in range(1, retries + 1):
        response = None
        try:
            kwargs: Dict[str, Any] = {
                "model": model,
                "temperature": 0.0,
                "max_tokens": max_tokens,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "extra_body": {
                    "reasoning": {
                        "effort": "none",
                        "exclude": True,
                    }
                },
            }

            if json_schema is not None:
                if json_object_fallback:
                    kwargs["response_format"] = {"type": "json_object"}
                else:
                    kwargs["response_format"] = {
                        "type": "json_schema",
                        "json_schema": json_schema,
                    }

            response = client.chat.completions.create(**kwargs)

            if not getattr(response, "choices", None):
                raise ValueError("no choices")

            choice = response.choices[0]
            msg = getattr(choice, "message", None)
            if msg is None:
                raise ValueError("no message")

            text = message_text(msg)
            if not text:
                reasoning = (
                    getattr(msg, "reasoning", None)
                    or getattr(msg, "reasoning_content", None)
                )
                raise ValueError(
                    "empty content; "
                    f"finish_reason={getattr(choice, 'finish_reason', None)!r}, "
                    f"reasoning_len={len(str(reasoning)) if reasoning else 0}"
                )

            if json_schema is not None:
                extract_json_object(text)

            p, c, t = usage_numbers(response)

            if debug_api:
                with _print_lock:
                    print(
                        "\n[api-ok]",
                        model,
                        "finish=",
                        getattr(choice, "finish_reason", None),
                        "usage=",
                        (p, c, t),
                        "preview=",
                        repr(text[:200]),
                    )

            return LLMResult(
                text=text,
                prompt_tokens=p,
                completion_tokens=c,
                total_tokens=t,
                finish_reason=getattr(choice, "finish_reason", None),
            )

        except Exception as e:
            last_error = e
            with _print_lock:
                print(
                    f"\n[retry {attempt}/{retries}] "
                    f"model={model} {type(e).__name__}: {e}"
                )

            if attempt < retries:
                time.sleep(
                    min(
                        0.5 * (2 ** (attempt - 1))
                        + random.random() * 0.25,
                        8.0,
                    )
                )

    raise RuntimeError(
        f"LLM failed after {retries} attempts: {last_error}"
    )


# =============================================================================
# DATASET
# =============================================================================

def ensure_dataset(data_dir: str) -> Path:
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / DATA_FILE

    if path.exists():
        print(f"[dataset] exists: {path}")
        return path

    print("[dataset] downloading LongMemEval oracle...")
    downloaded = hf_hub_download(
        repo_id=DATA_REPO,
        filename=DATA_FILE,
        repo_type="dataset",
        local_dir=str(data_dir),
    )
    return Path(downloaded)


def load_dataset(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"[dataset] examples={len(data)}")
    return data


def balanced_sample(
    data: List[Dict[str, Any]],
    limit: int,
    seed: int,
    include_abstention: bool = False,
) -> List[Dict[str, Any]]:
    rnd = random.Random(seed)
    groups = defaultdict(list)

    for x in data:
        qid = str(x["question_id"])
        if not include_abstention and qid.endswith("_abs"):
            continue
        groups[x["question_type"]].append(x)

    for xs in groups.values():
        rnd.shuffle(xs)

    print("\n[available question types]")
    for k in sorted(groups):
        print(f"  {k:30s}: {len(groups[k])}")

    selected = []
    keys = sorted(groups)
    i = 0

    while len(selected) < limit:
        added = False
        for k in keys:
            if i < len(groups[k]):
                selected.append(groups[k][i])
                added = True
                if len(selected) >= limit:
                    break
        if not added:
            break
        i += 1

    rnd.shuffle(selected)

    print("\n[selected distribution]")
    cnt = Counter(x["question_type"] for x in selected)
    for k in sorted(cnt):
        print(f"  {k:30s}: {cnt[k]}")
    print(f"  {'TOTAL':30s}: {len(selected)}")

    return selected


# =============================================================================
# TURN -> EXCHANGE
# =============================================================================

def turns_to_exchanges(
    session: List[Dict[str, Any]],
) -> List[List[Dict[str, Any]]]:
    """
    SeCom-style unit: a user turn plus following assistant turn(s)
    until the next user turn.

    Any leading non-user turns are kept as their own first exchange so
    nothing is dropped.
    """
    exchanges: List[List[Dict[str, Any]]] = []
    current: List[Dict[str, Any]] = []

    for turn in session:
        role = str(turn.get("role", "")).lower()

        if role == "user" and current:
            exchanges.append(current)
            current = []

        current.append(turn)

    if current:
        exchanges.append(current)

    return exchanges


def format_exchange(
    exchange: List[Dict[str, Any]],
    exchange_idx: int,
) -> str:
    lines = [f"[Exchange {exchange_idx}]"]
    for turn in exchange:
        role = str(turn.get("role", "unknown")).lower()
        content = str(turn.get("content", "")).strip()
        lines.append(f"[{role}]: {content}")
    return "\n".join(lines)


def format_exchange_range(
    exchanges: List[List[Dict[str, Any]]],
    start: int,
    end: int,
    date: Any,
    session_idx: int,
) -> str:
    lines = [
        f"SESSION: {session_idx}",
        f"DATE: {date}",
    ]
    for i in range(start, end + 1):
        lines.append(format_exchange(exchanges[i], i))
    return "\n\n".join(lines)


# =============================================================================
# FIXED GROUPING
# =============================================================================

def fixed_ranges(n_exchanges: int, fixed_size: int) -> List[Tuple[int, int]]:
    out = []
    for start in range(0, n_exchanges, fixed_size):
        end = min(n_exchanges - 1, start + fixed_size - 1)
        out.append((start, end))
    return out


# =============================================================================
# SEMANTIC SEGMENTATION
# =============================================================================

SEGMENT_SYSTEM = """
You segment a dialogue into contiguous semantically coherent conversation units.

A segment should keep consecutive exchanges together while they are about the
same main topic, event, task, person, decision, or tightly connected context.

Start a new segment only when the main topic/event/goal materially changes.

Hard requirements:
- Every exchange must belong to exactly one segment.
- Segments must be contiguous.
- No gaps.
- No overlaps.
- Preserve chronological order.
- Do NOT create one segment per exchange unless the topics truly change that often.
- Do NOT merge clearly unrelated topics just to make segments larger.
- Return JSON only, matching the required schema.
""".strip()


def segment_prompt(
    exchanges: List[List[Dict[str, Any]]],
    date: Any,
) -> str:
    rendered = "\n\n".join(
        format_exchange(ex, i)
        for i, ex in enumerate(exchanges)
    )

    return f"""
Segment the following conversation by semantic/topic/event boundaries.

There are exactly {len(exchanges)} exchanges, indexed 0 through {len(exchanges)-1}.

Return:
{{
  "segments": [
    {{"start": 0, "end": 2}},
    {{"start": 3, "end": 6}}
  ]
}}

Validation rules:
1. The first segment MUST start at 0.
2. The final segment MUST end at {len(exchanges)-1}.
3. If one segment ends at k, the next MUST start at k+1.
4. start <= end for every segment.
5. Every exchange appears exactly once.
6. Prefer a small number of coherent segments over over-segmentation.

SESSION DATE
============
{date}

CONVERSATION
============
{rendered}
""".strip()


def validate_segments(
    obj: Dict[str, Any],
    n: int,
) -> List[Tuple[int, int]]:
    raw = obj.get("segments")
    if not isinstance(raw, list) or not raw:
        raise ValueError("segments must be a non-empty list")

    ranges = []
    expected_start = 0

    for i, seg in enumerate(raw):
        if not isinstance(seg, dict):
            raise ValueError(f"segment {i} is not object")

        start = int(seg["start"])
        end = int(seg["end"])

        if start != expected_start:
            raise ValueError(
                f"gap/overlap at segment {i}: "
                f"expected start={expected_start}, got {start}"
            )

        if start < 0 or end < start or end >= n:
            raise ValueError(
                f"bad range segment {i}: ({start},{end}), n={n}"
            )

        ranges.append((start, end))
        expected_start = end + 1

    if expected_start != n:
        raise ValueError(
            f"segments do not cover all exchanges: covered through "
            f"{expected_start-1}, expected {n-1}"
        )

    return ranges


def detect_segments(
    *,
    exchanges: List[List[Dict[str, Any]]],
    date: Any,
    model: str,
    max_tokens: int,
    retries: int,
    debug_api: bool,
    json_object_fallback: bool,
) -> Tuple[List[Tuple[int, int]], Dict[str, int]]:
    if len(exchanges) == 1:
        return [(0, 0)], {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "calls": 0,
        }

    last_error = None
    usage = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "calls": 0,
    }

    for attempt in range(1, retries + 1):
        try:
            res = call_llm(
                model=model,
                system=SEGMENT_SYSTEM,
                user=segment_prompt(exchanges, date),
                max_tokens=max_tokens,
                json_schema=SEGMENT_SCHEMA,
                retries=3,
                debug_api=debug_api,
                json_object_fallback=json_object_fallback,
            )

            usage["prompt_tokens"] += res.prompt_tokens
            usage["completion_tokens"] += res.completion_tokens
            usage["total_tokens"] += res.total_tokens
            usage["calls"] += 1

            obj = extract_json_object(res.text)
            ranges = validate_segments(obj, len(exchanges))
            return ranges, usage

        except Exception as e:
            last_error = e
            if attempt < retries:
                time.sleep(0.25 * attempt)

    raise RuntimeError(
        f"semantic segmentation failed: {last_error}"
    )


# =============================================================================
# MEMORY EXTRACTION
# =============================================================================

MEMORY_SYSTEM = """
You are a deterministic long-term memory extraction engine.

Extract useful information that may help answer future questions about this
conversation.

Hard requirements:
- Use only information explicitly supported by the dialogue.
- Never invent.
- Preserve names, dates, numbers, quantities, preferences, relationships,
  actions, decisions, events, and changes over time.
- Preserve temporal information when available.
- If a state changes, preserve enough detail to distinguish old and new states.
- Prefer concise atomic memories.
- Ignore conversational filler.
- Return one valid JSON object only.
- No markdown, no explanation, no code fences.
""".strip()


def memory_prompt(unit_text: str) -> str:
    return f"""
Extract long-term memories from this dialogue unit.

Return exactly:
{{
  "memories": [
    "atomic memory"
  ]
}}

If there is nothing useful:
{{"memories":[]}}

Rules:
- Each memory must be independently understandable.
- Do not answer any future question.
- Do not infer unsupported details.
- Keep important event relationships inside one memory when splitting them
  would make the information ambiguous.
- Return JSON only.

DIALOGUE UNIT
=============
{unit_text}
""".strip()


def normalize_memories(obj: Dict[str, Any]) -> List[str]:
    out = []
    for x in obj.get("memories", []) or []:
        if isinstance(x, str):
            s = x.strip()
        elif isinstance(x, dict):
            s = str(x.get("memory", "")).strip()
        else:
            continue
        if s:
            out.append(s)
    return out


def extract_memory_unit(
    *,
    unit_text: str,
    model: str,
    max_tokens: int,
    debug_api: bool,
    json_object_fallback: bool,
) -> Tuple[List[str], LLMResult]:
    res = call_llm(
        model=model,
        system=MEMORY_SYSTEM,
        user=memory_prompt(unit_text),
        max_tokens=max_tokens,
        json_schema=MEMORY_SCHEMA,
        retries=5,
        debug_api=debug_api,
        json_object_fallback=json_object_fallback,
    )
    return normalize_memories(extract_json_object(res.text)), res


# =============================================================================
# QA / EM
# =============================================================================

QA_SYSTEM = """
Answer the question using only the extracted memories.

Rules:
- Consider all memories.
- Pay attention to dates and temporal changes.
- If information changed over time, use the state relevant to the question.
- Do not explain reasoning.
- Return only the shortest direct answer.
""".strip()


def answer_question(
    *,
    item: Dict[str, Any],
    memories: List[str],
    model: str,
    max_tokens: int,
    debug_api: bool,
) -> LLMResult:
    mem_text = "\n".join(
        f"- {m}"
        for m in memories
    )
    if not mem_text:
        mem_text = "(no extracted memories)"

    prompt = f"""
MEMORIES
========
{mem_text}

QUESTION DATE
=============
{item.get("question_date", "unknown")}

QUESTION
========
{item["question"]}

Return only the shortest direct answer.
""".strip()

    return call_llm(
        model=model,
        system=QA_SYSTEM,
        user=prompt,
        max_tokens=max_tokens,
        json_schema=None,
        retries=5,
        debug_api=debug_api,
    )


def normalize_answer(s: Any) -> str:
    if s is None:
        return ""

    s = str(s).lower()
    s = "".join(
        ch
        for ch in s
        if ch not in set(string.punctuation)
    )
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def exact_match(prediction: str, gold: Any) -> int:
    if isinstance(gold, list):
        return int(
            any(
                normalize_answer(prediction)
                == normalize_answer(g)
                for g in gold
            )
        )
    return int(
        normalize_answer(prediction)
        == normalize_answer(gold)
    )


# =============================================================================
# CACHE
# =============================================================================

class JsonlCache:
    def __init__(self, path: str):
        self.path = Path(path)
        self.lock = threading.Lock()
        self.data: Dict[str, Dict[str, Any]] = {}

        if self.path.exists():
            with open(self.path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        r = json.loads(line)
                    except Exception:
                        continue
                    key = r.get("cache_key")
                    if key:
                        self.data[key] = r

        print(f"[cache] loaded {len(self.data)} records")

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        return self.data.get(key)

    def put(self, key: str, value: Dict[str, Any]) -> None:
        record = {"cache_key": key, **value}
        with self.lock:
            self.data[key] = record
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")


# =============================================================================
# BUILD MEMORY UNITS
# =============================================================================

def build_units_fixed(
    item: Dict[str, Any],
    fixed_size: int,
) -> Tuple[List[str], Dict[str, Any]]:
    units = []
    ranges_meta = []

    for session_idx, (session, date) in enumerate(
        zip(item["haystack_sessions"], item["haystack_dates"]),
        start=1,
    ):
        exchanges = turns_to_exchanges(session)
        if not exchanges:
            continue

        ranges = fixed_ranges(
            len(exchanges),
            fixed_size,
        )

        for start, end in ranges:
            units.append(
                format_exchange_range(
                    exchanges,
                    start,
                    end,
                    date,
                    session_idx,
                )
            )
            ranges_meta.append(
                {
                    "session": session_idx,
                    "date": date,
                    "start": start,
                    "end": end,
                    "num_exchanges": end - start + 1,
                }
            )

    return units, {
        "ranges": ranges_meta,
        "segment_calls": 0,
        "segment_prompt_tokens": 0,
        "segment_completion_tokens": 0,
        "segment_total_tokens": 0,
    }


def build_units_segment(
    *,
    item: Dict[str, Any],
    segment_model: str,
    segment_tokens: int,
    segment_retries: int,
    debug_api: bool,
    json_object_fallback: bool,
) -> Tuple[List[str], Dict[str, Any]]:
    units = []
    ranges_meta = []

    stats = {
        "segment_calls": 0,
        "segment_prompt_tokens": 0,
        "segment_completion_tokens": 0,
        "segment_total_tokens": 0,
    }

    for session_idx, (session, date) in enumerate(
        zip(item["haystack_sessions"], item["haystack_dates"]),
        start=1,
    ):
        exchanges = turns_to_exchanges(session)
        if not exchanges:
            continue

        ranges, usage = detect_segments(
            exchanges=exchanges,
            date=date,
            model=segment_model,
            max_tokens=segment_tokens,
            retries=segment_retries,
            debug_api=debug_api,
            json_object_fallback=json_object_fallback,
        )

        stats["segment_calls"] += usage["calls"]
        stats["segment_prompt_tokens"] += usage["prompt_tokens"]
        stats["segment_completion_tokens"] += usage["completion_tokens"]
        stats["segment_total_tokens"] += usage["total_tokens"]

        for start, end in ranges:
            units.append(
                format_exchange_range(
                    exchanges,
                    start,
                    end,
                    date,
                    session_idx,
                )
            )
            ranges_meta.append(
                {
                    "session": session_idx,
                    "date": date,
                    "start": start,
                    "end": end,
                    "num_exchanges": end - start + 1,
                }
            )

    return units, {
        "ranges": ranges_meta,
        **stats,
    }


# =============================================================================
# ONE CONDITION
# =============================================================================

def run_condition(
    *,
    item: Dict[str, Any],
    method: str,
    args: argparse.Namespace,
    cache: JsonlCache,
) -> Dict[str, Any]:
    qid = str(item["question_id"])

    cache_key = (
        f"fixed-vs-segment-v1"
        f"::{qid}"
        f"::{method}"
        f"::fixedsizes={','.join(map(str, args.fixed_sizes))}"
        f"::seg={args.segment_model}"
        f"::mem={args.memory_model}"
        f"::qa={args.qa_model}"
        f"::segTok{args.segment_tokens}"
        f"::memTok{args.memory_tokens}"
        f"::qaTok{args.qa_tokens}"
    )

    cached = cache.get(cache_key)
    if cached and cached.get("status") == "ok":
        return cached

    t0 = time.time()

    if method.startswith("fixed_"):
        fixed_size = int(method.split("_", 1)[1])
        units, grouping_meta = build_units_fixed(
            item,
            fixed_size,
        )
    elif method == "segment":
        units, grouping_meta = build_units_segment(
            item=item,
            segment_model=args.segment_model,
            segment_tokens=args.segment_tokens,
            segment_retries=args.segment_retries,
            debug_api=args.debug_api,
            json_object_fallback=args.json_object_fallback,
        )
    else:
        raise ValueError(method)

    memories: List[str] = []

    mem_prompt_tokens = 0
    mem_completion_tokens = 0
    mem_total_tokens = 0
    mem_calls = 0

    for unit in units:
        mems, res = extract_memory_unit(
            unit_text=unit,
            model=args.memory_model,
            max_tokens=args.memory_tokens,
            debug_api=args.debug_api,
            json_object_fallback=args.json_object_fallback,
        )

        memories.extend(mems)
        mem_prompt_tokens += res.prompt_tokens
        mem_completion_tokens += res.completion_tokens
        mem_total_tokens += res.total_tokens
        mem_calls += 1

    qa = answer_question(
        item=item,
        memories=memories,
        model=args.qa_model,
        max_tokens=args.qa_tokens,
        debug_api=args.debug_api,
    )

    pred = qa.text.strip()
    gold = item["answer"]
    em = exact_match(pred, gold)

    result = {
        "status": "ok",
        "question_id": qid,
        "question_type": item.get("question_type"),
        "question": item.get("question"),
        "question_date": item.get("question_date"),
        "gold": gold,
        "prediction": pred,
        "em": em,
        "method": method,

        "num_units": len(units),
        "num_memories": len(memories),
        "unit_ranges": grouping_meta["ranges"],

        "segment_calls": grouping_meta["segment_calls"],
        "segment_prompt_tokens": grouping_meta["segment_prompt_tokens"],
        "segment_completion_tokens": grouping_meta["segment_completion_tokens"],
        "segment_total_tokens": grouping_meta["segment_total_tokens"],

        "memory_calls": mem_calls,
        "memory_prompt_tokens": mem_prompt_tokens,
        "memory_completion_tokens": mem_completion_tokens,
        "memory_total_tokens": mem_total_tokens,

        "qa_calls": 1,
        "qa_prompt_tokens": qa.prompt_tokens,
        "qa_completion_tokens": qa.completion_tokens,
        "qa_total_tokens": qa.total_tokens,

        "all_prompt_tokens": (
            grouping_meta["segment_prompt_tokens"]
            + mem_prompt_tokens
            + qa.prompt_tokens
        ),
        "all_completion_tokens": (
            grouping_meta["segment_completion_tokens"]
            + mem_completion_tokens
            + qa.completion_tokens
        ),
        "all_total_tokens": (
            grouping_meta["segment_total_tokens"]
            + mem_total_tokens
            + qa.total_tokens
        ),

        "elapsed_sec": time.time() - t0,
        "memories": memories,
    }

    cache.put(cache_key, result)
    return result


# =============================================================================
# PROGRESS
# =============================================================================

def progress_stats(
    results: List[Dict[str, Any]],
    methods: List[str],
) -> Dict[str, Any]:
    d = {}

    for method in methods:
        rows = [
            r
            for r in results
            if r.get("method") == method
        ]
        oks = [
            r
            for r in rows
            if r.get("status") == "ok"
        ]
        correct = sum(
            int(r.get("em", 0))
            for r in oks
        )
        wrong = len(oks) - correct
        err = len(rows) - len(oks)
        em = (
            correct / len(oks)
            if oks
            else 0.0
        )

        if method == "segment":
            prefix = "seg"
        elif method.startswith("fixed_"):
            prefix = "f" + method.split("_", 1)[1]
        else:
            prefix = method

        d[f"{prefix}_EM"] = f"{em:.3f}"
        d[f"{prefix}_✓"] = correct
        d[f"{prefix}_✗"] = wrong
        d[f"{prefix}_err"] = err

    return d


# =============================================================================
# OUTPUT
# =============================================================================

def mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def write_csv(path: Path, fieldnames: List[str], rows: List[Dict[str, Any]]):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k) for k in fieldnames})


def summarize(
    results: List[Dict[str, Any]],
    outdir: Path,
    methods: List[str],
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    ok = [r for r in results if r.get("status") == "ok"]

    print("\n" + "=" * 116)
    print("OVERALL")
    print("=" * 116)

    header = (
        f"{'method':10s} {'n':>5s} {'EM':>8s} "
        f"{'correct':>8s} {'wrong':>8s} {'errors':>8s} "
        f"{'units':>8s} {'memories':>10s} {'calls':>8s} "
        f"{'tokens':>12s} {'sec':>9s}"
    )
    print(header)
    print("-" * len(header))

    summary_rows = []

    for method in methods:
        attempted = [r for r in results if r.get("method") == method]
        rows = [r for r in attempted if r.get("status") == "ok"]

        correct = sum(int(r["em"]) for r in rows)
        wrong = len(rows) - correct
        errors = len(attempted) - len(rows)
        em = correct / len(rows) if rows else float("nan")

        avg_units = mean([float(r["num_units"]) for r in rows])
        avg_memories = mean([float(r["num_memories"]) for r in rows])
        avg_calls = mean([
            float(
                r["segment_calls"]
                + r["memory_calls"]
                + r["qa_calls"]
            )
            for r in rows
        ])
        avg_tokens = mean([float(r["all_total_tokens"]) for r in rows])
        avg_sec = mean([float(r["elapsed_sec"]) for r in rows])

        print(
            f"{method:10s} {len(rows):5d} {em:8.4f} "
            f"{correct:8d} {wrong:8d} {errors:8d} "
            f"{avg_units:8.2f} {avg_memories:10.2f} "
            f"{avg_calls:8.2f} {avg_tokens:12.1f} {avg_sec:9.2f}"
        )

        summary_rows.append({
            "method": method,
            "attempted": len(attempted),
            "n_ok": len(rows),
            "correct": correct,
            "wrong": wrong,
            "errors": errors,
            "em": em,
            "avg_units": avg_units,
            "avg_memories": avg_memories,
            "avg_calls": avg_calls,
            "avg_total_tokens": avg_tokens,
            "avg_elapsed_sec": avg_sec,
        })

    write_csv(
        outdir / "summary.csv",
        list(summary_rows[0].keys()),
        summary_rows,
    )

    # -------------------------------------------------------------------------
    # Paired comparison across every method
    # -------------------------------------------------------------------------
    by_qid: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)

    for r in ok:
        by_qid[r["question_id"]][r["method"]] = r

    paired_rows = []
    complete_qids = [
        qid
        for qid, mm in by_qid.items()
        if all(m in mm for m in methods)
    ]

    print("\n[PAIRED]")

    # pairwise win/loss/tie counts
    for i, m1 in enumerate(methods):
        for m2 in methods[i + 1:]:
            m1_only = 0
            m2_only = 0
            both_correct = 0
            both_wrong = 0

            for qid in complete_qids:
                a = int(by_qid[qid][m1]["em"])
                b = int(by_qid[qid][m2]["em"])

                if a and b:
                    both_correct += 1
                elif a and not b:
                    m1_only += 1
                elif b and not a:
                    m2_only += 1
                else:
                    both_wrong += 1

            print(
                f"{m1} vs {m2}: "
                f"{m1}_only={m1_only} | "
                f"{m2}_only={m2_only} | "
                f"both_correct={both_correct} | "
                f"both_wrong={both_wrong}"
            )

    for qid in complete_qids:
        first = by_qid[qid][methods[0]]

        row = {
            "question_id": qid,
            "question_type": first.get("question_type"),
            "gold": json.dumps(
                first.get("gold"),
                ensure_ascii=False,
            ),
        }

        for method in methods:
            r = by_qid[qid][method]
            row[f"{method}_em"] = int(r["em"])
            row[f"{method}_prediction"] = r.get("prediction")
            row[f"{method}_units"] = r.get("num_units")
            row[f"{method}_memories"] = r.get("num_memories")
            row[f"{method}_tokens"] = r.get("all_total_tokens")

        paired_rows.append(row)

    if paired_rows:
        fieldnames = []
        for row in paired_rows:
            for k in row.keys():
                if k not in fieldnames:
                    fieldnames.append(k)

        write_csv(
            outdir / "paired.csv",
            fieldnames,
            paired_rows,
        )

    # -------------------------------------------------------------------------
    # By question type
    # -------------------------------------------------------------------------
    by_type_rows = []

    qtypes = sorted(
        set(r.get("question_type") for r in ok)
    )

    print("\n[BY QUESTION TYPE]")
    print(
        f"{'type':30s} {'method':10s} {'n':>4s} {'EM':>8s}"
    )
    print("-" * 58)

    for qtype in qtypes:
        for method in methods:
            rows = [
                r
                for r in ok
                if r.get("question_type") == qtype
                and r.get("method") == method
            ]
            if not rows:
                continue

            em = mean([float(r["em"]) for r in rows])

            print(
                f"{str(qtype):30s} {method:10s} "
                f"{len(rows):4d} {em:8.4f}"
            )

            by_type_rows.append({
                "question_type": qtype,
                "method": method,
                "n": len(rows),
                "em": em,
            })

    if by_type_rows:
        write_csv(
            outdir / "by_question_type.csv",
            list(by_type_rows[0].keys()),
            by_type_rows,
        )

    # -------------------------------------------------------------------------
    # Details jsonl
    # -------------------------------------------------------------------------
    with open(
        outdir / "details.jsonl",
        "w",
        encoding="utf-8",
    ) as f:
        for r in results:
            f.write(
                json.dumps(r, ensure_ascii=False)
                + "\n"
            )


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=32)

    parser.add_argument(
        "--fixed-sizes",
        nargs="+",
        type=int,
        default=[1, 5],
        help=(
            "Fixed grouping sizes to compare, in user-assistant exchanges. "
            "Default: 1 5"
        ),
    )

    parser.add_argument(
        "--segment-model",
        default="qwen/qwen3.5-9b",
    )
    parser.add_argument(
        "--memory-model",
        default="qwen/qwen3.5-9b",
    )
    parser.add_argument(
        "--qa-model",
        default="qwen/qwen3.5-35b-a3b",
    )

    parser.add_argument(
        "--segment-tokens",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--memory-tokens",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--qa-tokens",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--segment-retries",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--data-dir",
        default="./longmemeval_data",
    )

    parser.add_argument(
        "--cache",
        default="./fixed_vs_segment_cache.jsonl",
    )

    parser.add_argument(
        "--outdir",
        default="./results_fixed_vs_segment",
    )

    parser.add_argument(
        "--include-abstention",
        action="store_true",
    )

    parser.add_argument(
        "--debug-api",
        action="store_true",
    )

    parser.add_argument(
        "--json-object-fallback",
        action="store_true",
        help="Use response_format=json_object instead of strict json_schema.",
    )

    parser.add_argument(
        "--print-every-result",
        action="store_true",
        help="Print every correct/wrong result as jobs complete.",
    )

    args = parser.parse_args()

    if not args.fixed_sizes:
        raise ValueError("--fixed-sizes must contain at least one value")

    if any(x < 1 for x in args.fixed_sizes):
        raise ValueError("all --fixed-sizes values must be >= 1")

    # remove duplicates while preserving user order
    args.fixed_sizes = list(dict.fromkeys(args.fixed_sizes))

    methods = [
        fixed_method_name(x)
        for x in args.fixed_sizes
    ] + ["segment"]

    print("=" * 110)
    print("LONGMEMEVAL: FIXED GROUPING vs SEMANTIC SEGMENTATION")
    print("=" * 110)
    print(f"limit          : {args.limit}")
    print(f"workers        : {args.workers}")
    print(f"fixed sizes    : {args.fixed_sizes} exchanges")
    print(f"segment model  : {args.segment_model}")
    print(f"memory model   : {args.memory_model}")
    print(f"qa model       : {args.qa_model}")
    print(f"segment tokens : {args.segment_tokens}")
    print(f"memory tokens  : {args.memory_tokens} / memory unit")
    print(f"qa tokens      : {args.qa_tokens}")
    print("reasoning      : OFF")
    print("memory type    : UNTYPED for both methods")
    print("structured JSON: ON")

    dataset_path = ensure_dataset(args.data_dir)
    data = load_dataset(dataset_path)

    subset = balanced_sample(
        data,
        limit=args.limit,
        seed=args.seed,
        include_abstention=args.include_abstention,
    )

    cache = JsonlCache(args.cache)

    jobs = []
    for item in subset:
        for method in methods:
            jobs.append((item, method))

    results: List[Dict[str, Any]] = []

    def worker(job):
        item, method = job
        qid = str(item["question_id"])

        try:
            return run_condition(
                item=item,
                method=method,
                args=args,
                cache=cache,
            )
        except Exception as e:
            with _print_lock:
                print(
                    f"\n[ERROR] qid={qid} method={method} "
                    f"{type(e).__name__}: {e}"
                )
            return {
                "status": "error",
                "question_id": qid,
                "question_type": item.get("question_type"),
                "question": item.get("question"),
                "gold": item.get("answer"),
                "method": method,
                "error_type": type(e).__name__,
                "error": str(e),
            }

    with ThreadPoolExecutor(
        max_workers=args.workers
    ) as ex:
        futs = {
            ex.submit(worker, job): job
            for job in jobs
        }

        with tqdm(
            total=len(jobs),
            desc="Running",
            dynamic_ncols=True,
        ) as pbar:

            for fut in as_completed(futs):
                r = fut.result()
                results.append(r)

                if (
                    args.print_every_result
                    and r.get("status") == "ok"
                ):
                    mark = "✓" if r["em"] else "✗"
                    with _print_lock:
                        tqdm.write(
                            f"{mark} "
                            f"{r['method']:7s} "
                            f"qid={r['question_id']} "
                            f"pred={r['prediction']!r} "
                            f"gold={r['gold']!r}"
                        )

                pbar.set_postfix(
                    progress_stats(results, methods),
                    refresh=False,
                )
                pbar.update(1)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    summarize(
        results,
        outdir,
        methods,
    )

    with open(
        outdir / "run_config.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            vars(args),
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"\nSaved under: {outdir}")
    print("  summary.csv")
    print("  paired.csv")
    print("  by_question_type.csv")
    print("  details.jsonl")
    print("  run_config.json")


if __name__ == "__main__":
    main()
