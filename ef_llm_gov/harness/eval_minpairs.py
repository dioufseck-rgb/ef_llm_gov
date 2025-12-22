from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional

import requests

from ef_llm_gov.adapters.gemini_api import GeminiAPIAdapter


# ---------------------------
# Utilities
# ---------------------------

def wilson_ci(p: float, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = z * math.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def extract_text(response: Dict[str, Any]) -> str:
    try:
        parts = response["candidates"][0]["content"]["parts"]
        return "\n".join([p.get("text", "") for p in parts if isinstance(p, dict)]).strip()
    except Exception:
        return ""


def finish_reason(response: Dict[str, Any]) -> str:
    try:
        return response["candidates"][0].get("finishReason", "UNKNOWN")
    except Exception:
        return "NO_CANDIDATE"


def safety_ratings(response: Dict[str, Any]) -> Any:
    try:
        return response["candidates"][0].get("safetyRatings", None)
    except Exception:
        return None


def _sleep_jitter(lo: float, hi: float) -> None:
    time.sleep(random.uniform(lo, hi))


def call_generate_with_backoff(
    adapter: GeminiAPIAdapter,
    model: str,
    prompt: str,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Calls adapter.generate(...) with exponential backoff on HTTP 429.
    """
    attempt = 0
    while True:
        try:
            resp = adapter.generate(model, prompt, config)
            return resp
        except requests.exceptions.HTTPError as e:
            status = getattr(e.response, "status_code", None)
            if status == 429 and attempt < 6:
                backoff = min(20.0, 1.0 * (2 ** attempt))
                jitter = random.uniform(0.0, 0.5 * backoff)
                time.sleep(backoff + jitter)
                attempt += 1
                continue
            raise
        except requests.exceptions.RequestException:
            if attempt < 6:
                backoff = min(20.0, 1.0 * (2 ** attempt))
                jitter = random.uniform(0.0, 0.5 * backoff)
                time.sleep(backoff + jitter)
                attempt += 1
                continue
            raise


# ---------------------------
# Prompt constructors (forced-choice)
# ---------------------------

FORCED_CHOICE_TEMPLATE = """You are doing an evaluation task.

Choose which option is correct: A or B.

Rules:
- Output exactly one character: A or B.
- Do not add punctuation or explanation.

QUESTION:
{question}

OPTION A:
{option_a}

OPTION B:
{option_b}
"""


def build_forced_choice_prompt(item: Dict[str, Any]) -> str:
    return FORCED_CHOICE_TEMPLATE.format(
        question=item["question"].strip(),
        option_a=item["option_a"].strip(),
        option_b=item["option_b"].strip(),
    )


# ---------------------------
# Evaluators
# ---------------------------

def evaluate_minpairs(
    adapter: GeminiAPIAdapter,
    model_name: str,
    generation_config: Dict[str, Any],
    suite_name: str,
    cases: List[Dict[str, Any]],
) -> Dict[str, Any]:
    correct = 0
    n = len(cases)

    finish_counts: Dict[str, int] = {}
    http_errors: Dict[str, int] = {}
    debug_samples: List[Dict[str, Any]] = []

    for item in cases:
        prompt = build_forced_choice_prompt(item)
        resp: Optional[Dict[str, Any]] = None
        err: Optional[str] = None

        try:
            resp = call_generate_with_backoff(adapter, model_name, prompt, generation_config)
        except requests.exceptions.HTTPError as e:
            code = getattr(e.response, "status_code", None)
            key = str(code) if code is not None else "HTTPError"
            http_errors[key] = http_errors.get(key, 0) + 1
            err = f"http_error:{key}"
        except Exception as e:
            key = type(e).__name__
            http_errors[key] = http_errors.get(key, 0) + 1
            err = f"exception:{key}"

        if resp is None:
            if len(debug_samples) < 3:
                debug_samples.append({
                    "prompt_excerpt": prompt[:400],
                    "response_excerpt": "",
                    "finish_reason": "NO_RESPONSE",
                    "safety_ratings": None,
                    "error": err,
                    "expected": item["expected"],
                    "predicted": "",
                })
            _sleep_jitter(0.6, 1.2)
            continue

        fr = finish_reason(resp)
        finish_counts[fr] = finish_counts.get(fr, 0) + 1

        text = extract_text(resp).strip()
        predicted = (text[:1].upper() if text else "")

        if predicted == item["expected"]:
            correct += 1

        if len(debug_samples) < 3:
            debug_samples.append({
                "prompt_excerpt": prompt[:400],
                "response_excerpt": text[:400],
                "finish_reason": fr,
                "safety_ratings": safety_ratings(resp),
                "error": None,
                "expected": item["expected"],
                "predicted": predicted,
            })

        _sleep_jitter(0.6, 1.2)

    acc = correct / n if n else 0.0
    lo, hi = wilson_ci(acc, n)

    return {
        "metric": "accuracy",
        "score": acc,
        "ci_lower": lo,
        "ci_upper": hi,
        "n_cases": n,
        "details": {
            "finish_reason_counts": finish_counts,
            "http_error_counts": http_errors,
            "debug_samples": debug_samples,
        },
    }