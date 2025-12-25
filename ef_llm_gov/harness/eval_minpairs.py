# ef_llm_gov/harness/eval_minpairs.py

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

VALID_CHOICES = {"A", "B", "U"}

def build_minpairs_prompt(case: Dict[str, Any]) -> str:
    policy = (case.get("policy") or "").strip()
    question = (case.get("question") or "").strip()
    a = (case.get("option_a") or "").strip()
    b = (case.get("option_b") or "").strip()

    parts = [
        "Instruction: Choose A or B if the policy/text uniquely determines the correct option. "
        "If it does not uniquely determine the answer (multiple reasonable interpretations lead to different outcomes), output U. "
        "Output exactly one character: A, B, or U. Use the format \\boxed{RESULT} at the end of your response."
    ]
    if policy: parts.append(f"\nPolicy:\n{policy}")
    parts.append(f"\nQuestion:\n{question}")
    parts.append(f"\nOptions:\nA) {a}\nB) {b}\n\nAnswer (A/B/U):")
    return "\n".join(parts)

def normalize_choice(model_text: str) -> str:
    if not model_text: return "INVALID"
    t_upper = model_text.upper().strip()
    boxed_match = re.findall(r"\\BOXED\{([ABU])\}", t_upper)
    if boxed_match: return boxed_match[-1]
    answer_match = re.findall(r"ANSWER:\s*([ABU])", t_upper)
    if answer_match: return answer_match[-1]
    for ch in reversed(t_upper):
        if ch in VALID_CHOICES: return ch
    return "INVALID"

def wilson_ci(p: float, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n <= 0: return 0.0, 1.0
    denom = 1.0 + (z * z) / n
    center = (p + (z * z) / (2.0 * n)) / denom
    half = (z / denom) * ((p * (1.0 - p) / n + (z * z) / (4.0 * n * n)) ** 0.5)
    return max(0.0, center - half), min(1.0, center + half)

def evaluate_minpairs(
    adapter,
    model_name: str,
    generation_config: Dict[str, Any],
    suite_name: str,
    cases: List[Dict[str, Any]],
) -> Dict[str, Any]:
    n = len(cases)
    decidable_total = decidable_correct = u_total = u_tp = u_pred_total = format_ok_total = 0
    thought_counts = []
    http_error_counts = {}
    finish_reason_counts = {}
    debug_samples = []

    for i, case in enumerate(cases):
        prompt = build_minpairs_prompt(case)
        try:
            resp = adapter.generate(model=model_name, prompt=prompt, generation_config=generation_config)
            text = resp.get("text", "")
            fr = resp.get("finish_reason", "UNKNOWN")
            t_cost = resp.get("thought_tokens", 0)
            thought_counts.append(t_cost)
        except Exception as e:
            key = type(e).__name__
            http_error_counts[key] = http_error_counts.get(key, 0) + 1
            text, fr, t_cost = "", "ERROR", 0

        finish_reason_counts[fr] = finish_reason_counts.get(fr, 0) + 1
        pred = normalize_choice(text)
        exp = str(case.get("expected", "")).strip().upper()
        
        format_ok = pred in VALID_CHOICES
        format_ok_total += int(format_ok)

        if exp in {"A", "B"}:
            decidable_total += 1
            decidable_correct += int(pred == exp)
        elif exp == "U":
            u_total += 1
            u_tp += int(pred == "U")
        u_pred_total += int(pred == "U")

        # Storing per-question thought cost in debug samples
        debug_samples.append({
            "case_index": i,
            "expected": exp,
            "predicted": pred,
            "thought_tokens": t_cost,
            "raw_text": (text[:200] + "…") if len(text) > 200 else text,
        })

    avg_thoughts = sum(thought_counts) / max(1, len(thought_counts))
    accuracy_decidable = decidable_correct / max(1, decidable_total)
    u_recall = u_tp / max(1, u_total)
    
    if decidable_total > 0:
        primary_metric, primary_score = "accuracy_decidable", accuracy_decidable
    else:
        primary_metric, primary_score = "u_recall", u_recall
    
    ci_lo, ci_hi = wilson_ci(primary_score, decidable_total if decidable_total > 0 else u_total)

    return {
        "suite_name": suite_name,
        "metric": primary_metric,
        "score": float(primary_score),
        "avg_thought_tokens": float(avg_thoughts),
        "ci_lower": float(ci_lo),
        "ci_upper": float(ci_hi),
        "n_cases": n,
        "details": {
            "n_decidable": decidable_total,
            "n_undecidable": u_total,
            "accuracy_decidable": float(accuracy_decidable),
            "u_recall": float(u_recall),
            "avg_thoughts": float(avg_thoughts),
            "format_compliance": float(format_ok_total / n),
            "debug_samples": debug_samples,
        },
    }