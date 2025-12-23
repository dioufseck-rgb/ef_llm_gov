from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

VALID_CHOICES = {"A", "B", "U"}


@dataclass
class MinPairsResult:
    score: float
    metric: str
    ci_lower: float
    ci_upper: float
    n_cases: int
    details: Dict[str, Any]


def build_minpairs_prompt(case: Dict[str, Any]) -> str:
    """
    Builds an Experimental-Frame prompt for a single min-pairs case.

    Supports optional fields:
      - policy: str  (prepended context)
      - question: str
      - option_a: str
      - option_b: str

    Output alphabet: A, B, U
      - A/B: uniquely determined
      - U: underdetermined by the policy/text (multiple reasonable interpretations)
    """
    policy = (case.get("policy") or "").strip()
    question = (case.get("question") or "").strip()
    a = (case.get("option_a") or "").strip()
    b = (case.get("option_b") or "").strip()

    parts: List[str] = []
    parts.append(
        "Instruction: Choose A or B if the policy/text uniquely determines the correct option. "
        "If it does not uniquely determine the answer (multiple reasonable interpretations lead to different outcomes), output U. "
        "Output exactly one character: A, B, or U."
    )

    if policy:
        parts.append(f"\nPolicy:\n{policy}")

    parts.append(f"\nQuestion:\n{question}")
    parts.append("\nOptions:")
    parts.append(f"A) {a}")
    parts.append(f"B) {b}")
    parts.append("\nAnswer (A/B/U):")

    return "\n".join(parts)


def normalize_choice(model_text: str) -> str:
    """
    Extract a single choice token (A/B/U) from model output.
    - Strips whitespace
    - Scans left-to-right for first valid choice char
    - Returns 'INVALID' if no valid choice found
    """
    t = (model_text or "").strip().upper()
    for ch in t:
        if ch in VALID_CHOICES:
            return ch
    return "INVALID"


def wilson_ci(p: float, n: int, z: float = 1.96) -> Tuple[float, float]:
    """
    Wilson score interval for a Bernoulli proportion.
    """
    if n <= 0:
        return 0.0, 1.0
    denom = 1.0 + (z * z) / n
    center = (p + (z * z) / (2.0 * n)) / denom
    half = (z / denom) * ((p * (1.0 - p) / n + (z * z) / (4.0 * n * n)) ** 0.5)
    lo = max(0.0, center - half)
    hi = min(1.0, center + half)
    return lo, hi


def evaluate_minpairs(
    adapter,
    model_name: str,
    generation_config: Dict[str, Any],
    suite_name: str,
    cases: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Evaluates min-pairs with output alphabet {A,B,U}.

    Metrics:
      - accuracy_decidable: accuracy on cases with expected in {A,B}
      - u_recall: correct-U / total-U
      - u_precision: correct-U / predicted-U
      - format_compliance: outputs in {A,B,U} / total
      - (primary) score + CI returned uses:
          * accuracy_decidable if there is at least one decidable item
          * otherwise u_recall (for pure-U suites)
    """
    n = len(cases)
    decidable_total = 0
    decidable_correct = 0

    u_total = 0
    u_tp = 0
    u_pred_total = 0

    format_ok_total = 0

    http_error_counts: Dict[str, int] = {}
    finish_reason_counts: Dict[str, int] = {}

    debug_samples = []

    for i, case in enumerate(cases):
        prompt = build_minpairs_prompt(case)

        try:
            resp = adapter.generate(
            model=model_name,
            prompt=prompt,
            generation_config=generation_config,
        )
            text = resp.get("text", "")
            finish_reason = resp.get("finish_reason", "UNKNOWN")
            finish_reason_counts[finish_reason] = finish_reason_counts.get(finish_reason, 0) + 1

        except Exception as e:
            # If your adapter exposes structured HTTP errors, map them here.
            # We'll bucket by exception name as a minimal fallback.
            key = type(e).__name__
            http_error_counts[key] = http_error_counts.get(key, 0) + 1
            text = ""
            finish_reason_counts["ERROR"] = finish_reason_counts.get("ERROR", 0) + 1

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
        else:
            # Unknown label in suite definition; treat as invalid suite data
            pass

        u_pred_total += int(pred == "U")

        if len(debug_samples) < 3:
            debug_samples.append(
                {
                    "case_index": i,
                    "expected": exp,
                    "predicted": pred,
                    "raw_text": (text[:200] + "…") if len(text) > 200 else text,
                }
            )

    # Derived metrics
    accuracy_decidable = decidable_correct / max(1, decidable_total)
    u_recall = u_tp / max(1, u_total)
    u_precision = u_tp / max(1, u_pred_total)
    format_compliance = format_ok_total / max(1, n)

    # Primary score selection
    if decidable_total > 0:
        primary_metric = "accuracy_decidable"
        primary_score = accuracy_decidable
        ci_lo, ci_hi = wilson_ci(primary_score, decidable_total)
        ci_n = decidable_total
    else:
        # pure-U suite
        primary_metric = "u_recall"
        primary_score = u_recall
        ci_lo, ci_hi = wilson_ci(primary_score, u_total)
        ci_n = u_total

    return {
        "suite_name": suite_name,
        "metric": primary_metric,
        "score": float(primary_score),
        "ci_lower": float(ci_lo),
        "ci_upper": float(ci_hi),
        "n_cases": n,
        "details": {
            "n_decidable": decidable_total,
            "n_undecidable": u_total,
            "accuracy_decidable": float(accuracy_decidable),
            "u_recall": float(u_recall),
            "u_precision": float(u_precision),
            "format_compliance": float(format_compliance),
            "http_error_counts": http_error_counts,
            "finish_reason_counts": finish_reason_counts,
            "debug_samples": debug_samples,
            "ci_n": ci_n,
        },
    }
