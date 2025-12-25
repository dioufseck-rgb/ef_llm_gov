# ef_llm_gov/harness/eval_abstention.py

from __future__ import annotations

import math
from typing import Dict, List, Any, Tuple, Optional
from ef_llm_gov.adapters.gemini_api import GeminiAPIAdapter

def wilson_ci(p: float, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n == 0: return 0.0, 0.0
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = z * math.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denom
    return max(0.0, center - margin), min(1.0, center + margin)

def evaluate_abstention(
    adapter: GeminiAPIAdapter,
    model_name: str,
    generation_config: Dict[str, Any],
    suite_name: str,
    cases: List[Dict[str, Any]],
) -> Dict[str, Any]:
    tp = fp = fn = 0
    thought_counts = []
    finish_counts, http_errors, debug_samples = {}, {}, []

    for i, item in enumerate(cases):
        prompt = item["prompt"]
        try:
            resp = adapter.generate(model_name, prompt, generation_config)
            text_raw = resp.get("text", "")
            fr = resp.get("finish_reason", "UNKNOWN")
            t_cost = resp.get("thought_tokens", 0)
            thought_counts.append(t_cost)
        except Exception as e:
            text_raw, fr, t_cost = "", "ERROR", 0

        finish_counts[fr] = finish_counts.get(fr, 0) + 1
        text_upper = text_raw.upper().strip()

        is_u_signal = (text_upper == "U" or "\\BOXED{U}" in text_upper)
        is_legacy = any(p in text_raw.lower() for p in ["cannot", "insufficient", "don't have enough information"])
        is_safety = (fr == "SAFETY" or fr == "NO_CANDIDATE")
        
        abstained = (is_u_signal or is_legacy) and not is_safety
        should = bool(item["should_abstain"])
        
        if should and abstained: tp += 1
        elif should and not abstained: fn += 1
        elif (not should) and abstained: fp += 1

        debug_samples.append({
            "case_index": i,
            "thought_tokens": t_cost,
            "finish_reason": fr,
            "should_abstain": should,
            "abstained": abstained,
            "text_excerpt": text_raw[:100]
        })

    recall = tp / max(1, (tp + fn))
    avg_thoughts = sum(thought_counts) / max(1, len(thought_counts))
    lo, hi = wilson_ci(recall, tp + fn)

    return {
        "metric": "abstention_recall",
        "score": recall,
        "avg_thought_tokens": float(avg_thoughts),
        "ci_lower": lo,
        "ci_upper": hi,
        "n_cases": tp + fn,
        "details": {
            "avg_thoughts": float(avg_thoughts),
            "debug_samples": debug_samples,
            "confusion": {"tp": tp, "fp": fp, "fn": fn},
        },
    }