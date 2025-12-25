from __future__ import annotations

import os
import time
import random
import traceback
from pathlib import Path
from typing import Dict, Any, List

from ef_llm_gov.adapters.gemini_api import GeminiAPIAdapter
from ef_llm_gov.harness.eval_minpairs import evaluate_minpairs
from ef_llm_gov.harness.eval_abstention import evaluate_abstention
from ef_llm_gov.harness.ledger import init_ledger, write_ledger
from ef_llm_gov.harness.suites import load_minpairs_suite, load_abstention_suite
from ef_llm_gov.configs.generation_configs import GENERATION_CONFIGS

# -------------------------
# Cognitive Tier Constants
# -------------------------
TIER_ATOMIC = "ATOMIC"             
TIER_COMPOSITE = "COMPOSITE"       
TIER_HIGH_LOGIC = "HIGH_LOGIC"     
TIER_GROUNDING = "GROUNDING"       

# -------------------------
# Suite Registry
# -------------------------
SUITE_REGISTRY = {
    "negation_scope": {
        "file": "negation_minpairs.json", 
        "tier": TIER_ATOMIC, 
        "type": "minpairs"
    },
    "exception_handling": {
        "file": "exception_minpair.json", 
        "tier": TIER_ATOMIC, 
        "type": "minpairs"
    },
    "positional_control": {
        "file": "positional_control_minpairs.json", 
        "tier": TIER_ATOMIC, 
        "type": "minpairs"
    },
    "temporal_x_exception": {
        "file": "temporal_x_exception_minpairs.json", 
        "tier": TIER_COMPOSITE, 
        "type": "minpairs"
    },
    "negation_x_conditional": {
        "file": "negation_x_conditional_minpairs.json", 
        "tier": TIER_COMPOSITE, 
        "type": "minpairs"
    },
    "negation_x_exception": {
        "file": "neg_x_exc_decidable.json", 
        "tier": TIER_COMPOSITE, 
        "type": "minpairs"
    },
    "deontic_nuance": {
        "file": "deontic_minpairs.json", 
        "tier": TIER_COMPOSITE, 
        "type": "minpairs"
    },
    "quantitative_boundaries": {
        "file": "quantitative_minpairs.json", 
        "tier": TIER_COMPOSITE, 
        "type": "minpairs"
    },
    "neg_x_exc_undecidable_blind": {
        "file": "neg_x_exc_undecidable_blind.json", 
        "tier": TIER_HIGH_LOGIC, 
        "type": "minpairs"
    },
    "negation_x_exception_x_temporal": {
        "file": "neg_x_exc_x_temp_minpairs.json", 
        "tier": TIER_HIGH_LOGIC, 
        "type": "minpairs"
    },
    "abstention_reliability": {
        "file": "abstension_missing_info.json", 
        "tier": TIER_GROUNDING, 
        "type": "abstention"
    },
}

# -------------------------
# Runtime Paths
# -------------------------
HARNESS_DIR = Path(__file__).resolve().parent
REPO_ROOT = HARNESS_DIR.parents[1]
SUITES_DIR = REPO_ROOT / "ef_llm_gov" / "suites"
OUT_DIR = REPO_ROOT / "out"

MAX_MODELS = int(os.getenv("MAX_MODELS", "6"))
JITTER_RANGE = (0.8, 1.5)

# -------------------------
# Logistics Helpers
# -------------------------

def _get_optimized_config(base_cfg: Dict[str, Any], tier: str) -> Dict[str, Any]:
    cfg = dict(base_cfg)
    budgets = {
        TIER_ATOMIC: 0,        
        TIER_GROUNDING: 1024,
        TIER_COMPOSITE: 2048,
        TIER_HIGH_LOGIC: 12000 
    }
    cfg["maxOutputTokens"] = 16384 
    cfg["thinking_budget"] = budgets.get(tier, 2048)
    return cfg

def _sleep_jitter() -> None:
    time.sleep(random.uniform(*JITTER_RANGE))

# -------------------------
# Main Orchestration Loop
# -------------------------

def main() -> None:
    adapter = GeminiAPIAdapter()

    # Discover models from the factory
    try:
        eligible_models = adapter.list_models()
    except Exception as e:
        print(f"[SYSTEM] CRITICAL ERROR listing models: {e}")
        return

    if not eligible_models:
        print("[SYSTEM] ERROR: No eligible models discovered.")
        return

    # RANDOMIZATION STEP: 
    # We shuffle the entire list so you can see variety beyond the Pro/Flash 2.5 standard.
    print(f"[SYSTEM] Randomizing evaluation order for {len(eligible_models)} models...")
    random.shuffle(eligible_models)

    models_to_evaluate = eligible_models[:MAX_MODELS]

    # Initialize the Registry
    ledger = init_ledger(
        surface="gemini_api",
        max_models_evaluated=MAX_MODELS,
        eligible_model_count=len(eligible_models),
        evaluated_model_count=len(models_to_evaluate),
    )

    for model_meta in models_to_evaluate:
        model_name = model_meta["name"]
        print(f"\n{'='*75}\nAUDITING MODEL: {model_name}\n{'='*75}")

        for profile_name, base_cfg in GENERATION_CONFIGS.items():
            print(f"\n>>> Running Profile: {profile_name}")

            for suite_name, metadata in SUITE_REGISTRY.items():
                suite_file = metadata["file"]
                suite_tier = metadata["tier"]
                suite_type = metadata["type"]
                suite_path = SUITES_DIR / suite_file
                
                if not suite_path.exists():
                    print(f"[LOGISTICS] WARNING: Skipping missing suite {suite_name} ({suite_file})")
                    continue

                optimized_cfg = _get_optimized_config(base_cfg, suite_tier)
                print(f"[LOGISTICS] {suite_name:35} | Tier: {suite_tier:12}")

                try:
                    if suite_type == "minpairs":
                        suite_data = load_minpairs_suite(suite_path)
                        result = evaluate_minpairs(
                            adapter=adapter,
                            model_name=model_name,
                            generation_config=optimized_cfg,
                            suite_name=suite_name,
                            cases=suite_data
                        )
                    elif suite_type == "abstention":
                        suite_data = load_abstention_suite(suite_path)
                        result = evaluate_abstention(
                            adapter=adapter,
                            model_name=model_name,
                            generation_config=optimized_cfg,
                            suite_name=suite_name,
                            cases=suite_data
                        )
                    else:
                        continue

                    # Record and Checkpoint
                    ledger.record(model_name, profile_name, suite_name, result)
                    write_ledger(ledger, OUT_DIR)

                except Exception:
                    print(f"[LOGISTICS] CRASH in suite {suite_name}")
                    traceback.print_exc()

                _sleep_jitter()

    print(f"\n[SYSTEM] Audit Complete. Capability Ledger located at: {OUT_DIR}")

if __name__ == "__main__":
    main()