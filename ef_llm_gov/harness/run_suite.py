"""
Run the Experimental-Frame LLM Evaluation Suite.

This script:
- Enumerates eligible Gemini text models
- Runs atomic and composite cognitive suites
- Produces a unified capability ledger

MVP assumptions:
- Forced-choice A/B for min-pairs
- Deterministic + production_default configs
"""

from __future__ import annotations

import json
import time
import random
from pathlib import Path
from typing import Dict, Any, List

from ef_llm_gov.adapters.gemini_api import GeminiAPIAdapter
from ef_llm_gov.harness.eval_minpairs import evaluate_minpairs
from ef_llm_gov.harness.eval_abstention import evaluate_abstention
from ef_llm_gov.harness.ledger import init_ledger, write_ledger
from ef_llm_gov.harness.suites import load_minpairs_suite, load_abstention_suite
from ef_llm_gov.configs.generation_configs import GENERATION_CONFIGS


# -------------------------
# Configuration
# -------------------------

SUITES_DIR = Path(__file__).parent.parent / "suites"
OUT_DIR = Path(__file__).parent.parent.parent / "out"
OUT_DIR.mkdir(exist_ok=True)

MAX_MODELS = 5            # MVP limit
JITTER_RANGE = (0.6, 1.2) # seconds


# Atomic suites
ATOMIC_MINPAIRS = {
    "negation_scope": "negation_minpairs.json",
    "exception_handling": "exception_minpair.json",
}

ABSTENTION_SUITE = {
    "abstention_reliability": "abstension_missing_info.json"
}

# Composite suites (2-way)
COMPOSITE_MINPAIRS = {
    "negation_x_exception": "neg_x_exc_minpairs.json",
    "temporal_x_exception": "temporal_x_exception_minpairs.json",
    "negation_x_conditional": "negation_x_conditional_minpairs.json",
}

# Composite suites (3-way)
COMPOSITE_3WAY_MINPAIRS = {
    "negation_x_exception_x_temporal": "neg_x_exc_x_temp_minpairs.json"
}


# -------------------------
# Main runner
# -------------------------

def main() -> None:
    adapter = GeminiAPIAdapter()

    # 1) Discover models
    models = adapter.list_models()
    models = models[:MAX_MODELS]

    ledger = init_ledger()

    # 2) Iterate models × configs × suites
    for model in models:
        model_name = model["name"]
        print(f"\n=== Evaluating model: {model_name} ===")

        for config_name, gen_cfg in GENERATION_CONFIGS.items():
            print(f"--- Config: {config_name} ---")

            # ---- Atomic min-pairs ----
            for suite_name, file_name in ATOMIC_MINPAIRS.items():
                suite_path = SUITES_DIR / file_name
                cases = load_minpairs_suite(suite_path)

                result = evaluate_minpairs(
                    adapter=adapter,
                    model_name=model_name,
                    generation_config=gen_cfg,
                    suite_name=suite_name,
                    cases=cases,
                )

                ledger.record(
                    model=model_name,
                    config=config_name,
                    suite=suite_name,
                    result=result,
                )

                _sleep_jitter()

            # ---- Composite (2-way) ----
            for suite_name, file_name in COMPOSITE_MINPAIRS.items():
                suite_path = SUITES_DIR / file_name
                cases = load_minpairs_suite(suite_path)

                result = evaluate_minpairs(
                    adapter=adapter,
                    model_name=model_name,
                    generation_config=gen_cfg,
                    suite_name=suite_name,
                    cases=cases,
                )

                ledger.record(
                    model=model_name,
                    config=config_name,
                    suite=suite_name,
                    result=result,
                )

                _sleep_jitter()

            # ---- Composite (3-way) ----
            for suite_name, file_name in COMPOSITE_3WAY_MINPAIRS.items():
                suite_path = SUITES_DIR / file_name
                cases = load_minpairs_suite(suite_path)

                result = evaluate_minpairs(
                    adapter=adapter,
                    model_name=model_name,
                    generation_config=gen_cfg,
                    suite_name=suite_name,
                    cases=cases,
                )

                ledger.record(
                    model=model_name,
                    config=config_name,
                    suite=suite_name,
                    result=result,
                )

                _sleep_jitter()

            # ---- Abstention ----
            for suite_name, file_name in ABSTENTION_SUITE.items():
                suite_path = SUITES_DIR / file_name
                cases = load_abstention_suite(suite_path)

                result = evaluate_abstention(
                    adapter=adapter,
                    model_name=model_name,
                    generation_config=gen_cfg,
                    suite_name=suite_name,
                    cases=cases,
                )

                ledger.record(
                    model=model_name,
                    config=config_name,
                    suite=suite_name,
                    result=result,
                )

                _sleep_jitter()

    # 3) Write ledger
    out_path = OUT_DIR / "capability_ledger.json"
    write_ledger(ledger, out_path)
    print(f"\nLedger written to {out_path}")


def _sleep_jitter() -> None:
    time.sleep(random.uniform(*JITTER_RANGE))


if __name__ == "__main__":
    main()
