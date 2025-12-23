from __future__ import annotations

import os
import time
import random
from pathlib import Path
from typing import Dict, Iterable, Tuple

from ef_llm_gov.adapters.gemini_api import GeminiAPIAdapter
from ef_llm_gov.harness.eval_minpairs import evaluate_minpairs
from ef_llm_gov.harness.eval_abstention import evaluate_abstention
from ef_llm_gov.harness.ledger import init_ledger, write_ledger
from ef_llm_gov.harness.suites import load_minpairs_suite, load_abstention_suite
from ef_llm_gov.configs.generation_configs import GENERATION_CONFIGS


# -------------------------
# Paths & runtime knobs
# -------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
SUITES_DIR = REPO_ROOT / "ef_llm_gov" / "suites"
OUT_DIR = REPO_ROOT / "out"
OUT_DIR.mkdir(exist_ok=True)

MAX_MODELS = int(os.getenv("MAX_MODELS", "1"))
JITTER_RANGE = (0.6, 1.2)


# -------------------------
# Suite catalog
# -------------------------

ATOMIC_MINPAIRS: Dict[str, str] = {
    "negation_scope": "negation_minpairs.json",
    "exception_handling": "exception_minpair.json",
    "positional_control": "positional_control_minpairs.json",
}

COMPOSITE_MINPAIRS: Dict[str, str] = {
    "negation_x_exception": "neg_x_exc_decidable.json",
    "neg_x_exc_undecidable_blind": "neg_x_exc_undecidable_blind.json",
    "temporal_x_exception": "temporal_x_exception_minpairs.json",
    "negation_x_conditional": "negation_x_conditional_minpairs.json",
}

COMPOSITE_3WAY_MINPAIRS: Dict[str, str] = {
    "negation_x_exception_x_temporal": "neg_x_exc_x_temp_minpairs.json",
}

ABSTENTION_SUITES: Dict[str, str] = {
    "abstention_reliability": "abstension_missing_info.json",
}


# -------------------------
# Helpers
# -------------------------

def _sleep_jitter() -> None:
    time.sleep(random.uniform(*JITTER_RANGE))


def _iter_existing_suites(
    suites: Dict[str, str],
    suites_dir: Path,
) -> Iterable[Tuple[str, Path]]:
    for suite_name, filename in suites.items():
        path = suites_dir / filename
        if not path.exists():
            print(f"[WARN] Missing suite file, skipping: {suite_name} -> {path}")
            continue
        yield suite_name, path


# -------------------------
# Main runner
# -------------------------

def main() -> None:
    adapter = GeminiAPIAdapter()

    # Discover models
    eligible_models = adapter.list_models()
    eligible_model_count = len(eligible_models)

    models = eligible_models[:MAX_MODELS]
    evaluated_model_count = len(models)

    if not models:
        raise RuntimeError("No models returned by adapter.list_models()")

    ledger = init_ledger(
        surface="gemini_api",
        max_models_evaluated=MAX_MODELS,
        eligible_model_count=eligible_model_count,
        evaluated_model_count=evaluated_model_count,
    )

    for model in models:
        model_name = model["name"]
        print(f"\n=== Evaluating model: {model_name} ===")

        for config_name, gen_cfg in GENERATION_CONFIGS.items():
            print(f"\n--- Config: {config_name} ---")

            # Atomic & control suites
            for suite_name, suite_path in _iter_existing_suites(ATOMIC_MINPAIRS, SUITES_DIR):
                cases = load_minpairs_suite(suite_path)
                result = evaluate_minpairs(
                    adapter=adapter,
                    model_name=model_name,
                    generation_config=gen_cfg,
                    suite_name=suite_name,
                    cases=cases,
                )
                ledger.record(model_name, config_name, suite_name, result)
                _sleep_jitter()

            # Composite (2-way)
            for suite_name, suite_path in _iter_existing_suites(COMPOSITE_MINPAIRS, SUITES_DIR):
                cases = load_minpairs_suite(suite_path)
                result = evaluate_minpairs(
                    adapter=adapter,
                    model_name=model_name,
                    generation_config=gen_cfg,
                    suite_name=suite_name,
                    cases=cases,
                )
                ledger.record(model_name, config_name, suite_name, result)
                _sleep_jitter()

            # Composite (3-way)
            for suite_name, suite_path in _iter_existing_suites(COMPOSITE_3WAY_MINPAIRS, SUITES_DIR):
                cases = load_minpairs_suite(suite_path)
                result = evaluate_minpairs(
                    adapter=adapter,
                    model_name=model_name,
                    generation_config=gen_cfg,
                    suite_name=suite_name,
                    cases=cases,
                )
                ledger.record(model_name, config_name, suite_name, result)
                _sleep_jitter()

            # Abstention suites
            for suite_name, suite_path in _iter_existing_suites(ABSTENTION_SUITES, SUITES_DIR):
                cases = load_abstention_suite(suite_path)
                result = evaluate_abstention(
                    adapter=adapter,
                    model_name=model_name,
                    generation_config=gen_cfg,
                    suite_name=suite_name,
                    cases=cases,
                )
                ledger.record(model_name, config_name, suite_name, result)
                _sleep_jitter()

    write_ledger(ledger, OUT_DIR)
    print("\nLedger written to out/capability_ledger.json")


if __name__ == "__main__":
    main()
