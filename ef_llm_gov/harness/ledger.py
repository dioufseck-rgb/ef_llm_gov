import json
from pathlib import Path
from typing import Dict, Any


class Ledger:
    def __init__(
        self,
        surface: str,
        max_models_evaluated: int,
        eligible_model_count: int,
        evaluated_model_count: int,
        max_debug_samples_per_suite: int,
        backoff: Dict[str, Any],
    ):
        self.data = {
            "_meta": {
                "surface": surface,
                "max_models_evaluated": max_models_evaluated,
                "eligible_model_count": eligible_model_count,
                "evaluated_model_count": evaluated_model_count,
                "max_debug_samples_per_suite": max_debug_samples_per_suite,
                "backoff": backoff,
            },
            "models": {},
        }

    def record(
        self,
        model: str,
        config: str,
        suite: str,
        result: Dict[str, Any],
    ) -> None:
        if model not in self.data["models"]:
            self.data["models"][model] = {"configs": {}}
        if config not in self.data["models"][model]["configs"]:
            self.data["models"][model]["configs"][config] = {"suites": {}}
        self.data["models"][model]["configs"][config]["suites"][suite] = result

    def write(self, out_dir: Path) -> None:
        (out_dir / "capability_ledger.json").write_text(
            json.dumps(self.data, indent=2),
            encoding="utf-8",
        )

        print("Wrote:")
        print(" - out/capability_ledger.json")


def init_ledger(
    surface: str = "gemini_api",
    max_models_evaluated: int = 5,
    eligible_model_count: int = 0,
    evaluated_model_count: int = 0,
    max_debug_samples_per_suite: int = 3,
    backoff: Dict[str, Any] = None,
) -> Ledger:
    if backoff is None:
        backoff = {
            "max_retries": 6,
            "base_backoff_s": 1.0,
            "max_backoff_s": 20.0,
            "inter_request_sleep_range": (0.6, 1.2),
        }
    return Ledger(
        surface=surface,
        max_models_evaluated=max_models_evaluated,
        eligible_model_count=eligible_model_count,
        evaluated_model_count=evaluated_model_count,
        max_debug_samples_per_suite=max_debug_samples_per_suite,
        backoff=backoff,
    )


def write_ledger(ledger: Ledger, out_dir: Path) -> None:
    ledger.write(out_dir)