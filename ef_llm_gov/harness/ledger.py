# ef_llm_gov/ledger.py

import json
import time
from pathlib import Path
from typing import Dict, Any, Optional


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
                "last_updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
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
        """
        Records a suite result.
        
        Interface Note: Maintains compatibility with evaluate_minpairs and 
        evaluate_abstention outputs.
        """
        if model not in self.data["models"]:
            self.data["models"][model] = {"configs": {}}
        
        if config not in self.data["models"][model]["configs"]:
            self.data["models"][model]["configs"][config] = {"suites": {}}
            
        # Enrich result with a timestamp for auditability
        result["evaluated_at_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        
        self.data["models"][model]["configs"][config]["suites"][suite] = result
        
        # LOGGING for terminal visibility
        score = result.get("score", 0.0)
        metric = result.get("metric", "unknown")
        print(f"[LEDGER] Recorded {model} ({config}) -> {suite}: {metric}={score:.4f}")

    def get_lccm(self, suite_name: str, threshold: float = 0.90) -> Optional[str]:
        """
        Strategic Method: Finds the Least-Cost Competent Model.
        Returns the model name that passes the threshold for a given primitive.
        """
        # Logic for 'Cost' ranking (Manual for now, can be enriched in _meta)
        # Note: In a bank, Flash is prioritized over Pro if scores are comparable.
        cost_priority = ["gemini-1.5-flash", "gemini-2.0-flash", "gemini-2.5-flash", 
                         "gemini-1.5-pro", "gemini-2.5-pro"]
        
        eligible_models = []
        for model_name, m_data in self.data["models"].items():
            for cfg_name, c_data in m_data["configs"].items():
                suite_res = c_data["suites"].get(suite_name)
                if suite_res and suite_res.get("score", 0) >= threshold:
                    eligible_models.append(model_name)
        
        # Return the first one based on our cost-priority list
        for candidate in cost_priority:
            for em in eligible_models:
                if candidate in em.lower():
                    return em
        return None

    def write(self, out_dir: Path) -> None:
        """
        Persists the ledger to disk. 
        Uses atomic-style write (temporary file swap) to prevent corruption.
        """
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "capability_ledger.json"
        temp_path = out_dir / "capability_ledger.json.tmp"
        
        # Update meta timestamp
        self.data["_meta"]["last_updated_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        # Write to temp file first
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)
        
        # Move temp to final
        temp_path.replace(path)

        print(f"[LEDGER] Persistent state saved to: {path}")


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
            "inter_request_sleep_range": [0.6, 1.2],
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