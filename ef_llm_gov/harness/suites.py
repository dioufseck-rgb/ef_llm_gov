import json
import logging
from pathlib import Path
from typing import List, Dict, Any

# Setup basic logging for suite validation
logger = logging.getLogger(__name__)

def load_minpairs_suite(path: Path) -> List[Dict[str, Any]]:
    """
    Loads and validates a Minimal Pairs suite.
    Ensures every case has the required logic fields.
    """
    if not path.exists():
        raise FileNotFoundError(f"Suite file not found: {path}")

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        
        # Validation & Normalization Loop
        validated_data = []
        for i, case in enumerate(data):
            # Check required fields
            required = ["question", "option_a", "option_b", "expected"]
            missing = [f for f in required if f not in case]
            if missing:
                raise ValueError(f"Case {i} in {path.name} is missing keys: {missing}")
            
            # Normalize labels (ensure 'A' instead of 'a')
            case["expected"] = str(case["expected"]).strip().upper()
            if case["expected"] not in {"A", "B", "U"}:
                raise ValueError(f"Case {i} in {path.name} has invalid expected label: {case['expected']}")
            
            validated_data.append(case)
            
        return validated_data

    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON in {path}: {e}")


def load_abstention_suite(path: Path) -> List[Dict[str, Any]]:
    """
    Loads and validates an Abstention suite.
    Ensures 'should_abstain' is a proper boolean.
    """
    if not path.exists():
        raise FileNotFoundError(f"Suite file not found: {path}")

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        
        validated_data = []
        for i, case in enumerate(data):
            # Check required fields
            if "prompt" not in case:
                raise ValueError(f"Case {i} in {path.name} is missing 'prompt'")
            
            if "should_abstain" not in case:
                raise ValueError(f"Case {i} in {path.name} is missing 'should_abstain'")
            
            # Coerce should_abstain to actual boolean
            case["should_abstain"] = bool(case["should_abstain"])
            
            validated_data.append(case)
            
        return validated_data

    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON in {path}: {e}")

def get_suite_stats(path: Path, suite_type: str) -> Dict[str, Any]:
    """
    Utility for the Orchestrator to report the 'Weight' of the supply chain.
    """
    try:
        if suite_type == "minpairs":
            data = load_minpairs_suite(path)
            return {
                "count": len(data),
                "u_count": sum(1 for c in data if c["expected"] == "U"),
                "decidable_count": sum(1 for c in data if c["expected"] in {"A", "B"})
            }
        else:
            data = load_abstention_suite(path)
            return {
                "count": len(data),
                "abstain_count": sum(1 for c in data if c["should_abstain"]),
                "answer_count": sum(1 for c in data if not c["should_abstain"])
            }
    except Exception:
        return {"count": 0, "error": "Could not calculate stats"}