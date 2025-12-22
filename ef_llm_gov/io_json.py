from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Type, TypeVar

from .models import (
    TaskHeader, TaskContract, RoutingDecision,
    ModelCapabilityProfile, CapabilityScore,
    ControlRequirement, StabilityCondition, RoleContract,
    ContextProfile, TaskSignature
)

T = TypeVar("T")


def _to_jsonable(obj: Any) -> Any:
    """Convert dataclasses recursively to JSON-serializable dicts."""
    if is_dataclass(obj):
        return {k: _to_jsonable(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_jsonable(x) for x in obj]
    return obj


def dump_json(obj: Any, path: str | Path, *, indent: int = 2) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(_to_jsonable(obj), f, indent=indent, ensure_ascii=False)


def load_json(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


# ---- Builders (dict -> dataclass) ----

def task_header_from_dict(d: Dict[str, Any]) -> TaskHeader:
    return TaskHeader(**d)


def capability_score_from_dict(d: Dict[str, Any]) -> CapabilityScore:
    return CapabilityScore(**d)


def model_profile_from_dict(d: Dict[str, Any]) -> ModelCapabilityProfile:
    caps = [capability_score_from_dict(x) for x in d.get("capabilities", [])]
    return ModelCapabilityProfile(
        model_id=d["model_id"],
        capabilities=caps,
        max_context_tokens=d.get("max_context_tokens", 8192),
        tooling_supported=d.get("tooling_supported", []),
        last_evaluated_at=d.get("last_evaluated_at", ""),
        harness_version=d.get("harness_version", "harness-0.1.0"),
    )


def ledger_from_path(path: str | Path) -> Dict[str, ModelCapabilityProfile]:
    """
    Expects JSON like:
    {
      "model_A": {...ModelCapabilityProfile...},
      "model_B": {...}
    }
    """
    raw = load_json(path)
    return {mid: model_profile_from_dict(profile) for mid, profile in raw.items()}


def ledger_to_path(ledger: Dict[str, ModelCapabilityProfile], path: str | Path) -> None:
    dump_json({k: _to_jsonable(v) for k, v in ledger.items()}, path)


def control_from_dict(d: Dict[str, Any]) -> ControlRequirement:
    return ControlRequirement(**d)


def stability_from_dict(d: Dict[str, Any]) -> StabilityCondition:
    return StabilityCondition(**d)


def role_contract_from_dict(d: Dict[str, Any]) -> RoleContract:
    return RoleContract(**d)


def context_profile_from_dict(d: Dict[str, Any]) -> ContextProfile:
    return ContextProfile(**d)


def task_signature_from_dict(d: Dict[str, Any]) -> TaskSignature:
    # stored as break_ in our dataclass
    if "break_" not in d and "break" in d:
        d["break_"] = d.pop("break")
    return TaskSignature(**d)


def task_contract_from_dict(d: Dict[str, Any]) -> TaskContract:
    return TaskContract(
        schema_version=d["schema_version"],
        task_header=d["task_header"],
        role_contract=role_contract_from_dict(d["role_contract"]),
        context_profile=context_profile_from_dict(d["context_profile"]),
        task_signature=task_signature_from_dict(d["task_signature"]),
        required_stability_conditions=[stability_from_dict(x) for x in d.get("required_stability_conditions", [])],
        required_controls=[control_from_dict(x) for x in d.get("required_controls", [])],
        threshold_policy=d.get("threshold_policy", {}),
        audit=d.get("audit", {}),
    )


def routing_decision_from_dict(d: Dict[str, Any]) -> RoutingDecision:
    return RoutingDecision(
        schema_version=d["schema_version"],
        decision=d["decision"],
        selected_models=d.get("selected_models", []),
        applied_controls=[control_from_dict(x) for x in d.get("applied_controls", [])],
        reasons=d.get("reasons", []),
        failed_requirements=d.get("failed_requirements", []),
    )
