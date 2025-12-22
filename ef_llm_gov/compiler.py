from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Set
import re

from .config import (
    SCHEMA_VERSION, PRIMITIVES,
    BASE_THRESHOLDS, PRIMITIVE_OVERRIDES, DEFAULT_TEST_PROFILES,
    ROLE_CONTRACTS, TASK_TYPE_TEMPLATES, PROMPT_CUE_RULES
)
from .models import (
    TaskHeader, TaskContract, TaskSignature, RoleContract, ContextProfile,
    ControlRequirement, StabilityCondition, RuleHit
)


def _dedupe_controls(controls: List[ControlRequirement]) -> List[ControlRequirement]:
    by_id: Dict[str, ControlRequirement] = {}
    for c in controls:
        if c.control_id not in by_id:
            by_id[c.control_id] = ControlRequirement(c.control_id, c.required, dict(c.params), c.notes)
        else:
            e = by_id[c.control_id]
            e.required = e.required or c.required
            e.params.update(c.params or {})
            if c.notes and c.notes not in e.notes:
                e.notes = (e.notes + " | " + c.notes).strip(" |")
    return list(by_id.values())


def _threshold_for_primitive(risk: str, primitive_id: str) -> float:
    base = BASE_THRESHOLDS.get(risk, 1.0)
    override = PRIMITIVE_OVERRIDES.get(primitive_id, {}).get(risk)
    return max(base, override if override is not None else 0.0)


def _default_profiles_for_primitive(primitive_id: str) -> List[str]:
    return list(DEFAULT_TEST_PROFILES.get(primitive_id, [f"{primitive_id.upper()}_v1"]))


def _normalize_signature(preserve: Set[str], relax: Set[str], break_: Set[str]):
    relax = set(relax) - set(preserve)
    break_ = set(break_) - set(preserve) - set(relax)
    return sorted(preserve), sorted(relax), sorted(break_)


class FrameCompiler:
    def __init__(self, compiler_version: str = "frame-compiler-0.1.0"):
        self.compiler_version = compiler_version

    def compile(self, header: TaskHeader, prompt_text: str, parameter_schema: Optional[Dict[str, Any]] = None) -> TaskContract:
        header.validate()

        if header.risk == "prohibited":
            return self._blocked_contract(header, reason="risk=prohibited")

        tpl = TASK_TYPE_TEMPLATES[header.task_type]
        preserve: Set[str] = set(tpl["preserve"])
        relax: Set[str] = set(tpl["relax"])
        break_: Set[str] = set(tpl["break"])

        role_cfg = ROLE_CONTRACTS[header.role]
        role_contract = RoleContract(
            role=header.role,
            allowed_actions=role_cfg["allowed_actions"],
            disallowed_actions=role_cfg["disallowed_actions"],
            epistemic_limits=role_cfg["epistemic_limits"],
        )

        # Conservative: preserve frame boundaries for all roles
        preserve.add("frame_boundary_preservation")

        context_profile = ContextProfile(
            risk=header.risk,
            evidence_regime=header.evidence,
            sources_regime=header.sources,
            stakes={"reversibility": "medium", "harm_potential": "medium"},
            domain_tags=list(header.domain_tags),
        )

        rule_hits: List[RuleHit] = []
        controls: List[ControlRequirement] = []

        # Header rules
        if header.output_mode == "json_schema":
            preserve.update({"schema_validity", "field_binding"})
            controls.extend([
                ControlRequirement("require_json_schema", True, {"schema_id": header.output_schema_id}, "Structured output requested."),
                ControlRequirement("postcheck_schema", True, {}, "Validate JSON against schema."),
            ])
            rule_hits.append(RuleHit("H001_output_json_schema", ["output_mode=json_schema"]))

        if header.evidence == "required":
            preserve.update({"evidence_traceability", "citation_integrity", "no_claim_without_evidence"})
            controls.extend([
                ControlRequirement("require_citations", True, {}, "Evidence required by header."),
                ControlRequirement("no_claim_without_evidence", True, {}, "Block unsupported claims."),
                ControlRequirement("postcheck_citations", True, {}, "Verify citation spans/ids."),
            ])
            rule_hits.append(RuleHit("H002_evidence_required", ["evidence=required"]))

        if header.sources == "provided_only":
            controls.append(ControlRequirement("require_rag", True, {"corpus": "provided"}, "Sources restricted to provided docs."))
            rule_hits.append(RuleHit("H003_sources_provided_only_requires_rag", ["sources=provided_only"]))

        if header.risk == "high":
            preserve.update({"abstention_reliability", "uncertainty_calibration"})
            controls.append(ControlRequirement("abstain_on_low_confidence", True, {"min_confidence": 0.6}, "High risk posture."))
            rule_hits.append(RuleHit("H004_risk_high_abstain_and_human_review", ["risk=high"]))

        if header.abstain_policy == "strict":
            preserve.add("abstention_reliability")
            controls.append(ControlRequirement("abstain_on_missing_inputs", True, {"ask_clarifying_questions": True}, "Strict abstention policy."))
            rule_hits.append(RuleHit("H005_abstain_policy_strict", ["abstain_policy=strict"]))

        if header.human_review == "required":
            controls.append(ControlRequirement("human_review_required", True, {"reason": "header"}, "Human review required by header."))
            rule_hits.append(RuleHit("H006_human_review_required", ["human_review=required"]))

        # Prompt cue rules
        prompt_lower = prompt_text.lower()
        test_profiles_map: Dict[str, Set[str]] = {}

        for rule in PROMPT_CUE_RULES:
            hits = []
            for pat in rule["patterns"]:
                if re.search(pat, prompt_lower):
                    hits.append(pat)
            if not hits:
                continue

            preserve.update(rule.get("add_preserve", set()))
            for prim, profs in (rule.get("test_profiles", {}) or {}).items():
                test_profiles_map.setdefault(prim, set()).update(profs)

            for c in (rule.get("add_controls", []) or []):
                controls.append(ControlRequirement(c["control_id"], c["required"], c.get("params", {}), c.get("notes", "")))

            rule_hits.append(RuleHit(rule["rule_id"], hits))

        # Normalize signature
        preserve_list, relax_list, break_list = _normalize_signature(preserve, relax, break_)

        # Derive stability conditions from preserve
        stability_conditions: List[StabilityCondition] = []
        for prim in preserve_list:
            if prim not in PRIMITIVES:
                continue
            threshold = _threshold_for_primitive(header.risk, prim)
            profiles = set(_default_profiles_for_primitive(prim))
            profiles.update(test_profiles_map.get(prim, set()))
            stability_conditions.append(
                StabilityCondition(
                    primitive_id=prim,
                    threshold=threshold,
                    test_profiles=sorted(profiles),
                    weight=1.0,
                )
            )

        # Ensure implied controls
        if "no_claim_without_evidence" in preserve_list:
            controls.append(ControlRequirement("no_claim_without_evidence", True, {}, "Implied by preserve set."))

        controls = _dedupe_controls(controls)

        threshold_policy = {
            "risk": header.risk,
            "base_thresholds": dict(BASE_THRESHOLDS),
            "overrides": [
                {"primitive_id": pid, "threshold": PRIMITIVE_OVERRIDES[pid][header.risk], "reason": "primitive_override"}
                for pid in PRIMITIVE_OVERRIDES
                if header.risk in PRIMITIVE_OVERRIDES[pid]
            ],
        }

        audit = {
            "compiler_version": self.compiler_version,
            "rule_hits": [asdict(rh) for rh in rule_hits],
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }

        return TaskContract(
            schema_version=SCHEMA_VERSION,
            task_header=asdict(header),
            role_contract=role_contract,
            context_profile=context_profile,
            task_signature=TaskSignature(preserve_list, relax_list, break_list),
            required_stability_conditions=stability_conditions,
            required_controls=controls,
            threshold_policy=threshold_policy,
            audit=audit,
        )

    def _blocked_contract(self, header: TaskHeader, reason: str) -> TaskContract:
        role_cfg = ROLE_CONTRACTS[header.role]
        role_contract = RoleContract(
            role=header.role,
            allowed_actions=role_cfg["allowed_actions"],
            disallowed_actions=role_cfg["disallowed_actions"],
            epistemic_limits=role_cfg["epistemic_limits"],
        )
        context_profile = ContextProfile(
            risk=header.risk,
            evidence_regime=header.evidence,
            sources_regime=header.sources,
            stakes={"reversibility": "low", "harm_potential": "high"},
            domain_tags=list(header.domain_tags),
        )
        return TaskContract(
            schema_version=SCHEMA_VERSION,
            task_header=asdict(header),
            role_contract=role_contract,
            context_profile=context_profile,
            task_signature=TaskSignature([], [], []),
            required_stability_conditions=[],
            required_controls=[ControlRequirement("human_review_required", True, {"reason": reason}, "Blocked contract.")],
            threshold_policy={"risk": header.risk, "base_thresholds": dict(BASE_THRESHOLDS), "overrides": []},
            audit={
                "compiler_version": self.compiler_version,
                "rule_hits": [{"rule_id": "BLOCK", "evidence": [reason]}],
                "timestamp": datetime.utcnow().isoformat() + "Z",
            },
        )
