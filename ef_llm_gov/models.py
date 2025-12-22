from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


from .config import (
    SCHEMA_VERSION, TASK_TYPES, ROLES, RISKS, EVIDENCE, SOURCES, OUTPUT_MODES,
    ABSTAIN_POLICIES, HUMAN_REVIEW, TOOLS_ALLOWED
)


@dataclass
class TaskHeader:
    task_type: str
    role: str
    risk: str
    evidence: str
    sources: str
    output_mode: str
    output_schema_id: Optional[str] = None
    abstain_policy: str = "normal"
    human_review: str = "none"
    tools_allowed: List[str] = field(default_factory=list)
    domain_tags: List[str] = field(default_factory=list)
    schema_version: str = SCHEMA_VERSION

    def validate(self) -> None:
        if self.task_type not in TASK_TYPES:
            raise ValueError(f"task_type must be one of {sorted(TASK_TYPES)}")
        if self.role not in ROLES:
            raise ValueError(f"role must be one of {sorted(ROLES)}")
        if self.risk not in RISKS:
            raise ValueError(f"risk must be one of {sorted(RISKS)}")
        if self.evidence not in EVIDENCE:
            raise ValueError(f"evidence must be one of {sorted(EVIDENCE)}")
        if self.sources not in SOURCES:
            raise ValueError(f"sources must be one of {sorted(SOURCES)}")
        if self.output_mode not in OUTPUT_MODES:
            raise ValueError(f"output_mode must be one of {sorted(OUTPUT_MODES)}")
        if self.output_mode == "json_schema" and not self.output_schema_id:
            raise ValueError("output_schema_id is required when output_mode == 'json_schema'")
        if self.abstain_policy not in ABSTAIN_POLICIES:
            raise ValueError(f"abstain_policy must be one of {sorted(ABSTAIN_POLICIES)}")
        if self.human_review not in HUMAN_REVIEW:
            raise ValueError(f"human_review must be one of {sorted(HUMAN_REVIEW)}")
        for t in self.tools_allowed:
            if t not in TOOLS_ALLOWED:
                raise ValueError(f"tools_allowed contains invalid tool '{t}'")


@dataclass
class ControlRequirement:
    control_id: str
    required: bool
    params: Dict[str, Any] = field(default_factory=dict)
    notes: str = ""


@dataclass
class StabilityCondition:
    primitive_id: str
    threshold: float
    test_profiles: List[str]
    weight: float = 1.0
    notes: str = ""


@dataclass
class RoleContract:
    role: str
    allowed_actions: List[str]
    disallowed_actions: List[str]
    epistemic_limits: List[str]


@dataclass
class ContextProfile:
    risk: str
    evidence_regime: str
    sources_regime: str
    stakes: Dict[str, str] = field(default_factory=lambda: {"reversibility": "medium", "harm_potential": "medium"})
    domain_tags: List[str] = field(default_factory=list)


@dataclass
class TaskSignature:
    preserve: List[str]
    relax: List[str]
    break_: List[str]  # 'break' reserved


@dataclass
class RuleHit:
    rule_id: str
    evidence: List[str]


@dataclass
class TaskContract:
    schema_version: str
    task_header: Dict[str, Any]
    role_contract: RoleContract
    context_profile: ContextProfile
    task_signature: TaskSignature
    required_stability_conditions: List[StabilityCondition]
    required_controls: List[ControlRequirement]
    threshold_policy: Dict[str, Any]
    audit: Dict[str, Any]






@dataclass
class RoutingDecision:
    schema_version: str
    decision: str  # allow | allow_with_controls | route | abstain | require_human | block
    selected_models: List[str]
    applied_controls: List["ControlRequirement"]
    reasons: List[Dict[str, str]]
    failed_requirements: List[Dict[str, Any]] = field(default_factory=list)

    # --- New (backward-compatible) fields ---
    decision_id: str = ""
    timestamp: str = ""
    fallback_models: List[str] = field(default_factory=list)

    policy: Dict[str, Any] = field(default_factory=dict)
    requirements_summary: Dict[str, Any] = field(default_factory=dict)
    evidence: Dict[str, Any] = field(default_factory=dict)
    actions: List[Dict[str, Any]] = field(default_factory=list)
    audit: Dict[str, Any] = field(default_factory=dict)



@dataclass
class CapabilityScore:
    primitive_id: str
    test_profile: str
    score: float
    ci_lower: float
    ci_upper: float
    n_cases: int


@dataclass
class ModelCapabilityProfile:
    model_id: str
    capabilities: List[CapabilityScore]
    max_context_tokens: int = 8192
    tooling_supported: List[str] = field(default_factory=list)
    last_evaluated_at: str = ""
    harness_version: str = "harness-0.1.0"

    def index(self) -> Dict[Tuple[str, str], CapabilityScore]:
        return {(c.primitive_id, c.test_profile): c for c in self.capabilities}
