# ef_llm_gov/gate.py

from __future__ import annotations

from typing import Dict, List, Tuple, Any
from datetime import datetime
import secrets

from .models import TaskContract, RoutingDecision, ModelCapabilityProfile
from .config import SCHEMA_VERSION


def _now_utc_iso() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def _make_decision_id(prefix: str = "rd") -> str:
    # short, unique-enough for logs; swap with UUID if you prefer
    return f"{prefix}_{datetime.utcnow().strftime('%Y%m%d_%H%M%SZ')}_{secrets.token_hex(2)}"


class EligibilityGate:
    """
    Gating policy (MVP):
    - Conservative: uses CI lower bound.
    - For each required primitive, requires at least ONE test_profile to meet threshold.
      (Tighten later to ALL profiles.)
    """

    MEETS_RULE = "ALL_PRIMITIVES_MET_ANY_PROFILE"
    MODE = "conservative_ci_lower"
    TIE_BREAKER = "max_margin_sum"

    def decide(self, contract: TaskContract, ledger: Dict[str, ModelCapabilityProfile]) -> RoutingDecision:
        decision_id = _make_decision_id()
        timestamp = _now_utc_iso()

        required_primitives = [r.primitive_id for r in contract.required_stability_conditions]
        controls_required = [c.control_id for c in contract.required_controls if c.required]

        policy = {
            "risk": contract.context_profile.risk,
            "mode": self.MODE,
            "meets_rule": self.MEETS_RULE,
            "tie_breaker": self.TIE_BREAKER,
        }

        requirements_summary = {
            "required_primitives": required_primitives,
            "controls_required": controls_required,
        }

        # 0) Prohibited short-circuit
        if contract.context_profile.risk == "prohibited":
            return RoutingDecision(
                schema_version=SCHEMA_VERSION,
                decision_id=decision_id,
                timestamp=timestamp,
                decision="block",
                selected_models=[],
                fallback_models=[],
                applied_controls=contract.required_controls,
                reasons=[{"type": "risk", "message": "risk=prohibited"}],
                policy=policy,
                requirements_summary=requirements_summary,
                evidence={"passed": [], "failed": [], "scorecard": {}},
                actions=[{"action_id": "REQUEST_HUMAN", "params": {"queue": "policy_review"}}],
                audit={
                    "task_contract_hash": "",  # add if you later hash contracts
                    "ledger_version": "",
                    "harness_version": "",
                    "notes": "",
                },
            )

        # 1) Human review override
        if any(c.control_id == "human_review_required" and c.required for c in contract.required_controls):
            return RoutingDecision(
                schema_version=SCHEMA_VERSION,
                decision_id=decision_id,
                timestamp=timestamp,
                decision="require_human",
                selected_models=[],
                fallback_models=[],
                applied_controls=contract.required_controls,
                reasons=[{"type": "control", "message": "Human review required by policy/header."}],
                policy=policy,
                requirements_summary=requirements_summary,
                evidence={"passed": [], "failed": [], "scorecard": {}},
                actions=[{"action_id": "REQUEST_HUMAN", "params": {"queue": "high_risk_review"}}],
                audit={"task_contract_hash": "", "ledger_version": "", "harness_version": "", "notes": ""},
            )

        # 2) No stability conditions => default allow
        if not contract.required_stability_conditions:
            default_model = next(iter(ledger.keys()), None)
            chosen = [default_model] if default_model else []
            decision = "allow_with_controls" if contract.required_controls else "allow"

            return RoutingDecision(
                schema_version=SCHEMA_VERSION,
                decision_id=decision_id,
                timestamp=timestamp,
                decision=decision,
                selected_models=chosen,
                fallback_models=[],
                applied_controls=contract.required_controls,
                reasons=[{"type": "invariance", "message": "No gating stability conditions; default routing."}],
                policy=policy,
                requirements_summary=requirements_summary,
                evidence={"passed": [], "failed": [], "scorecard": {}},
                actions=(
                    [{"action_id": "EXECUTE_LLM", "params": {"model_id": chosen[0], "controls": controls_required}}]
                    if chosen else
                    [{"action_id": "REQUEST_HUMAN", "params": {"queue": "triage"}}]
                ),
                audit={"task_contract_hash": "", "ledger_version": "", "harness_version": "", "notes": ""},
            )

        # 3) Evaluate each model
        eligible: List[Tuple[str, float]] = []
        failures_all: List[Dict[str, Any]] = []
        scorecard: Dict[str, Dict[str, float]] = {}

        # We'll also collect "passed" evidence for the final selected model
        per_model_passed: Dict[str, List[Dict[str, Any]]] = {}

        for model_id, profile in ledger.items():
            idx = profile.index()
            failed: List[Dict[str, Any]] = []
            passed: List[Dict[str, Any]] = []

            for req in contract.required_stability_conditions:
                met = False
                best_ci_lower = None
                best_tp = None

                for tp in req.test_profiles:
                    key = (req.primitive_id, tp)
                    if key in idx:
                        ci = idx[key].ci_lower
                        if best_ci_lower is None or ci > best_ci_lower:
                            best_ci_lower = ci
                            best_tp = tp
                        if ci >= req.threshold:
                            met = True
                            # for passed evidence, keep the best meeting profile if possible
                            # (not necessarily the first meeting; we still track best overall)
                if met:
                    passed.append({
                        "primitive_id": req.primitive_id,
                        "threshold": req.threshold,
                        "best_test_profile": best_tp,
                        "best_ci_lower": best_ci_lower,
                    })
                else:
                    failed.append({
                        "model_id": model_id,
                        "primitive_id": req.primitive_id,
                        "required_threshold": req.threshold,
                        "best_test_profile": best_tp,
                        "best_ci_lower": best_ci_lower,
                        "test_profiles": req.test_profiles,
                    })

            if failed:
                failures_all.extend(failed)
                continue

            # rank by total margin, track min margin too
            margin_sum = 0.0
            min_margin = None

            for req in contract.required_stability_conditions:
                best = 0.0
                for tp in req.test_profiles:
                    key = (req.primitive_id, tp)
                    if key in idx:
                        best = max(best, idx[key].ci_lower)
                margin = (best - req.threshold) * req.weight
                margin_sum += margin
                min_margin = margin if min_margin is None else min(min_margin, margin)

            eligible.append((model_id, margin_sum))
            scorecard[model_id] = {"margin_sum": float(margin_sum), "min_margin": float(min_margin or 0.0)}
            per_model_passed[model_id] = passed

        # 4) No eligible models => abstain/require_human
        if not eligible:
            risk = contract.context_profile.risk
            decision = "abstain" if risk in ("medium", "high") else "require_human"

            actions = [{"action_id": "REQUEST_HUMAN", "params": {"queue": "high_risk_review" if risk == "high" else "triage"}}]
            if any(c.control_id == "abstain_on_missing_inputs" for c in contract.required_controls):
                actions.append({
                    "action_id": "ASK_CLARIFYING_QUESTIONS",
                    "params": {"questions": ["Provide admissible sources/evidence.", "Confirm role authority and desired output format."]},
                })

            return RoutingDecision(
                schema_version=SCHEMA_VERSION,
                decision_id=decision_id,
                timestamp=timestamp,
                decision=decision,
                selected_models=[],
                fallback_models=[],
                applied_controls=contract.required_controls,
                reasons=[{"type": "invariance", "message": "No eligible model meets required stability thresholds under CI-lower policy."}],
                failed_requirements=failures_all[:50],
                policy=policy,
                requirements_summary=requirements_summary,
                evidence={
                    "passed": [],
                    "failed": failures_all[:50],
                    "scorecard": scorecard,
                },
                actions=actions,
                audit={"task_contract_hash": "", "ledger_version": "", "harness_version": "", "notes": ""},
            )

        # 5) Select best model and optional fallback
        eligible.sort(key=lambda x: x[1], reverse=True)
        selected = eligible[0][0]
        fallback = [eligible[1][0]] if len(eligible) > 1 else []

        decision = "allow_with_controls" if contract.required_controls else "allow"

        return RoutingDecision(
            schema_version=SCHEMA_VERSION,
            decision_id=decision_id,
            timestamp=timestamp,
            decision=decision,
            selected_models=[selected],
            fallback_models=fallback,
            applied_controls=contract.required_controls,
            reasons=[{"type": "selection", "message": f"Selected {selected} because it meets all required stability thresholds and has the largest margin under CI-lower policy."}],
            policy=policy,
            requirements_summary=requirements_summary,
            evidence={
                "passed": per_model_passed.get(selected, []),
                "failed": [],
                "scorecard": scorecard,
            },
            actions=[{
                "action_id": "EXECUTE_LLM",
                "params": {"model_id": selected, "controls": controls_required},
            }],
            audit={"task_contract_hash": "", "ledger_version": "", "harness_version": "", "notes": ""},
        )
