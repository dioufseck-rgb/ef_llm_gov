from __future__ import annotations

from dataclasses import asdict

from ef_llm_gov import (
    TaskHeader, FrameCompiler, EligibilityGate,
    ModelCapabilityProfile, CapabilityScore,
    dump_json, ledger_to_path
)

def main():
    header = TaskHeader(
        task_type="grounded_qa",
        role="analyst",
        risk="high",
        evidence="required",
        sources="provided_only",
        output_mode="bullets",
        abstain_policy="strict",
        tools_allowed=["retrieval"],
        domain_tags=["ops", "compliance"],
    )

    prompt = """
    Based on the attached incident report, explain what happened before and after the service restart.
    Do not speculate. Cite sources. Except for the maintenance window, focus on customer impact.
    """

    compiler = FrameCompiler()
    contract = compiler.compile(header, prompt)

    ledger = {
        "model_A": ModelCapabilityProfile(
            model_id="model_A",
            capabilities=[
                CapabilityScore("evidence_traceability", "EVIDENCE_TRACEABILITY_v1", 0.95, 0.93, 0.97, 500),
                CapabilityScore("citation_integrity", "CITATION_INTEGRITY_v1", 0.96, 0.94, 0.98, 500),
                CapabilityScore("no_claim_without_evidence", "NO_CLAIM_WO_EVIDENCE_v1", 0.94, 0.92, 0.96, 500),
                CapabilityScore("temporal_order", "TEMPORAL_ORDER_v1", 0.90, 0.87, 0.93, 400),
                CapabilityScore("exception_handling", "EXCEPTION_MINPAIRS_v1", 0.90, 0.86, 0.94, 400),
                CapabilityScore("negation_scope", "NEGATION_MINPAIRS_v1", 0.92, 0.89, 0.95, 400),
                CapabilityScore("abstention_reliability", "ABSTAIN_MISSING_INFO_v1", 0.97, 0.96, 0.99, 300),
                CapabilityScore("frame_boundary_preservation", "FRAME_BOUNDARY_PRESERVATION_v1", 0.90, 0.88, 0.92, 300),
                CapabilityScore("consistency", "CONSISTENCY_BASIC_v1", 0.92, 0.90, 0.94, 300),
                CapabilityScore("topic_relevance", "TOPIC_RELEVANCE_v1", 0.95, 0.94, 0.96, 300),
            ],
        ),
        "model_B": ModelCapabilityProfile(
            model_id="model_B",
            capabilities=[
                CapabilityScore("evidence_traceability", "EVIDENCE_TRACEABILITY_v1", 0.90, 0.88, 0.92, 500),
                CapabilityScore("citation_integrity", "CITATION_INTEGRITY_v1", 0.92, 0.90, 0.94, 500),
                CapabilityScore("no_claim_without_evidence", "NO_CLAIM_WO_EVIDENCE_v1", 0.88, 0.84, 0.92, 500),
                CapabilityScore("temporal_order", "TEMPORAL_ORDER_v1", 0.85, 0.82, 0.88, 400),
                CapabilityScore("exception_handling", "EXCEPTION_MINPAIRS_v1", 0.84, 0.80, 0.88, 400),
                CapabilityScore("negation_scope", "NEGATION_MINPAIRS_v1", 0.86, 0.83, 0.89, 400),
                CapabilityScore("abstention_reliability", "ABSTAIN_MISSING_INFO_v1", 0.90, 0.87, 0.93, 300),
                CapabilityScore("frame_boundary_preservation", "FRAME_BOUNDARY_PRESERVATION_v1", 0.85, 0.82, 0.88, 300),
                CapabilityScore("consistency", "CONSISTENCY_BASIC_v1", 0.88, 0.86, 0.90, 300),
                CapabilityScore("topic_relevance", "TOPIC_RELEVANCE_v1", 0.92, 0.90, 0.94, 300),
            ],
        )
    }

    gate = EligibilityGate()
    decision = gate.decide(contract, ledger)

    dump_json(contract, "out/task_contract.json")
    dump_json(decision, "out/routing_decision.json")
    ledger_to_path(ledger, "out/capability_ledger.json")

    print("Wrote:")
    print(" - out/task_contract.json")
    print(" - out/routing_decision.json")
    print(" - out/capability_ledger.json")
    print("\nDecision:", decision.decision, "Models:", decision.selected_models)

if __name__ == "__main__":
    main()
