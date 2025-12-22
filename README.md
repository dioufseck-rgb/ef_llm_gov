# ef_llm_gov

Experimental-Frame LLM Governance MVP

A Python library for governing Large Language Model (LLM) usage through structured task definitions, capability profiling, and eligibility routing.

## Overview

`ef_llm_gov` provides a framework for:

- **Task Definition**: Specify LLM tasks with detailed headers including risk levels, evidence requirements, and operational constraints
- **Contract Compilation**: Transform task specifications into executable contracts
- **Capability Profiling**: Define and score model capabilities across various dimensions
- **Eligibility Routing**: Automatically route tasks to appropriate models based on capability matching

## Installation

This MVP uses only Python standard library modules. Requires Python 3.10+.

```bash
git clone https://github.com/dioufseck-rgb/ef_llm_gov.git
cd ef_llm_gov
pip install -r requirements.txt  # Currently empty, but may include optional dependencies
```

Optional dependencies for extended functionality:
- PyYAML>=6.0.1 (for YAML task header input)
- jsonschema>=4.23.0 (for JSON schema validation)

## Quick Start

See `demo.py` for a complete example. Here's a simplified version:

```python
from ef_llm_gov import (
    TaskHeader, FrameCompiler, EligibilityGate,
    ModelCapabilityProfile, CapabilityScore
)

# Define a task
header = TaskHeader(
    task_type="grounded_qa",
    role="analyst",
    risk="high",
    evidence="required",
    sources="provided_only",
    output_mode="bullets",
    abstain_policy="strict",
    tools_allowed=["retrieval"],
    domain_tags=["ops", "compliance"]
)

# Compile into a contract
compiler = FrameCompiler()
contract = compiler.compile(header, "Your prompt here...")

# Define model capabilities
ledger = {
    "model_A": ModelCapabilityProfile(
        model_id="model_A",
        capabilities=[
            CapabilityScore("evidence_traceability", "EVIDENCE_TRACEABILITY_v1", 0.95, 0.93, 0.97, 500),
            # ... more capabilities
        ]
    )
}

# Route to eligible models
gate = EligibilityGate()
decision = gate.decide(contract, ledger)

print(f"Selected models: {decision.selected_models}")
```

## Core Components

### TaskHeader
Defines the parameters of an LLM task:
- `task_type`: Type of task (e.g., "grounded_qa", "creative_writing")
- `role`: Expected role (e.g., "analyst", "assistant")
- `risk`: Risk level ("low", "medium", "high")
- `evidence`: Evidence requirements ("none", "preferred", "required")
- `sources`: Source constraints ("any", "provided_only")
- `output_mode`: Output format ("text", "bullets", "json_schema")
- `abstain_policy`: When to abstain ("normal", "strict")
- `tools_allowed`: List of allowed tools
- `domain_tags`: Domain-specific tags

### FrameCompiler
Compiles TaskHeader + prompt into a TaskContract with governance frames.

### EligibilityGate
Evaluates model capabilities against task requirements to determine eligible models.

### ModelCapabilityProfile
Contains capability scores for a model across various dimensions like evidence traceability, citation integrity, etc.

## Running the Demo

```bash
python demo.py
```

This will generate:
- `out/task_contract.json`: The compiled task contract
- `out/routing_decision.json`: The eligibility decision
- `out/capability_ledger.json`: The model capability profiles

## API Reference

### Models
- `TaskHeader`: Task specification dataclass
- `TaskContract`: Compiled contract with governance frames
- `RoutingDecision`: Eligibility decision result
- `ModelCapabilityProfile`: Model capability profile
- `CapabilityScore`: Individual capability score with confidence intervals

### Core Classes
- `FrameCompiler`: Compiles headers into contracts
- `EligibilityGate`: Makes routing decisions

### I/O Utilities
- `dump_json()`: Save objects to JSON
- `load_json()`: Load objects from JSON
- `ledger_to_path()`: Save capability ledger
- `ledger_from_path()`: Load capability ledger

## Contributing

This is an experimental MVP. Contributions welcome for:
- Additional capability dimensions
- New task types and validation rules
- Integration with LLM APIs
- Performance optimizations

## License

[Add license information here]