# ef_llm_gov

LLM Capability Evaluation Framework

A Python framework for systematically evaluating Large Language Model (LLM) capabilities across multiple dimensions using structured test suites and API adapters.

## Overview

`ef_llm_gov` provides tools for:

- **Adapters**: API interfaces to various LLM providers (currently supports Google Gemini)
- **Harness**: Evaluation execution engine with support for minimal pairs, abstention tests, and custom suites
- **Suites**: Pre-defined test cases covering capabilities like negation scope, exception handling, temporal reasoning, and abstention reliability

## Installation

Requires Python 3.10+. Uses Python standard library plus `requests` for API calls.

```bash
git clone https://github.com/dioufseck-rgb/ef_llm_gov.git
cd ef_llm_gov
pip install requests
```

## Quick Start

1. Set your API key:
   ```bash
   export GEMINI_KEY=your_gemini_api_key
   ```

2. Run the evaluation suites:
   ```bash
   PYTHONPATH=/workspaces/ef_llm_gov python ef_llm_gov/harness/run_suite.py
   ```

This will evaluate available Gemini models against all test suites and generate a capability ledger in `out/capability_ledger.json`.

## Core Components

### Adapters

API wrappers for LLM providers located in `ef_llm_gov/adapters/`.

#### GeminiAPIAdapter

- **File**: `ef_llm_gov/adapters/gemini_api.py`
- **Purpose**: Interface to Google Gemini API
- **Features**:
  - Model listing and metadata retrieval
  - Text generation with configurable parameters
  - Error handling and response parsing
- **Configuration**: Set `GEMINI_KEY` or `GEMINI_API_KEY` environment variable

### Harness

Evaluation execution tools in `ef_llm_gov/harness/`.

#### run_suite.py

- **Purpose**: Main evaluation runner
- **Features**:
  - Discovers and runs all test suites
  - Evaluates multiple models with different generation configs
  - Generates capability ledger with scores and metadata
- **Output**: `out/capability_ledger.json`

#### eval_minpairs.py

- **Purpose**: Evaluates minimal pairs test cases
- **Features**:
  - Compares model responses on paired examples
  - Calculates accuracy and consistency scores
  - Supports various capability dimensions

#### eval_abstention.py

- **Purpose**: Tests model abstention behavior
- **Features**:
  - Evaluates reliability of abstention on missing information
  - Measures false positive/negative rates

#### ledger.py

- **Purpose**: Capability scoring and aggregation
- **Features**:
  - Initializes and updates capability ledgers
  - Aggregates results across models and configs

#### suites.py

- **Purpose**: Suite loading utilities
- **Features**:
  - Loads minimal pairs and abstention suites from JSON
  - Validates suite structure

### Suites

Test case definitions in `ef_llm_gov/suites/`.

#### Minimal Pairs Suites

Test cases designed to probe specific capabilities through controlled comparisons:

- `negation_minpairs.json`: Negation scope understanding
- `exception_minpair.json`: Exception handling in conditional logic
- `positional_control_minpairs.json`: Positional constraints
- `neg_x_exc_decidable.json`: Negation + exception (decidable)
- `neg_x_exc_undecidable_blind.json`: Negation + exception (undecidable)
- `temporal_x_exception_minpairs.json`: Temporal + exception reasoning
- `negation_x_conditional_minpairs.json`: Negation + conditional logic
- `neg_x_exc_x_temp_minpairs.json`: Three-way: negation + exception + temporal

#### Abstention Suites

- `abstension_missing_info.json`: Tests for proper abstention when information is missing

## Configuration

### Generation Configs

Defined in `ef_llm_gov/configs/generation_configs.py`:

- `default`: Standard generation parameters
- `creative`: Higher temperature for creative tasks
- `precise`: Lower temperature for analytical tasks

### Runtime Options

Environment variables:
- `MAX_MODELS`: Maximum number of models to evaluate (default: 3)
- `GEMINI_KEY` or `GEMINI_API_KEY`: API key for Gemini

## Output

Results are saved to the `out/` directory:

- `capability_ledger.json`: Comprehensive evaluation results with scores, confidence intervals, and sample counts

## Extending the Framework

### Adding New Adapters

1. Create a new adapter class in `ef_llm_gov/adapters/`
2. Implement the adapter interface (see `GeminiAPIAdapter` for reference)
3. Update `run_suite.py` to use the new adapter

### Adding New Suites

1. Create JSON test cases following the existing format
2. Place in `ef_llm_gov/suites/`
3. Add to the suite catalog in `run_suite.py`

### Custom Evaluations

Use the harness components directly:

```python
from ef_llm_gov.adapters.gemini_api import GeminiAPIAdapter
from ef_llm_gov.harness.eval_minpairs import evaluate_minpairs
from ef_llm_gov.harness.suites import load_minpairs_suite

adapter = GeminiAPIAdapter()
cases = load_minpairs_suite("path/to/suite.json")
result = evaluate_minpairs(
    adapter=adapter,
    model_name="gemini-1.5-pro",
    generation_config={},
    suite_name="custom_suite",
    cases=cases
)
```

## Demo

For the original governance framework demo, see `demo.py`.

## Contributing

Contributions welcome for:
- Additional LLM provider adapters
- New capability test suites
- Evaluation metrics and scoring methods
- Performance improvements

## License

[Add license information here]