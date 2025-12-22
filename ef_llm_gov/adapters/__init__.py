"""
Adapters for external LLM providers.

Each adapter should expose:
- list_models()
- get_model(model_name)
- generate(...)
"""

from .gemini_api import GeminiAPIAdapter

__all__ = ["GeminiAPIAdapter"]
