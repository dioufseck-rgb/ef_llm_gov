# ef_llm_gov/adapters/gemini_api.py

from __future__ import annotations

import os
import json
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional

import requests

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"

def _normalize_model_for_call(model_name: str) -> str:
    m = (model_name or "").strip()
    if m.startswith("models/"):
        m = m[len("models/") :]
    return m

def _extract_text_and_finish(resp: Dict[str, Any]) -> Tuple[str, str]:
    """
    Robustly extract text parts, ignoring internal thought parts.
    """
    feedback = resp.get("promptFeedback", {})
    if feedback.get("blockReason"):
        return "", f"BLOCKED_PROMPT_{feedback.get('blockReason')}"

    candidates = resp.get("candidates") or []
    if not candidates:
        return "", "NO_CANDIDATES"

    c0 = candidates[0] or {}
    finish_reason = c0.get("finishReason", "UNKNOWN")
    content = c0.get("content") or {}
    parts = content.get("parts") or []
    
    text_chunks: List[str] = []
    for p in parts:
        if isinstance(p, dict) and "text" in p:
            text_chunks.append(p["text"])
    
    final_text = "\n".join(text_chunks).strip()

    if not final_text and finish_reason == "MAX_TOKENS":
        usage = resp.get("usageMetadata", {})
        print(f"[GEMINI_DEBUG] Still hit MAX_TOKENS. Thoughts: {usage.get('thoughtsTokenCount')}")

    return final_text, finish_reason

def _ensure_min_max_output_tokens(
    generation_config: Dict[str, Any],
    minimum_floor: int = 8192,
    default_total: int = 16384,
) -> Dict[str, Any]:
    """
    Forces a very large token budget so reasoning doesn't 'eat' the answer.
    """
    cfg = dict(generation_config or {})

    for k in ["max_tokens", "max_output_tokens", "maxOutputTokens"]:
        if k in cfg:
            cfg["maxOutputTokens"] = cfg.pop(k)

    requested_mot = cfg.get("maxOutputTokens")
    
    if requested_mot is None:
        cfg["maxOutputTokens"] = default_total
    else:
        try:
            val = int(requested_mot)
            if val < minimum_floor:
                # We raise this to ensure the model has 'thinking' room
                cfg["maxOutputTokens"] = minimum_floor
            else:
                cfg["maxOutputTokens"] = val
        except:
            cfg["maxOutputTokens"] = default_total
            
    return cfg

class GeminiAPIAdapter:
    """
    Adapter for Gemini REST API with high-budget reasoning support.
    """

    def __init__(self, api_key: str | None = None, debug_dir: str | None = None):
        self.api_key = api_key or os.getenv("GEMINI_KEY") or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_KEY not set")
        self.debug_dir = debug_dir or os.getenv("GEMINI_DEBUG_DIR")

    def _headers(self) -> Dict[str, str]:
        return {"Content-Type": "application/json"}

    def list_models(self) -> List[Dict[str, Any]]:
        url = f"{GEMINI_BASE_URL}/models?key={self.api_key}"
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        return [m for m in r.json().get("models", []) if "generateContent" in m.get("supportedGenerationMethods", [])]

    def generate(
        self,
        model: str,
        prompt: str,
        generation_config: Dict[str, Any],
        safety_settings: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        call_model = _normalize_model_for_call(model)
        url = f"{GEMINI_BASE_URL}/models/{call_model}:generateContent?key={self.api_key}"

        # Ensure the token limit is high enough for reasoning models
        cfg = _ensure_min_max_output_tokens(generation_config)

        payload: Dict[str, Any] = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": cfg,
        }

        if safety_settings:
            payload["safetySettings"] = safety_settings

        try:
            r = requests.post(url, headers=self._headers(), json=payload, timeout=120)
            r.raise_for_status()
            resp = r.json()
        except Exception as e:
            print(f"[GEMINI_ERROR] Model: {call_model} | Request failed: {str(e)}")
            raise

        text, finish_reason = _extract_text_and_finish(resp)

        # Truncate prompt for the log (first 60 chars)
        display_prompt = (prompt[:60] + "..") if len(prompt) > 60 else prompt
        display_prompt = display_prompt.replace("\n", " ")

        # Usage Metadata
        usage = resp.get("usageMetadata", {})
        thought_tokens = usage.get("thoughtsTokenCount", 0)
        
        # FINAL LOG STATEMENT
        print(f"[{call_model}] Q: {display_prompt} | Thoughts: {thought_tokens} | Answer: {text} | Finish: {finish_reason}")

        return {"text": text, "finish_reason": finish_reason, "raw": resp}