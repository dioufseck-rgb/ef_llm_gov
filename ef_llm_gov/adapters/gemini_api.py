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
    if feedback := resp.get("promptFeedback"):
        if reason := feedback.get("blockReason"):
            return "", f"BLOCKED_PROMPT_{reason}"

    candidates = resp.get("candidates") or []
    if not candidates:
        return "", "NO_CANDIDATES"

    c0 = candidates[0] or {}
    finish_reason = c0.get("finishReason", "UNKNOWN")

    content = c0.get("content") or {}
    parts = content.get("parts") or []
    
    text_chunks: List[str] = []
    for p in parts:
        if not isinstance(p, dict):
            continue
        if t := p.get("text"):
            if isinstance(t, str) and t.strip():
                text_chunks.append(t)
    
    if not text_chunks and isinstance(content.get("parts"), str):
        text_chunks.append(content["parts"])

    return "\n".join(text_chunks).strip(), finish_reason


def _is_text_generation_model(model: Dict[str, Any]) -> Tuple[bool, str]:
    methods = set(model.get("supportedGenerationMethods", []) or [])
    if "generateContent" not in methods:
        return False, "method_unsupported"

    input_m = [str(m).upper() for m in (model.get("inputModalities", []) or [])]
    output_m = [str(m).upper() for m in (model.get("outputModalities", []) or [])]

    if input_m and "TEXT" not in input_m:
        return False, "no_text_input_capability"
    if output_m and "TEXT" not in output_m:
        return False, "no_text_output_capability"

    name = model.get("name", "").lower()
    excluded_keywords = ["embeddings", "aqa", "imagen", "medlm"]
    if any(k in name for k in excluded_keywords):
        return False, "specialized_model_type"

    return True, ""


def _ensure_min_max_output_tokens(
    generation_config: Dict[str, Any],
    minimum_floor: int = 8192,
) -> Dict[str, Any]:
    cfg = dict(generation_config or {})
    for k in ["max_tokens", "max_output_tokens", "maxOutputTokens"]:
        if k in cfg:
            cfg["maxOutputTokens"] = cfg.pop(k)

    requested = cfg.get("maxOutputTokens")
    try:
        val = int(requested) if requested is not None else 0
        cfg["maxOutputTokens"] = max(val, minimum_floor)
    except:
        cfg["maxOutputTokens"] = 16384

    budget = cfg.pop("thinking_budget", None)
    if budget is not None:
        budget_int = int(budget)
        if budget_int > 0:
            cfg["thinkingConfig"] = {
                "includeThoughts": True,
                "thinkingBudget": budget_int
            }
    
    return cfg


class GeminiAPIAdapter:
    def __init__(self, api_key: str | None = None, debug_dir: str | None = None):
        self.api_key = api_key or os.getenv("GEMINI_KEY") or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_KEY not set")
        self.debug_dir = debug_dir

    def _headers(self) -> Dict[str, str]:
        return {"Content-Type": "application/json"}

    def list_models(self) -> List[Dict[str, Any]]:
        url = f"{GEMINI_BASE_URL}/models?key={self.api_key}"
        try:
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            all_models = r.json().get("models", [])
            return [m for m in all_models if _is_text_generation_model(m)[0]]
        except Exception as e:
            print(f"[SYSTEM] Model discovery failed: {e}")
            return []

    def generate(
        self,
        model: str,
        prompt: str,
        generation_config: Dict[str, Any],
        safety_settings: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        call_model = _normalize_model_for_call(model)
        url = f"{GEMINI_BASE_URL}/models/{call_model}:generateContent?key={self.api_key}"

        cfg = _ensure_min_max_output_tokens(generation_config)
        payload = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": cfg,
        }

        if safety_settings:
            payload["safetySettings"] = safety_settings

        r = requests.post(url, headers=self._headers(), json=payload, timeout=120)
        r.raise_for_status()
        resp = r.json()

        text, finish_reason = _extract_text_and_finish(resp)
        usage = resp.get("usageMetadata") or {}
        thought_tokens = usage.get("thoughtsTokenCount", 0)
        
        clean_ans = text[:30].replace('\n', ' ')
        print(f"[{call_model:22}] Thoughts: {thought_tokens:5} | Ans: {clean_ans:30} | Finish: {finish_reason}")

        return {
            "text": text, 
            "finish_reason": finish_reason, 
            "thought_tokens": thought_tokens, 
            "raw": resp
        }