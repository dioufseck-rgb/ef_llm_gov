# ef_llm_gov/adapters/gemini_api.py

from __future__ import annotations

import os
import json
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional

import requests

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"


def _extract_text_and_finish(resp: dict) -> tuple[str, str]:
    finish_reason = "UNKNOWN"
    text_chunks: list[str] = []

    candidates = resp.get("candidates") or []
    if not candidates:
        return "", finish_reason

    c0 = candidates[0] or {}
    finish_reason = c0.get("finishReason", "UNKNOWN") or "UNKNOWN"

    content = c0.get("content") or {}
    parts = content.get("parts") or []

    # parts is usually a list of dicts, but be defensive
    for p in parts:
        if isinstance(p, dict):
            # common case
            t = p.get("text")
            if isinstance(t, str) and t.strip():
                text_chunks.append(t)
        elif isinstance(p, str) and p.strip():
            text_chunks.append(p)

    return ("\n".join(text_chunks)).strip(), finish_reason


def _is_text_generation_model(model: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Returns (is_eligible, reason_if_excluded).

    Eligibility rule (MVP, robust-ish):
    - Must support generateContent (text-gen API surface)
    - Must NOT advertise image/vision modality if modalities are provided
    """
    methods = set(model.get("supportedGenerationMethods", []) or [])
    if "generateContent" not in methods:
        return False, "does_not_support_generateContent"

    modalities = (model.get("inputModalities", []) or []) + (model.get("outputModalities", []) or [])
    modalities_l = {str(m).lower() for m in modalities}

    if "image" in modalities_l or "vision" in modalities_l:
        return False, "image_or_vision_modality"

    return True, ""


def _ensure_min_max_output_tokens(generation_config: Dict[str, Any], minimum: int = 16, default_if_missing: int = 32) -> Dict[str, Any]:
    """
    Gemini requests can silently fail with finishReason=MAX_TOKENS if maxOutputTokens is
    absent/0/small under some model variants or prompt regimes.

    We force a sane floor (>=minimum). For forced-choice tasks, 16–32 is ample.
    """
    cfg = dict(generation_config or {})

    # Normalize common alternate keys (defensive)
    if "max_output_tokens" in cfg and "maxOutputTokens" not in cfg:
        cfg["maxOutputTokens"] = cfg.pop("max_output_tokens")

    mot = cfg.get("maxOutputTokens", None)
    if mot is None:
        cfg["maxOutputTokens"] = max(minimum, default_if_missing)
        return cfg

    try:
        mot_i = int(mot)
    except Exception:
        cfg["maxOutputTokens"] = max(minimum, default_if_missing)
        return cfg

    cfg["maxOutputTokens"] = max(minimum, mot_i)
    return cfg

def _normalize_model_for_call(model_name: str) -> str:
    """
    Gemini catalog returns names like 'models/gemini-2.5-pro'.
    The generateContent endpoint expects 'gemini-2.5-pro'.
    """
    if model_name.startswith("models/"):
        return model_name[len("models/"):]
    return model_name

class GeminiAPIAdapter:
    """
    Thin, explicit adapter for Gemini API.

    Changes in this version:
    - Enforce maxOutputTokens >= 16 (prevents empty outputs + finishReason=MAX_TOKENS in practice)
    - Optional diagnostics logging on anomalous finishReasons / errors

    Diagnostics:
    - Set env GEMINI_DEBUG_DIR to a directory path (e.g., "out/gemini_debug")
      to persist payload+response for debugging.
    """



    def __init__(self, api_key: str | None = None, debug_dir: str | None = None):
        self.api_key = api_key or os.getenv("GEMINI_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not set")

        self.debug_dir = debug_dir or os.getenv("GEMINI_DEBUG_DIR")
        if self.debug_dir:
            os.makedirs(self.debug_dir, exist_ok=True)

    def _headers(self) -> Dict[str, str]:
        return {"Content-Type": "application/json"}

    def _write_debug(self, tag: str, record: Dict[str, Any]) -> None:
        if not self.debug_dir:
            return
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%fZ")
        path = os.path.join(self.debug_dir, f"{ts}_{tag}.json")
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(record, f, indent=2, ensure_ascii=False)
        except Exception:
            # Never break the main flow due to debug logging.
            pass

    def list_models_raw(self) -> List[Dict[str, Any]]:
        url = f"{GEMINI_BASE_URL}/models?key={self.api_key}"
        r = requests.get(url, headers=self._headers(), timeout=30)
        r.raise_for_status()
        return r.json().get("models", []) or []

    def list_models(self) -> List[Dict[str, Any]]:
        models = self.list_models_raw()
        included: List[Dict[str, Any]] = []
        for m in models:
            ok, _reason = _is_text_generation_model(m)
            if ok:
                included.append(m)
        return included

    def list_models_snapshot(self) -> Dict[str, Any]:
        models = self.list_models_raw()
        included: List[Dict[str, Any]] = []
        excluded: List[Dict[str, Any]] = []

        for m in models:
            ok, reason = _is_text_generation_model(m)
            if ok:
                included.append(m)
            else:
                excluded.append(
                    {
                        "name": m.get("name"),
                        "reason": reason,
                        "supportedGenerationMethods": m.get("supportedGenerationMethods", []),
                        "inputModalities": m.get("inputModalities", []),
                        "outputModalities": m.get("outputModalities", []),
                    }
                )

        return {"included": included, "excluded": excluded}

    def get_model(self, model_name: str) -> Dict[str, Any]:
        url = f"{GEMINI_BASE_URL}/models/{model_name}?key={self.api_key}"
        r = requests.get(url, headers=self._headers(), timeout=30)
        r.raise_for_status()
        return r.json()



    def generate(
    self,
    model: str,
    prompt: str,
    generation_config: Dict[str, Any],
    safety_settings: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:

        call_model = _normalize_model_for_call(model)

        url = f"{GEMINI_BASE_URL}/models/{call_model}:generateContent?key={self.api_key}"

        cfg = _ensure_min_max_output_tokens(
            generation_config,
            minimum=16,
            default_if_missing=32,
        )

        payload: Dict[str, Any] = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": cfg,
        }

        if safety_settings is not None:
            payload["safetySettings"] = safety_settings

        try:
            r = requests.post(url, headers=self._headers(), json=payload, timeout=60)
            r.raise_for_status()
            resp = r.json()
        except requests.exceptions.HTTPError as e:
            status = getattr(e.response, "status_code", None)
            body = None
            try:
                body = e.response.text if e.response is not None else None
            except Exception:
                body = None

            self._write_debug(
                tag="http_error",
                record={
                    "catalog_model": model,
                    "call_model": call_model,
                    "status_code": status,
                    "error": str(e),
                    "response_text": body,
                    "payload": payload,
                },
            )
            raise
        except requests.exceptions.RequestException as e:
            self._write_debug(
                tag="request_exception",
                record={
                    "catalog_model": model,
                    "call_model": call_model,
                    "error": str(e),
                    "payload": payload,
                },
            )
            raise

        # Diagnostics on anomalous completion
        try:
            candidates = resp.get("candidates", []) or []
            if not candidates:
                self._write_debug(
                    tag="no_candidates",
                    record={
                        "catalog_model": model,
                        "call_model": call_model,
                        "payload": payload,
                        "response": resp,
                    },
                )
                return resp

            fr = candidates[0].get("finishReason", "UNKNOWN")
            if fr != "STOP":
                self._write_debug(
                    tag=f"finish_{fr}",
                    record={
                        "catalog_model": model,
                        "call_model": call_model,
                        "payload": payload,
                        "response": resp,
                    },
                )
        except Exception:
            pass



    # inside generate():
        text, finish_reason = _extract_text_and_finish(resp)
        return {"text": text, "finish_reason": finish_reason, "raw": resp}



        return resp

