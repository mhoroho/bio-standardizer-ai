# src/llm_client_openai.py
from __future__ import annotations
import json, os, re
from pathlib import Path
from typing import Any, Dict
from openai import OpenAI

# ---------- key loading ----------
def _get_api_key() -> str:
    key = os.getenv("OPENAI_API_KEY")
    if key:
        return key.strip()
    key_path = Path(__file__).resolve().parent.parent / "API_Key.txt"
    if key_path.exists():
        text = key_path.read_text(encoding="utf-8").strip()
        if text:
            return text
    raise EnvironmentError("No OpenAI API key found. Set OPENAI_API_KEY or add API_Key.txt at repo root.")

def _client() -> OpenAI:
    return OpenAI(api_key=_get_api_key())

# ---------- json parsing helper ----------
def _parse_json_loose(text: str) -> Dict[str, Any]:
    # try direct
    try:
        return json.loads(text)
    except Exception:
        pass
    # strip fences/backticks
    cleaned = re.sub(r"^```(?:json)?\s*|```$", "", text.strip(), flags=re.I | re.M).strip()
    try:
        return json.loads(cleaned)
    except Exception:
        pass
    # find first {...}
    m = re.search(r"\{(?:[^{}]|(?R))*\}", cleaned, flags=re.S)
    if not m:
        raise ValueError("No JSON found in model output.")
    block = re.sub(r",\s*([}\]])", r"\1", m.group(0))  # remove trailing commas
    return json.loads(block)

# ---------- public API ----------
def openai_json(prompt: str, model: str = "gpt-4o-mini", temperature: float = 0.1) -> Dict[str, Any]:
    """
    Return a JSON object from OpenAI regardless of SDK feature set.
    Tries:
      1) Responses API with response_format (newer SDKs)
      2) Responses API without response_format (older 1.x/2.x variants)
      3) Chat Completions (broadest compatibility)
    """
    client = _client()

    # 1) Responses API with response_format (best)
    try:
        resp = client.responses.create(
            model=model,
            input=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=temperature,
        )
        return json.loads(resp.output_text)
    except TypeError:
        # this SDK doesn't support response_format
        pass
    except AttributeError:
        # responses API missing in this SDK
        pass

    # 2) Responses API without response_format (instruct JSON in prompt)
    try:
        resp = client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": "Return ONLY valid JSON. No commentary, no code fences."},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
        )
        return _parse_json_loose(resp.output_text)
    except AttributeError:
        # responses API missing
        pass

    # 3) Chat Completions fallback (works on older SDKs)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Return ONLY valid JSON. No commentary, no code fences."},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
    )
    return _parse_json_loose(resp.choices[0].message.content)
