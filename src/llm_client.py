from __future__ import annotations
import requests, json, re
from typing import Any, Dict, Tuple

OLLAMA_URL = "http://localhost:11434"

def _extract_json_block(text: str) -> Dict[str, Any]:
    """Try to parse a JSON object from text (code fences, extra prose tolerated)."""
    # fast path
    try:
        return json.loads(text)
    except Exception:
        pass
    # strip code fences/backticks
    cleaned = re.sub(r"^```(?:json)?\s*|```$", "", text.strip(), flags=re.I | re.M).strip()
    try:
        return json.loads(cleaned)
    except Exception:
        pass
    # find first {...} block
    m = re.search(r"\{(?:[^{}]|(?R))*\}", cleaned, flags=re.S)
    if not m:
        raise ValueError("No JSON found in model output")
    block = m.group(0)
    # remove common trailing commas
    block = re.sub(r",\s*([}\]])", r"\1", block)
    return json.loads(block)

def _generate(payload: Dict[str, Any], timeout: float = 120.0) -> str:
    r = requests.post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    return data.get("response", "")

def ollama_json(
    prompt: str,
    model: str = "gpt-oss:20b",
    temperature: float = 0.1,
    timeout: float = 600.0,
    num_ctx: int = 8192,
    num_thread: int = 8,
    max_retries: int = 3,
    debug: bool = False,
) -> Dict[str, Any]:
    """
    Try: (1) format=json (strict), (2) no format + reinforced prompt, (3) JSON-fixer prompt.
    Returns parsed dict or raises ValueError with last raw text in debug mode.
    """
    last_raw = ""
    # 1) Strict JSON mode (some models ignore, but try)
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature, "num_ctx": num_ctx},
        "format": "json",
    }
    try:
        last_raw = _generate(payload, timeout)
        return _extract_json_block(last_raw)
    except Exception:
        pass

    # 2) No format + very explicit instruction
    reinforced = (
        "Return ONLY a JSON object. No explanations, no backticks, no extra keys.\n"
        + prompt
    )
    payload.pop("format", None)
    payload["prompt"] = reinforced
    try:
        last_raw = _generate(payload, timeout)
        return _extract_json_block(last_raw)
    except Exception:
        pass

    # 3) JSON fixer: ask model to convert its own last output to valid JSON
    fixer = (
        "Convert the following into VALID JSON only. Keep keys as in the schema. "
        "No commentary, no code fences.\n"
        "-----\n"
        f"{last_raw}\n"
        "-----"
    )
    payload["prompt"] = fixer
    try:
        last_raw = _generate(payload, timeout)
        return _extract_json_block(last_raw)
    except Exception as e:
        if debug:
            raise ValueError(f"No JSON found.\nRAW:\n{last_raw}") from e
        raise ValueError("No JSON found in model output") from e
