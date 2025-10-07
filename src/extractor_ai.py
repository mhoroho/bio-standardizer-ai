# src/extractor_ai.py
from __future__ import annotations
from pathlib import Path
import re, json, argparse
from typing import Any, Dict, List

from schema import Bio
from parser import read_text, looks_like_name   # reuse non-AI helpers
from llm_client_openai import openai_json       # OpenAI client wrapper

def load_prompt(path: Path, default: str) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else default

DEFAULT_EXTRACT_PROMPT = """Return ONLY a single JSON object with EXACTLY these keys (include them even if null or []):
full_name, current_title, department_or_practice, location, email, phone, linkedin_url, summary_paragraph, expertise_bullets, selected_experience, education, certifications, affiliations, awards.
No extra keys. No prose. No code fences.
Rules: If unknown use null or []; phone E.164 if possible; text in third-person.

BIO TEXT:
<<<BIO>>>
""".strip()

def ai_extract_bio_from_text(text: str, model: str = "gpt-4o-mini", debug: bool = False) -> Bio:
    prompt_tpl = load_prompt(Path("prompts") / "extract_schema.txt", DEFAULT_EXTRACT_PROMPT)
    prompt = prompt_tpl.replace("<<<BIO>>>", text)
    data: Dict[str, Any] = openai_json(prompt, model=model)

    payload: Dict[str, Any] = {
        "full_name": None,
        "current_title": None,
        "department_or_practice": None,
        "location": None,
        "email": None,
        "phone": None,
        "linkedin_url": None,
        "summary_paragraph": None,
        "expertise_bullets": [],
        "selected_experience": [],
        "education": [],
        "certifications": [],
        "affiliations": [],
        "awards": [],
    }
    if isinstance(data, dict):
        for k in payload.keys():
            if k in data:
                payload[k] = data[k]

    # repair full_name if missing
    if not payload.get("full_name") or not str(payload["full_name"]).strip():
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        for ln in lines[:8]:
            if looks_like_name(ln):
                payload["full_name"] = ln.strip()
                break
        if not payload["full_name"]:
            payload["full_name"] = "Unknown"

    # coerce list fields
    for list_key in ["expertise_bullets","education","certifications","affiliations","awards"]:
        v = payload.get(list_key)
        if isinstance(v, str):
            items = [s.strip(" •-\t") for s in re.split(r"[;\n]|,\s+(?=[A-Za-z])", v) if s.strip()]
            payload[list_key] = items
        elif v is None:
            payload[list_key] = []

    # normalize selected_experience
    se = payload.get("selected_experience")
    if isinstance(se, dict):
        payload["selected_experience"] = [se]
    elif isinstance(se, list):
        norm: List[Dict[str, Any]] = []
        for item in se:
            if isinstance(item, dict):
                norm.append({
                    "client_or_project": (item.get("client_or_project") or "").strip(),
                    "role": (item.get("role") or None),
                    "impact": (item.get("impact") or None),
                })
            elif isinstance(item, str):
                norm.append({"client_or_project": item.strip(), "role": None, "impact": None})
        payload["selected_experience"] = norm
    else:
        payload["selected_experience"] = []

    bio = Bio(**payload)
    if debug:
        filled = [k for k,v in payload.items() if v not in (None, [], "")]
        print(f"[extractor_ai] filled keys: {sorted(filled)}")
    return bio

def main():
    ap = argparse.ArgumentParser(description="AI extractor (OpenAI)")
    ap.add_argument("--file", required=True, help="Path to input .docx")
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    text = read_text(Path(args.file))
    bio = ai_extract_bio_from_text(text, model=args.model, debug=args.verbose)
    print(json.dumps(bio.model_dump(mode="json"), indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
