# src/extractor.py
from __future__ import annotations

import re
import argparse
import glob
from pathlib import Path
from typing import List, Optional

from docx import Document

from schema import Bio, ExperienceItem
from llm_client import ollama_json  # local client for Ollama (gemma3:4b default)


# ---------------------- Heuristic helpers ----------------------
EMAIL_RE = re.compile(r'[\w.\-+]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}')
LI_RE = re.compile(r'https?://(?:www\.)?linkedin\.com/[A-Za-z0-9\-_/]+', re.I)

SECTION_ALIASES = {
    "summary": {"summary", "profile", "about"},
    "expertise": {"expertise", "skills", "capabilities", "core competencies"},
    "experience": {"experience", "selected experience", "professional experience", "engagements"},
    "education": {"education", "academic"},
    "certifications": {"certifications", "licenses"},
    "affiliations": {"affiliations", "memberships"},
    "awards": {"awards", "recognition", "honors"},
}


def read_docx_lines(fp: Path) -> List[str]:
    doc = Document(str(fp))
    out = []
    for p in doc.paragraphs:
        t = (p.text or "").strip()
        if t:
            out.append(t)
    return out


def read_text(fp: Path) -> str:
    """Join all non-empty paragraphs for LLM input."""
    return "\n".join(read_docx_lines(fp))


def split_sections(lines: List[str]) -> dict:
    """Very light section splitter: heading lines are known aliases or short Title/ALLCAPS."""
    sections = {k: [] for k in SECTION_ALIASES.keys()}
    current_key: Optional[str] = None

    def classify_heading(s: str) -> Optional[str]:
        base = s.strip().lower()
        base = re.sub(r'[^a-z\s]', '', base)
        for key, names in SECTION_ALIASES.items():
            for nm in names:
                if base == nm or base.startswith(nm):
                    return key
        # Heuristic: <=3 words, Title Case or ALLCAPS
        if len(s.split()) <= 3 and (s.isupper() or re.match(r'^([A-Z][a-z]+)(\s[A-Z][a-z]+){0,2}$', s)):
            return None
        return None

    for ln in lines:
        key = classify_heading(ln)
        if key is not None:
            current_key = key
            continue
        if current_key:
            sections[current_key].append(ln)

    return sections


def detect_email(text: str) -> Optional[str]:
    m = EMAIL_RE.search(text)
    return m.group(0) if m else None


def detect_linkedin(text: str) -> Optional[str]:
    m = LI_RE.search(text)
    return m.group(0) if m else None


def detect_phone(lines: List[str]) -> Optional[str]:
    # naive scan; Bio schema will further normalize if phonenumbers is installed
    for ln in lines:
        if len(ln) < 8:
            continue
        if re.search(r'\d{3}.*\d{3}.*\d{4}', ln) or re.search(r'\+\d{7,}', ln):
            digits = re.sub(r'[^\d+]', '', ln)
            if len(re.sub(r'\D', '', digits)) >= 10:
                return digits
    return None


def looks_like_name(line: str) -> bool:
    tokens = [t for t in re.split(r'\s+', line) if t]
    if not (2 <= len(tokens) <= 5):
        return False
    upp = sum(1 for t in tokens if re.match(r'^[A-Z][a-z\-]+$', t))
    return upp >= max(2, len(tokens) - 1)


def clean_bullets(raw: List[str]) -> List[str]:
    out = []
    for r in raw:
        s = re.sub(r'^[\u2022\-\*\•\·]\s*', '', r).strip()
        if s:
            out.append(s)
    return out


def parse_bio_from_lines(lines: List[str]) -> Bio:
    joined = "\n".join(lines)
    email = detect_email(joined)
    linkedin = detect_linkedin(joined)
    phone = detect_phone(lines)

    # name + title guess from first 6 lines
    full_name = "Unknown"
    current_title: Optional[str] = None
    for i, ln in enumerate(lines[:6]):
        if looks_like_name(ln):
            full_name = ln.strip()
            if i + 1 < len(lines) and not looks_like_name(lines[i + 1]):
                current_title = lines[i + 1].strip()
            break

    sections = split_sections(lines)
    summary_paragraph = " ".join(sections.get("summary", [])).strip() or None

    expertise_bullets = clean_bullets(sections.get("expertise", []))
    education = clean_bullets(sections.get("education", []))
    certifications = clean_bullets(sections.get("certifications", []))
    affiliations = clean_bullets(sections.get("affiliations", []))
    awards = clean_bullets(sections.get("awards", []))

    # experience: naive pattern — triplets of (project, role, impact)
    exp_raw = sections.get("experience", [])
    selected_experience: List[ExperienceItem] = []
    i = 0
    while i < len(exp_raw):
        chunk = exp_raw[i : i + 3]
        if len(chunk) >= 2:
            client_or_project = chunk[0].strip()
            role = chunk[1].strip()
            impact = (chunk[2].strip() if len(chunk) > 2 else None)
            selected_experience.append(
                ExperienceItem(
                    client_or_project=client_or_project,
                    role=role,
                    impact=impact,
                )
            )
            i += 3
        else:
            break

    return Bio(
        full_name=full_name,
        current_title=current_title,
        email=email,
        phone=phone,
        linkedin_url=linkedin,
        summary_paragraph=summary_paragraph,
        expertise_bullets=expertise_bullets,
        selected_experience=selected_experience,
        education=education,
        certifications=certifications,
        affiliations=affiliations,
        awards=awards,
    )


# ---------------------- AI (Ollama) extraction ----------------------
DEFAULT_EXTRACT_PROMPT = """Return ONLY valid JSON matching the schema keys below. No explanations, no prose, no backticks.
If a field is unknown, set it to null or [].
Use third-person voice for text fields. Prefer corporate email if multiple.
Phone: E.164 if possible, else raw digits.

SCHEMA KEYS (types):
full_name: string
current_title: string
department_or_practice: string
location: string
email: string
phone: string
linkedin_url: string
summary_paragraph: string
expertise_bullets: [string]
selected_experience: [{client_or_project: string, role: string, impact: string}]
education: [string]
certifications: [string]
affiliations: [string]
awards: [string]

BIO TEXT:
<<<BIO>>>
""".strip()


def load_prompt(path: Path) -> str:
    if path.exists():
        return path.read_text(encoding="utf-8")
    return DEFAULT_EXTRACT_PROMPT


def ai_extract_bio(text: str, model: str = "gemma3:4b") -> Bio:
    prompt_tpl = load_prompt(Path("prompts") / "extract_schema.txt")
    prompt = prompt_tpl.replace("<<<BIO>>>", text)
    data = ollama_json(prompt, model=model, temperature=0.1)  # strict JSON helper
    return Bio(**data)  # Pydantic will normalize/validate


# ---------------------- CLI ----------------------
def main():
    import sys
    ap = argparse.ArgumentParser(description="Extract a structured Bio from a .docx")
    ap.add_argument("--file", help="Path to input .docx (if omitted, uses first in ./input)")
    ap.add_argument("--verbose", action="store_true", help="Print debug info")
    ap.add_argument("--use-ai", action="store_true", help="Use Ollama (LLM) for JSON extraction")
    ap.add_argument("--model", default="gemma3:4b", help="Ollama model (e.g., gemma3:4b, qwen2.5:7b-instruct)")
    args = ap.parse_args()

    # Resolve input
    if args.file:
        path = Path(args.file)
    else:
        candidates = sorted(glob.glob(str(Path("input") / "*.docx")))
        if not candidates:
            print("No .docx files found in ./input. Add one or pass --file.", file=sys.stderr)
            sys.exit(2)
        path = Path(candidates[0])

    if not path.exists():
        print(f"File not found: {path}", file=sys.stderr)
        sys.exit(2)

    if args.verbose:
        print(f"[extractor] reading: {path}")

    # Choose path
    if args.use_ai:
        text = read_text(path)
        if args.verbose:
            print(f"[extractor] characters sent to LLM: {len(text)}")
            print(f"[extractor] model: {args.model}")
        bio = ai_extract_bio(text, model=args.model)
    else:
        lines = read_docx_lines(path)
        if args.verbose:
            print(f"[extractor] paragraphs read: {len(lines)}")
        if not lines:
            print("No readable text found in the document. Is it a valid .docx?", file=sys.stderr)
            sys.exit(2)
        bio = parse_bio_from_lines(lines)

    # Print JSON-safe output
    from json import dumps
    print(dumps(bio.model_dump(mode="json"), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
