# src/parser.py
from __future__ import annotations
import re
from pathlib import Path
from typing import List, Optional, Dict
from docx import Document
from schema import Bio, ExperienceItem

EMAIL_RE = re.compile(r'[\w.\-+]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}')
LI_RE = re.compile(r'https?://(?:www\.)?linkedin\.com/[A-Za-z0-9\-_/]+', re.I)

SECTION_ALIASES: Dict[str, set[str]] = {
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
    out: List[str] = []
    for p in doc.paragraphs:
        t = (p.text or "").strip()
        if t:
            out.append(t)
    return out

def read_text(fp: Path, max_chars: int = 4000) -> str:
    """Join paragraphs, optionally trim for speed."""
    txt = "\n".join(read_docx_lines(fp))
    return txt[:max_chars]

def looks_like_name(line: str) -> bool:
    tokens = [t for t in re.split(r'\s+', line) if t]
    if not (2 <= len(tokens) <= 5):
        return False
    upp = sum(1 for t in tokens if re.match(r'^[A-Z][a-z\-]+$', t))
    return upp >= max(2, len(tokens)-1)

def detect_email(text: str) -> Optional[str]:
    m = EMAIL_RE.search(text)
    return m.group(0) if m else None

def detect_linkedin(text: str) -> Optional[str]:
    m = LI_RE.search(text)
    return m.group(0) if m else None

def detect_phone(lines: List[str]) -> Optional[str]:
    for ln in lines:
        if len(ln) < 8:
            continue
        if re.search(r'\d{3}.*\d{3}.*\d{4}', ln) or re.search(r'\+\d{7,}', ln):
            digits = re.sub(r'[^\d+]', '', ln)
            if len(re.sub(r'\D', '', digits)) >= 10:
                return digits
    return None

def clean_bullets(raw: List[str]) -> List[str]:
    out = []
    for r in raw:
        s = re.sub(r'^[\u2022\-\*\•\·]\s*', '', r).strip()
        if s:
            out.append(s)
    return out

def split_sections(lines: List[str]) -> Dict[str, List[str]]:
    sections = {k: [] for k in SECTION_ALIASES.keys()}
    current_key: Optional[str] = None

    def classify_heading(s: str) -> Optional[str]:
        base = s.strip().lower()
        base = re.sub(r'[^a-z\s]', '', base)
        for key, names in SECTION_ALIASES.items():
            for nm in names:
                if base == nm or base.startswith(nm):
                    return key
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

def parse_bio_from_lines(lines: List[str]) -> Bio:
    joined = "\n".join(lines)
    email = detect_email(joined)
    linkedin = detect_linkedin(joined)
    phone = detect_phone(lines)

    full_name = "Unknown"
    current_title: Optional[str] = None
    for i, ln in enumerate(lines[:6]):
        if looks_like_name(ln):
            full_name = ln.strip()
            if i + 1 < len(lines) and not looks_like_name(lines[i+1]):
                current_title = lines[i+1].strip()
            break

    sections = split_sections(lines)
    summary_paragraph = " ".join(sections.get("summary", [])).strip() or None

    expertise_bullets = clean_bullets(sections.get("expertise", []))
    education = clean_bullets(sections.get("education", []))
    certifications = clean_bullets(sections.get("certifications", []))
    affiliations = clean_bullets(sections.get("affiliations", []))
    awards = clean_bullets(sections.get("awards", []))

    exp_raw = sections.get("experience", [])
    selected_experience: List[ExperienceItem] = []
    i = 0
    while i < len(exp_raw):
        chunk = exp_raw[i:i+3]
        if len(chunk) >= 2:
            client_or_project = chunk[0].strip()
            role = chunk[1].strip()
            impact = (chunk[2].strip() if len(chunk) > 2 else None)
            selected_experience.append(ExperienceItem(
                client_or_project=client_or_project,
                role=role,
                impact=impact
            ))
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

def parse_bio_from_file(fp: Path) -> Bio:
    return parse_bio_from_lines(read_docx_lines(fp))

if __name__ == "__main__":
    import argparse, json
    ap = argparse.ArgumentParser(description="Heuristic parser (non-AI)")
    ap.add_argument("--file", required=True)
    args = ap.parse_args()
    bio = parse_bio_from_file(Path(args.file))
    print(json.dumps(bio.model_dump(mode="json"), indent=2, ensure_ascii=False))
