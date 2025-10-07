# src/style_planner_ai.py
from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any
import json
from docx import Document
from llm_client_openai import openai_json

# ---- collect raw facts from exemplar (NO heuristics beyond reading) ----
def read_paragraph_facts(exemplar_path: Path, max_paras: int = 400):
    doc = Document(str(exemplar_path))
    facts = []
    for p in doc.paragraphs[:max_paras]:
        txt = (p.text or "").strip()
        style = p.style.name if p.style is not None else ""
        if txt:
            facts.append({"style_name": style, "text": txt})
    return facts

PROMPT = """You analyze an exemplar Word document described as a list of paragraphs with their Word style names.

Return ONLY JSON named render_plan that fully defines how to render bios to match the exemplar, including a HEADER block and SECTION blocks.

Rules:
- Use ONLY style names that appear in the exemplar (e.g., "Heading 1", "Heading 2", "Normal", "List Bullet", "List Paragraph", "Subtitle", etc.).
- Infer section order and choose styles per section.
- Include a HEADER mapping for identity/contacts so we can render the top block consistently.
- Keep text tone third-person; propose simple word limits.

JSON schema to return (keys and types must match; include all keys even if a field is unused):

{
  "header": {
    "name_style": "Heading 1",
    "title_style": "Subtitle",
    "department_style": "Normal",
    "location_style": "Normal",
    "contact_style": "Normal",   // used for "email | phone | linkedin"
    "line_order": ["name","title","department_or_practice","location","contact"]  // order of header lines
  },
  "sections": [
    {
      "key": "summary",                   // one of: summary, expertise, experience, education, certifications, affiliations, awards; or snake_case fallback
      "label": "Summary",
      "heading_style": "Heading 2",
      "body_style": "Normal",
      "bullet_style": "List Bullet",      // may be null for narrative sections
      "bullet_limit": 0                   // 0 = narrative paragraph(s)
    }
    // more sections in desired order
  ],
  "tone": {
    "person": "third",
    "bullets_max_words": 16,
    "summary_max_words": 120
  }
}

Notes:
- If a section is list-like (expertise, education, certifications, affiliations, awards), set bullet_limit to 4–8.
- If a section uses bullets in exemplar but style name isn't obvious, choose the closest list style seen.
- For HEADER, choose the most appropriate styles found in the exemplar paragraphs. If a style is unclear, pick a reasonable default from styles you see.

PARAGRAPHS (array of {style_name,text}):
<<<PARA_JSON>>>
"""


def build_style_plan_with_llm(exemplar_path: Path, model: str = "gpt-4o-mini") -> Dict[str, Any]:
    facts = read_paragraph_facts(exemplar_path)
    para_json = json.dumps(facts, ensure_ascii=False)
    prompt = PROMPT.replace("<<<PARA_JSON>>>", para_json)
    plan = openai_json(prompt, model=model)
    # minimal validation
    if not isinstance(plan, dict) or "sections" not in plan:
        raise ValueError("LLM did not return a valid render_plan with 'sections'.")
    return plan

def save_plan(plan: Dict[str, Any], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="LLM style planner from exemplar")
    ap.add_argument("--exemplar", required=True)
    ap.add_argument("--out", default="logs/render_plan.json")
    ap.add_argument("--model", default="gpt-4o-mini")
    args = ap.parse_args()

    plan = build_style_plan_with_llm(Path(args.exemplar), model=args.model)
    save_plan(plan, Path(args.out))
    print(f"✓ render plan written to {args.out}")
