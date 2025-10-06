from __future__ import annotations
from typing import List, Optional
from pydantic import BaseModel, EmailStr, AnyUrl, Field, field_validator, model_validator
import re

try:
    import phonenumbers  # type: ignore
except Exception:  # phonenumbers is optional at runtime
    phonenumbers = None


# -------------------- helpers --------------------
ACRONYMS = {"AI", "ML", "NLP", "LLM", "R&D", "SQL", "BI", "M&A", "ESG", "CFA", "CPA", "AWS", "GCP", "GPU", "API", "CEO", "CIO", "CTO"}
WORD_LIMIT_BULLET = 16  # keep bullets tight and scannable

def smart_title(s: str) -> str:
    if not s:
        return s
    words = re.split(r"(\s+|-|/)", s.strip())  # keep separators
    out = []
    for w in words:
        if re.match(r"\s+|-|/", w):  # keep separators as-is
            out.append(w)
            continue
        if w.upper() in ACRONYMS:
            out.append(w.upper())
        elif w.isupper() and len(w) <= 3:  # short all-caps like "SVP"
            out.append(w.upper())
        else:
            out.append(w[:1].upper() + w[1:])
    return "".join(out)

def word_count(s: str) -> int:
    return len([t for t in re.split(r"\s+", s.strip()) if t])


# -------------------- data models --------------------
class ExperienceItem(BaseModel):
    client_or_project: str = Field(..., min_length=2)
    role: Optional[str] = Field(default=None)
    impact: Optional[str] = Field(default=None)

    @field_validator("client_or_project", "role", "impact", mode="before")
    @classmethod
    def _strip(cls, v):
        return v.strip() if isinstance(v, str) else v

    @field_validator("impact")
    @classmethod
    def _short_impact(cls, v):
        if not v:
            return v
        # soft limit; don’t fail, but trim excess spaces
        return re.sub(r"\s+", " ", v).strip()


class Bio(BaseModel):
    # identity / headline
    full_name: str = Field(..., min_length=2)
    current_title: Optional[str] = None
    department_or_practice: Optional[str] = None
    location: Optional[str] = None

    # contacts
    email: Optional[EmailStr] = None
    phone: Optional[str] = None
    linkedin_url: Optional[AnyUrl] = None  # we’ll normalize host in validator

    # narrative + sections
    summary_paragraph: Optional[str] = None
    expertise_bullets: List[str] = Field(default_factory=list)
    selected_experience: List[ExperienceItem] = Field(default_factory=list)
    education: List[str] = Field(default_factory=list)
    certifications: List[str] = Field(default_factory=list)
    affiliations: List[str] = Field(default_factory=list)
    awards: List[str] = Field(default_factory=list)

    # -------------------- field normalizers --------------------
    @field_validator("full_name", mode="before")
    @classmethod
    def _name_strip(cls, v: str) -> str:
        if not isinstance(v, str):
            raise TypeError("full_name must be a string")
        v = re.sub(r"\s+", " ", v).strip()
        return " ".join([p.capitalize() if p.upper() not in ACRONYMS else p.upper() for p in v.split()])

    @field_validator("current_title", mode="before")
    @classmethod
    def _title_case(cls, v: Optional[str]) -> Optional[str]:
        if not isinstance(v, str) or not v.strip():
            return v
        return smart_title(re.sub(r"\s+", " ", v).strip())

    @field_validator("department_or_practice", "location", "summary_paragraph", mode="before")
    @classmethod
    def _compact_whitespace(cls, v: Optional[str]) -> Optional[str]:
        if isinstance(v, str):
            v = re.sub(r"\s+", " ", v).strip()
            return v or None
        return v

    @field_validator("linkedin_url")
    @classmethod
    def _linkedin_only(cls, v: Optional[AnyUrl]) -> Optional[AnyUrl]:
        if not v:
            return v
        # ensure host contains linkedin.com
        if "linkedin.com" not in v.host:
            raise ValueError("linkedin_url must be a linkedin.com URL")
        return v

    @field_validator("phone")
    @classmethod
    def _normalize_phone(cls, v: Optional[str]) -> Optional[str]:
        if not v:
            return v
        s = re.sub(r"[^\d+]", "", v)
        if not s:
            return None
        if phonenumbers is None:
            # fallback: naive formats like +15551234567 or (555) 123-4567
            return s
        try:
            num = phonenumbers.parse(s, "US")  # adjust default region if needed
            if phonenumbers.is_valid_number(num):
                # E.164 or national – choose one house format; here national:
                return phonenumbers.format_number(num, phonenumbers.PhoneNumberFormat.NATIONAL)
        except Exception:
            pass
        return s  # don’t fail; keep raw digits

    @field_validator(
        "expertise_bullets",
        "education",
        "certifications",
        "affiliations",
        "awards",
        mode="before",
    )
    @classmethod
    def _list_clean(cls, v):
        if not v:
            return []
        if isinstance(v, str):
            v = [v]
        cleaned = []
        for item in v:
            if not isinstance(item, str):
                continue
            item = re.sub(r"\s+", " ", item).strip(" •-\t")
            if item:
                cleaned.append(item)
        # dedupe preserving order
        seen = set()
        out = []
        for it in cleaned:
            key = it.lower()
            if key not in seen:
                seen.add(key)
                out.append(it)
        return out

    # -------------------- model-level rules --------------------
    @model_validator(mode="after")
    def _enforce_bullet_limits(self):
        def trim_bullets(lst: List[str]) -> List[str]:
            out = []
            for b in lst:
                txt = re.sub(r"\s+", " ", b).strip()
                if word_count(txt) > WORD_LIMIT_BULLET:
                    # soft rule: split on ';' or '—' or add ellipsis if too long
                    parts = re.split(r"[;–—]", txt, maxsplit=1)
                    txt = parts[0]
                out.append(txt)
            return out

        self.expertise_bullets = trim_bullets(self.expertise_bullets)
        self.education = trim_bullets(self.education)
        self.certifications = trim_bullets(self.certifications)
        self.affiliations = trim_bullets(self.affiliations)
        self.awards = trim_bullets(self.awards)
        return self

    # -------------------- convenience APIs --------------------
    def empty_fields(self) -> List[str]:
        """Quick audit: which important fields are still empty?"""
        missing = []
        for f in ["current_title", "summary_paragraph", "expertise_bullets", "education"]:
            val = getattr(self, f)
            if val in (None, "", []) or (isinstance(val, list) and len(val) == 0):
                missing.append(f)
        return missing

    def manifest_row(self) -> dict:
        """Flatten key info for a CSV/log manifest."""
        return {
            "full_name": self.full_name,
            "title": self.current_title or "",
            "email": self.email or "",
            "phone": self.phone or "",
            "linkedin": str(self.linkedin_url) if self.linkedin_url else "",
            "expertise_count": len(self.expertise_bullets),
            "experience_count": len(self.selected_experience),
        }


# -------------------- quick self-test --------------------
if __name__ == "__main__":
    demo = Bio(
        full_name="jane DOE",
        current_title="senior managing director, ai/ml",
        department_or_practice="Data & Analytics",
        location="Boston, MA",
        email="jane.doe@example.com",
        phone="+1 (617) 555-0100",
        linkedin_url="https://www.linkedin.com/in/janedoe",
        summary_paragraph="Leads AI and analytics programs across financial services.",
        expertise_bullets=[
            "Responsible AI & model risk",
            "GenAI solutions; prompt engineering; governance frameworks for LLMs",
            "Pricing analytics and forecasting"
        ],
        selected_experience=[
            ExperienceItem(
                client_or_project="Top-10 US Bank",
                role="Program Lead, Model Governance",
                impact="Designed AI governance and monitoring for >50 models."
            )
        ],
        education=["BS, Computer Science — Northeastern University"],
        certifications=["CFA Level I", "AWS Certified Solutions Architect"],
        affiliations=["IEEE", "ACM"],
        awards=["Firm Innovation Award (2023)"],
    )
    from json import dumps
    print(dumps(demo.model_dump(mode="json"), indent=2))
    print("Missing fields:", demo.empty_fields())
    print("Manifest:", demo.manifest_row())
