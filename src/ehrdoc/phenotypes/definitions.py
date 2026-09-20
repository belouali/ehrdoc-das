from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Sequence

@dataclass(frozen=True)
class TextRule:
    include: Sequence[str] = field(default_factory=list)
    exclude: Sequence[str] = field(default_factory=list)
    icd_regex: Optional[str] = None

@dataclass(frozen=True)
class PhenotypeRule:
    name: str
    diagnosis: TextRule
    past_history: TextRule
    medication: TextRule
