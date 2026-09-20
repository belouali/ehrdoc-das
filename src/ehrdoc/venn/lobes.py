from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple

# Lobe mapping in the manuscript order:
# 1: Dx∩Hx∩Med
# 2: Dx∩Hx
# 3: Dx∩Med
# 4: Dx only
# 5: Hx∩Med
# 6: Hx only
# 7: Med only

LOBE_LABELS = [
    "Dx∩Hx∩Med",
    "Dx∩Hx",
    "Dx∩Med",
    "Dx only",
    "Hx∩Med",
    "Hx only",
    "Med only",
]

def encode_lobe(dx: bool, hx: bool, med: bool) -> int:
    if dx and hx and med: return 0
    if dx and hx and (not med): return 1
    if dx and (not hx) and med: return 2
    if dx and (not hx) and (not med): return 3
    if (not dx) and hx and med: return 4
    if (not dx) and hx and (not med): return 5
    if (not dx) and (not hx) and med: return 6
    # no evidence in any source -> not part of the Venn
    return -1
