"""Check that every decimal number in paper/ROBUSTNESS_PAPER_DRAFT.md appears in an audited study source.

Sources: the step reports, pre-registrations and generated tables in experiments/robust/. A number passes if the same
digits (sign ignored, "−" read as "-") occur in any source as a whole token. This catches typos and numbers with no
source; it does not check that a number is used in the right place, so the text still needs reading against the reports.
The References section and the header are skipped. Exits 1 if any number is unsourced.

Usage:
    .venv/Scripts/python.exe paper/check_robustness_paper_numbers.py
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
R = ROOT / "experiments" / "robust"
SOURCES = [R / n for n in ("REPORT.md", "PHASE2_PREREGISTRATION.md", "PHASE3_PREREGISTRATION.md", "PHASE3_REPORT.md",
                           "PHASE4A_PREREGISTRATION.md", "PHASE4A_REPORT.md", "PHASE4ABIS_PREREGISTRATION.md",
                           "PHASE4ABIS_REPORT.md", "PHASE4B_PREREGISTRATION.md", "PHASE4B_REPORT.md")]
NUM = re.compile(r"(?<![\w.§])[+\-−]?(\d+\.\d+(?:e[+\-]\d+)?)(?![\w.])")


def numbers(text: str) -> set[str]:
    return {m.group(1) for m in NUM.finditer(text.replace("−", "-"))}


def main() -> None:
    paper = (ROOT / "paper" / "ROBUSTNESS_PAPER_DRAFT.md").read_text(encoding="utf-8")
    body = paper.split("## Abstract", 1)[1].split("## References", 1)[0]
    body = "\n".join(ln for ln in body.splitlines() if not ln.startswith("#"))  # skip section numbers
    known = set()
    for s in SOURCES:
        known |= numbers(s.read_text(encoding="utf-8"))
    claimed = numbers(body)
    missing = sorted(claimed - known)
    print(f"{len(claimed)} distinct numbers in the paper body, {len(claimed) - len(missing)} found in sources, {len(missing)} not found")
    for x in missing:
        lines = [ln.strip()[:110] for ln in body.splitlines() if re.search(rf"(?<![\w.])[+\-−]?{re.escape(x)}(?![\w.])", ln)]
        print(f"  NOT FOUND {x}: {lines[0] if lines else ''}")
    sys.exit(1 if missing else 0)


if __name__ == "__main__":
    main()
