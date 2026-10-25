"""Check that every decimal number in paper/FINAL_RESEARCH_PAPER.md appears in an audited source.

Sources: the step reports, pre-registrations and generated tables in experiments/robust/ (Phases 2-5, Steps 6-8), the
literature review (for numbers quoted from prior work), the error-map record and the preliminary study's statistics
(Appendix A; its earlier manuscript is a source too because its tables are generated from the same files). JSON sources
count with every float rounded to 2, 3 and 4 decimals. A number passes if the same digits (sign ignored, "−" read as "-")
occur in a source as a whole token. This catches typos and unsourced numbers, not numbers used in the wrong place; that is
check_final_paper_tables.py's job. The header and References are skipped. Exits 1 if any number is unsourced.

Usage:
    .venv/Scripts/python.exe paper/check_final_paper_numbers.py
"""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
R = ROOT / "experiments" / "robust"
TEXT_SOURCES = [R / n for n in (
    "REPORT.md", "PHASE2_PREREGISTRATION.md", "PHASE3_PREREGISTRATION.md", "PHASE3_REPORT.md", "PHASE4A_PREREGISTRATION.md",
    "PHASE4A_REPORT.md", "PHASE4ABIS_PREREGISTRATION.md", "PHASE4ABIS_REPORT.md", "PHASE4B_PREREGISTRATION.md",
    "PHASE4B_REPORT.md", "PHASE5S_PREREGISTRATION.md", "PHASE5S_REPORT.md", "PHASE5_PREREGISTRATION.md", "PHASE5_REPORT.md",
    "PHASE6_PREREGISTRATION.md", "PHASE6_REPORT.md", "PHASE7_PREREGISTRATION.md", "PHASE7_REPORT.md",
    "PHASE8_PREREGISTRATION.md", "PHASE8_REPORT.md")] + [
    ROOT / "docs" / "final" / "LITERATURE_REVIEW.md", ROOT / "paper" / "RESEARCH_PAPER_FINAL.md"]
JSON_SOURCES = [R / "error_maps" / "selection.json", ROOT / "experiments" / "paper_statistics.json"]
NUM = re.compile(r"(?<![\w.§])[+\-−]?(\d+\.\d+(?:e[+\-]\d+)?)(?![\w.])")


def numbers(text: str) -> set[str]:
    return {m.group(1) for m in NUM.finditer(text.replace("−", "-"))}


def json_numbers(obj) -> set[str]:
    if isinstance(obj, dict):
        return set().union(*(json_numbers(v) for v in obj.values())) if obj else set()
    if isinstance(obj, list):
        return set().union(*(json_numbers(v) for v in obj)) if obj else set()
    if isinstance(obj, float):
        return {f"{abs(obj):.{d}f}" for d in (2, 3, 4)}
    return set()


def main() -> None:
    paper = (ROOT / "paper" / "FINAL_RESEARCH_PAPER.md").read_text(encoding="utf-8")
    body = paper.split("## Abstract", 1)[1]
    body = body.split("## References", 1)[0] + body.split("## Appendix A", 1)[1]  # skip the reference list
    body = "\n".join(ln for ln in body.splitlines() if not ln.startswith("#"))  # skip section numbers
    known = set()
    for s in TEXT_SOURCES:
        known |= numbers(s.read_text(encoding="utf-8"))
    for s in JSON_SOURCES:
        known |= json_numbers(json.loads(s.read_text(encoding="utf-8")))
    claimed = numbers(body)
    missing = sorted(claimed - known)
    print(f"{len(claimed)} distinct numbers in the paper body, {len(claimed) - len(missing)} found in sources, {len(missing)} not found")
    for x in missing:
        lines = [ln.strip()[:110] for ln in body.splitlines() if re.search(rf"(?<![\w.])[+\-−]?{re.escape(x)}(?![\w.])", ln)]
        print(f"  NOT FOUND {x}: {lines[0] if lines else ''}")
    sys.exit(1 if missing else 0)


if __name__ == "__main__":
    main()
