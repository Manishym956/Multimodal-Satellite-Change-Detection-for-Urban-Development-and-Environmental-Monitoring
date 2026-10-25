"""
Build the research paper as a formatted Word (.docx) document.
Includes all tables, all 11 figures embedded, proper academic formatting.

Run from project root:
    .venv/Scripts/python.exe build_paper_docx.py
"""

import re
import sys
from pathlib import Path

try:
    from docx import Document
    from docx.shared import Pt, Inches, RGBColor, Cm
    from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
    from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
    from docx.oxml.ns import qn
    from docx.oxml import OxmlElement
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "python-docx", "-q"])
    from docx import Document
    from docx.shared import Pt, Inches, RGBColor, Cm
    from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
    from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
    from docx.oxml.ns import qn
    from docx.oxml import OxmlElement

ROOT = Path(__file__).resolve().parent
PAPER_MD = ROOT / "paper" / "RESEARCH_PAPER_FINAL.md"
FIGURES_DIR = ROOT / "figures" / "paper"
OUT = ROOT / "paper" / "RESEARCH_PAPER_FINAL.docx"

# ─────────────────────────────────────────────────────────────
# Document setup
# ─────────────────────────────────────────────────────────────

doc = Document()

# Page margins: 1 inch all sides
for section in doc.sections:
    section.top_margin    = Inches(1.0)
    section.bottom_margin = Inches(1.0)
    section.left_margin   = Inches(1.25)
    section.right_margin  = Inches(1.25)
    section.page_width    = Inches(8.5)
    section.page_height   = Inches(11)

# Default paragraph font
style = doc.styles['Normal']
style.font.name = 'Times New Roman'
style.font.size = Pt(12)
style.paragraph_format.space_after = Pt(6)
style.paragraph_format.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE

def set_heading_style(heading, size, bold=True, color=None, center=False):
    for run in heading.runs:
        run.font.name = 'Times New Roman'
        run.font.size = Pt(size)
        run.font.bold = bold
        if color:
            run.font.color.rgb = RGBColor(*color)
    heading.paragraph_format.space_before = Pt(12)
    heading.paragraph_format.space_after  = Pt(4)
    if center:
        heading.alignment = WD_ALIGN_PARAGRAPH.CENTER


# ─────────────────────────────────────────────────────────────
# Helper: apply inline bold/italic to a paragraph run
# ─────────────────────────────────────────────────────────────

_SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
_LATEX_SYMBOLS = [
    (r"\mathbb{R}", "ℝ"), (r"\mathbb{1}", "𝟙"), (r"\in", "∈"), (r"\times", "×"), (r"\geq", "≥"),
    (r"\leq", "≤"), (r"\sigma", "σ"), (r"\tau", "τ"), (r"\theta", "θ"), (r"\cdot", "·"),
    (r"\{", "{"), (r"\}", "}"), (r"\,", " "), ("~", " "),
]


def latex_to_text(expr: str) -> str:
    """Approximate a small inline-LaTeX subset as Unicode, e.g. \\mathbf{S}_1 -> S₁, \\hat{\\mathbf{Y}} -> Ŷ."""
    expr = re.sub(r"\\hat\{\\mathbf\{(\w)\}\}", lambda m: m.group(1) + "\u0302", expr)
    expr = re.sub(r"\\(?:mathbf|mathrm|text)\{([^{}]*)\}", r"\1", expr)
    for latex, uni in _LATEX_SYMBOLS:
        expr = expr.replace(latex, uni)
    expr = re.sub(r"_\{?(\d+)\}?", lambda m: m.group(1).translate(_SUB), expr)
    expr = re.sub(r"_\{?([A-Za-zθ])\}?", r"_\1", expr)
    expr = re.sub(r"\^\{([^{}]*)\}", r"^(\1)", expr)
    return re.sub(r"\s+", " ", expr).strip()


assert latex_to_text(r"\mathbf{S}_1, \mathbf{S}_2 \in \mathbb{R}^{H \times W \times 1}") == "S₁, S₂ ∈ ℝ^(H × W × 1)"
assert latex_to_text(r"\hat{\mathbf{Y}} = \mathbb{1}[\sigma(\mathbf{Z}) \geq \tau]") == "Ŷ = 𝟙[σ(Z) ≥ τ]"


def apply_inline(para, text: str, base_size=12, base_bold=False, base_italic=False):
    """Parse **bold**, *italic*, `code`, and plain text segments and add runs."""
    # Render inline $...$ math as readable Unicode text (python-docx has no equation support)
    text = re.sub(r'\$([^$]+)\$', lambda m: latex_to_text(m.group(1)), text)

    # Pattern: **bold**, *italic*, `code`, or plain
    pattern = re.compile(r'(\*\*.*?\*\*|\*[^*]+\*|`[^`]+`)')
    parts = pattern.split(text)
    for part in parts:
        if not part:
            continue
        run = para.add_run()
        run.font.name = 'Times New Roman'
        run.font.size = Pt(base_size)
        if part.startswith('**') and part.endswith('**'):
            run.text = part[2:-2]
            run.font.bold = True
            run.font.italic = base_italic
        elif part.startswith('*') and part.endswith('*') and not part.startswith('**'):
            run.text = part[1:-1]
            run.font.italic = True
            run.font.bold = base_bold
        elif part.startswith('`') and part.endswith('`'):
            run.text = part[1:-1]
            run.font.name = 'Courier New'
            run.font.size = Pt(10)
            run.font.bold = base_bold
        else:
            run.text = part
            run.font.bold = base_bold
            run.font.italic = base_italic


def add_body(text: str, indent=False, italic=False, bold=False, size=12, center=False):
    para = doc.add_paragraph()
    para.paragraph_format.space_after = Pt(6)
    para.paragraph_format.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE
    if indent:
        para.paragraph_format.left_indent = Inches(0.4)
    if center:
        para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    apply_inline(para, text, base_size=size, base_bold=bold, base_italic=italic)
    return para


def add_bullet(text: str, level=0):
    para = doc.add_paragraph(style='List Bullet')
    para.paragraph_format.left_indent = Inches(0.3 + level * 0.2)
    para.paragraph_format.space_after = Pt(3)
    apply_inline(para, text, base_size=12)
    return para


def add_numbered(text: str, number: int):
    para = doc.add_paragraph()
    para.paragraph_format.left_indent = Inches(0.4)
    para.paragraph_format.first_line_indent = Inches(-0.3)
    para.paragraph_format.space_after = Pt(4)
    run_num = para.add_run(f"{number}. ")
    run_num.font.name = 'Times New Roman'
    run_num.font.size = Pt(12)
    run_num.font.bold = True
    apply_inline(para, text, base_size=12)
    return para


def add_blockquote(text: str):
    para = doc.add_paragraph()
    para.paragraph_format.left_indent  = Inches(0.5)
    para.paragraph_format.right_indent = Inches(0.3)
    para.paragraph_format.space_before = Pt(4)
    para.paragraph_format.space_after  = Pt(4)
    # Light gray border on left (simulate blockquote)
    pPr = para._p.get_or_add_pPr()
    pBdr = OxmlElement('w:pBdr')
    left = OxmlElement('w:left')
    left.set(qn('w:val'), 'single')
    left.set(qn('w:sz'), '6')
    left.set(qn('w:space'), '4')
    left.set(qn('w:color'), '888888')
    pBdr.append(left)
    pPr.append(pBdr)
    apply_inline(para, text, base_size=11, base_italic=True)
    return para


def add_code_block(lines: list[str]):
    para = doc.add_paragraph()
    para.paragraph_format.left_indent = Inches(0.4)
    para.paragraph_format.space_before = Pt(4)
    para.paragraph_format.space_after  = Pt(4)
    run = para.add_run('\n'.join(lines))
    run.font.name = 'Courier New'
    run.font.size = Pt(9)
    # Light gray background shading
    pPr = para._p.get_or_add_pPr()
    shd = OxmlElement('w:shd')
    shd.set(qn('w:val'), 'clear')
    shd.set(qn('w:color'), 'auto')
    shd.set(qn('w:fill'), 'F2F2F2')
    pPr.append(shd)


def add_figure(fig_path: Path, caption: str, width_inches=5.5):
    if not fig_path.exists():
        add_body(f"[Figure not found: {fig_path.name}]", italic=True, center=True)
        return
    para = doc.add_paragraph()
    para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    para.paragraph_format.space_before = Pt(8)
    run = para.add_run()
    run.add_picture(str(fig_path), width=Inches(width_inches))
    # Caption
    cap = doc.add_paragraph()
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap.paragraph_format.space_after = Pt(12)
    apply_inline(cap, caption, base_size=10, base_italic=True)


def add_table_from_rows(header: list[str], rows: list[list[str]], bold_last=False):
    n_cols = len(header)
    table = doc.add_table(rows=1, cols=n_cols)
    table.style = 'Table Grid'
    table.alignment = WD_TABLE_ALIGNMENT.CENTER

    # Header row
    hdr = table.rows[0]
    for i, cell_text in enumerate(header):
        cell = hdr.cells[i]
        cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
        para = cell.paragraphs[0]
        para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = para.add_run(cell_text)
        run.font.name = 'Times New Roman'
        run.font.size = Pt(10)
        run.font.bold = True
        # Gray background
        tc_pr = cell._tc.get_or_add_tcPr()
        shd = OxmlElement('w:shd')
        shd.set(qn('w:val'), 'clear')
        shd.set(qn('w:color'), 'auto')
        shd.set(qn('w:fill'), 'D9D9D9')
        tc_pr.append(shd)

    # Data rows
    for r_idx, row_data in enumerate(rows):
        row = table.add_row()
        is_bold_row = bold_last and (r_idx == len(rows) - 1)
        is_mean_row = any('mean' in str(c).lower() or '±' in str(c) for c in row_data)
        for i, cell_text in enumerate(row_data):
            cell = row.cells[i]
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            para = cell.paragraphs[0]
            para.alignment = WD_ALIGN_PARAGRAPH.CENTER
            # Strip markdown bold markers
            clean = str(cell_text).strip().lstrip('*').rstrip('*')
            run = para.add_run(clean)
            run.font.name = 'Times New Roman'
            run.font.size = Pt(9)
            run.font.bold = is_bold_row or is_mean_row or str(cell_text).startswith('**')
            if is_mean_row:
                tc_pr = cell._tc.get_or_add_tcPr()
                shd = OxmlElement('w:shd')
                shd.set(qn('w:val'), 'clear')
                shd.set(qn('w:color'), 'auto')
                shd.set(qn('w:fill'), 'FFF3CD')
                tc_pr.append(shd)

    doc.add_paragraph()  # spacing after table


def add_separator():
    para = doc.add_paragraph()
    para.paragraph_format.space_before = Pt(2)
    para.paragraph_format.space_after  = Pt(2)
    pPr = para._p.get_or_add_pPr()
    pBdr = OxmlElement('w:pBdr')
    bottom = OxmlElement('w:bottom')
    bottom.set(qn('w:val'), 'single')
    bottom.set(qn('w:sz'), '4')
    bottom.set(qn('w:space'), '1')
    bottom.set(qn('w:color'), 'AAAAAA')
    pBdr.append(bottom)
    pPr.append(pBdr)


# ─────────────────────────────────────────────────────────────
# Figure map: label → (filename, caption, width)
# ─────────────────────────────────────────────────────────────

FIGURE_MAP = {
    'Figure 1': ('fig01_f1_bar.png', 'Figure 1. OSCD test F1 (τ=0.5) for all configurations: bars show the mean over seeds, error bars the sample SD, black dots individual seeds; n is given above each bar. CNNs are trained with pos_weight=1, Transformers with pos_weight=41.', 5.5),
    'Figure 2': ('fig02_precision_recall.png', 'Figure 2. Precision vs. recall (mean over seeds) on the OSCD test set. Marker size is proportional to parameter count; dotted lines are F1 iso-curves.', 5.0),
    'Figure 3': ('fig03_training_curves_cnn.png', 'Figure 3. Validation F1 per epoch for the CNN models (seed 42). Validation uses 4 cities and is noisy; undefined points (no predicted positives) are gaps.', 5.5),
    'Figure 4': ('fig04_training_curves_transformer.png', 'Figure 4. Validation F1 per epoch for the original Transformer models (seed 42).', 5.5),
    'Figure 5': ('fig05_per_city_heatmap.png', 'Figure 5. Per-city test F1, averaged over seeds, for five CNN models and the 10 OSCD test cities. Colour scale: red = low F1, yellow = medium, green = high; blank cells are undefined (no predicted or true positives).', 6.0),
    'Figure 6': ('fig06_far_vs_f1.png', 'Figure 6. False alarm rate vs. F1 (mean ± sample SD over seeds). The preferred region is the upper-left (high F1, low FAR). Marker size is proportional to parameter count.', 5.0),
    'Figure 7': ('fig07_scenario2.png', 'Figure 7. SEN12-MT Scenario 2 (preliminary) test F1 for three models (τ=0.5, seed 42).', 4.0),
    'Figure 8': ('fig08_qualitative_brasilia.png', 'Figure 8. Qualitative results for Brasilia (OSCD test city, seed 42): SAR and optical inputs at both dates, predictions, probability maps, error maps and ground truth.', 5.5),
    'Figure 9': ('fig09_confusion_breakdown.png', 'Figure 9. TP, FP and FN pixel counts on the OSCD test set for the CNN models (seed 42 runs).', 5.5),
    'Figure 10': ('fig10_architecture_schematic.png', 'Figure 10. Architecture schematics: (left) clean missing-optical dual-stream CNN; (right) Transformer cross-attention model.', 6.0),
    'Figure 11': ('fig11_iou_bar.png', 'Figure 11. OSCD test IoU (mean ± sample SD over seeds; n above each bar).', 5.5),
    'Figure 12': ('fig12_controlled_comparison.png', "Figure 12. Matched-loss comparison: all models trained with pos_weight=1 (mean ± sample SD; dots = seeds). The Transformer's selected checkpoints come from epochs 1–5, before its predictions collapsed to all-negative.", 5.5),
}

# Track which figures have been inserted
figures_inserted = set()

def maybe_insert_figure(label: str):
    if label in figures_inserted:
        return
    if label not in FIGURE_MAP:
        return
    fname, caption, width = FIGURE_MAP[label]
    fig_path = FIGURES_DIR / fname
    add_figure(fig_path, caption, width)
    figures_inserted.add(label)


# ─────────────────────────────────────────────────────────────
# Markdown table parser
# ─────────────────────────────────────────────────────────────

def parse_md_table(lines: list[str]):
    """Return (header, rows) from markdown table lines."""
    header, rows = [], []
    for i, line in enumerate(lines):
        if not line.strip().startswith('|'):
            break
        if re.match(r'^\s*\|[\s\-|:]+\|\s*$', line):
            continue  # separator row
        cells = [c.strip() for c in line.strip().strip('|').split('|')]
        if i == 0:
            header = cells
        else:
            rows.append(cells)
    return header, rows


# ─────────────────────────────────────────────────────────────
# Main parser: walk the markdown line by line
# ─────────────────────────────────────────────────────────────

def build_doc():
    lines = PAPER_MD.read_text(encoding='utf-8').splitlines()
    i = 0
    list_number = 0
    in_code_block = False
    code_lines = []
    skip_note_line = False  # skip the figures directory note line

    while i < len(lines):
        line = lines[i]

        # ── Code block ───────────────────────────────────────
        if line.strip().startswith('```'):
            if not in_code_block:
                in_code_block = True
                code_lines = []
            else:
                in_code_block = False
                add_code_block(code_lines)
                code_lines = []
            i += 1
            continue

        if in_code_block:
            code_lines.append(line)
            i += 1
            continue

        # ── Skip meta notes ──────────────────────────────────
        if line.strip().startswith('> **Figures directory:**') or line.strip().startswith('<!--'):
            i += 1
            continue

        if line.strip() == '---':
            # Only add separator between major sections, skip consecutive ones
            add_separator()
            i += 1
            continue

        # ── Headings ─────────────────────────────────────────
        if line.startswith('#### '):
            text = line[5:].strip()
            h = doc.add_heading(text, level=4)
            set_heading_style(h, 12, bold=True, color=(0, 70, 127))
            list_number = 0
            i += 1
            continue

        if line.startswith('### '):
            text = line[4:].strip()
            h = doc.add_heading(text, level=3)
            set_heading_style(h, 13, bold=True, color=(0, 70, 127))
            list_number = 0
            i += 1
            continue

        if line.startswith('## '):
            text = line[3:].strip()
            # Insert figures before certain sections
            if 'Discussion' in text:
                for fig_lbl in ['Figure 1','Figure 2','Figure 3','Figure 4','Figure 5','Figure 6','Figure 7','Figure 8','Figure 9']:
                    maybe_insert_figure(fig_lbl)
            if 'Conclusion' in text:
                maybe_insert_figure('Figure 10')
                maybe_insert_figure('Figure 11')
                maybe_insert_figure('Figure 12')

            h = doc.add_heading(text, level=2)
            set_heading_style(h, 14, bold=True, color=(0, 51, 102))
            list_number = 0
            i += 1
            continue

        if line.startswith('# '):
            text = line[2:].strip()
            h = doc.add_heading(text, level=1)
            for run in h.runs:
                run.font.name = 'Times New Roman'
                run.font.size = Pt(18)
                run.font.bold = True
                run.font.color.rgb = RGBColor(0, 0, 0)
            h.alignment = WD_ALIGN_PARAGRAPH.CENTER
            h.paragraph_format.space_after = Pt(6)
            i += 1
            continue

        # ── Author / affiliation lines ───────────────────────
        if line.startswith('**Manish'):
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_after = Pt(2)
            apply_inline(p, line.strip(), base_size=12, base_bold=True)
            i += 1
            continue
        if line.startswith('*[Institution]'):
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_after = Pt(2)
            apply_inline(p, line.strip(), base_size=11, base_italic=True)
            i += 1
            continue
        if '@' in line and line.strip() and not line.startswith('#'):
            # email line
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_after = Pt(8)
            apply_inline(p, line.strip(), base_size=11)
            i += 1
            continue

        # ── Keywords line ────────────────────────────────────
        if line.startswith('**Keywords:**'):
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(4)
            p.paragraph_format.space_after  = Pt(8)
            apply_inline(p, line.strip(), base_size=11, base_italic=True)
            i += 1
            continue

        # ── Markdown table ───────────────────────────────────
        if line.strip().startswith('|'):
            table_lines = []
            while i < len(lines) and lines[i].strip().startswith('|'):
                table_lines.append(lines[i])
                i += 1
            header, rows = parse_md_table(table_lines)
            if header:
                add_table_from_rows(header, rows, bold_last=False)
            continue

        # ── Table footnotes (italics lines starting with *) ──
        if line.startswith('*') and not line.startswith('**') and line.endswith('*'):
            add_body(line.strip('* '), italic=True, size=10)
            i += 1
            continue

        # ── Blockquote ───────────────────────────────────────
        if line.startswith('> '):
            quote_lines = []
            while i < len(lines) and lines[i].startswith('> '):
                quote_lines.append(lines[i][2:])
                i += 1
            add_blockquote(' '.join(quote_lines))
            continue

        # ── Numbered list ────────────────────────────────────
        num_match = re.match(r'^(\d+)\.\s+(.+)', line)
        if num_match:
            list_number = int(num_match.group(1))
            add_numbered(num_match.group(2), list_number)
            i += 1
            continue

        # ── Bullet list ──────────────────────────────────────
        if re.match(r'^[-*]\s+', line):
            text = re.sub(r'^[-*]\s+', '', line)
            add_bullet(text)
            i += 1
            continue

        # ── Bold table caption lines (e.g. **Table 1: ...**) ─
        if line.strip().startswith('**Table') and line.strip().endswith('**'):
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(8)
            p.paragraph_format.space_after  = Pt(2)
            apply_inline(p, line.strip(), base_size=11, base_bold=True)
            i += 1
            continue

        # ── Figure label in List of Figures section ──────────
        # (skip — figures are embedded at insertion points above)

        # ── Empty line ───────────────────────────────────────
        if line.strip() == '':
            list_number = 0
            i += 1
            continue

        # ── Italic-only lines (e.g. *Manuscript version: ...*) ─
        if line.strip().startswith('*') and line.strip().endswith('*') and not line.strip().startswith('**'):
            add_body(line.strip('* '), italic=True, size=10, center=True)
            i += 1
            continue

        # ── Regular paragraph ────────────────────────────────
        add_body(line.strip())
        i += 1

    # Ensure all figures inserted at document end if missed
    for lbl in FIGURE_MAP:
        maybe_insert_figure(lbl)


# ─────────────────────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────────────────────

print("Building Word document...")
build_doc()
try:
    doc.save(str(OUT))
    print(f"Saved: paper/RESEARCH_PAPER_FINAL.docx ({OUT.stat().st_size // 1024} KB)")
except PermissionError:
    OUT_ALT = OUT.parent / "RESEARCH_PAPER_FINAL_v2.docx"
    doc.save(str(OUT_ALT))
    print(f"Saved (locked, used alt): paper/RESEARCH_PAPER_FINAL_v2.docx ({OUT_ALT.stat().st_size // 1024} KB)")
