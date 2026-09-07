"""Convert a Markdown doc to PDF using reportlab.

Install the extra:  uv pip install -e ".[docs]"
Usage:           python tools/md2pdf.py in.md out.pdf
"""

from __future__ import annotations

import html
import re
import sys
from pathlib import Path


def _inline(text: str) -> str:
    text = html.escape(text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"`([^`]+?)`", r'<font face="Courier">\1</font>', text)
    return text


def build(md: str, dst: Path) -> None:
    title = next((ln[2:] for ln in md.splitlines() if ln.startswith("# ")), "Tutorial")
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.platypus import (
        HRFlowable,
        ListFlowable,
        ListItem,
        Paragraph,
        Preformatted,
        SimpleDocTemplate,
        Spacer,
    )

    ss = getSampleStyleSheet()
    body = ParagraphStyle("body", parent=ss["BodyText"], fontSize=10, leading=14, alignment=TA_LEFT)
    code = ParagraphStyle("code", parent=ss["Code"], fontSize=8, leading=10,
                          backColor="#f2f2f2", borderPadding=4)
    h1 = ParagraphStyle("h1", parent=ss["Title"], fontSize=20, spaceAfter=10)
    h2 = ParagraphStyle("h2", parent=ss["Heading2"], fontSize=14, spaceBefore=14, spaceAfter=6)
    h3 = ParagraphStyle("h3", parent=ss["Heading3"], fontSize=11, spaceBefore=10, spaceAfter=4)

    flow = []
    para: list[str] = []
    bullets: list[str] = []
    in_code = False
    code_buf: list[str] = []

    def flush_para():
        if para:
            flow.append(Paragraph(" ".join(_inline(p) for p in para), body))
            para.clear()

    def flush_bullets():
        if bullets:
            flow.append(ListFlowable(
                [ListItem(Paragraph(_inline(b), body), leftIndent=10) for b in bullets],
                bulletType="bullet", start="•",
            ))
            bullets.clear()

    for line in md.splitlines():
        if line.strip().startswith("```"):
            if in_code:
                flow.append(Preformatted("\n".join(code_buf), code))
                flow.append(Spacer(1, 6))
                code_buf.clear()
            else:
                flush_para()
                flush_bullets()
            in_code = not in_code
            continue
        if in_code:
            code_buf.append(line)
            continue
        s = line.strip()
        head = next((n for n, p in ((1, "# "), (2, "## "), (3, "### ")) if s.startswith(p)), 0)
        if not s:
            flush_para()
            flush_bullets()
        elif head:
            flush_para()
            flush_bullets()
            flow.append(Paragraph(_inline(s[head + 1:]), {1: h1, 2: h2, 3: h3}[head]))
        elif s == "---":
            flush_para()
            flush_bullets()
            flow.append(HRFlowable(width="100%", color="#bbbbbb", spaceBefore=8, spaceAfter=8))
        elif s.startswith("- ") or re.match(r"\d+\.\s+", s):
            flush_para()
            bullets.append(s[2:] if s.startswith("- ") else re.sub(r"^\d+\.\s+", "", s))
        else:
            para.append(s)
    flush_para()
    flush_bullets()

    doc = SimpleDocTemplate(str(dst), pagesize=A4,
                            leftMargin=2 * cm, rightMargin=2 * cm,
                            topMargin=1.8 * cm, bottomMargin=1.8 * cm,
                            title=title)
    doc.build(flow)


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: python tools/md2pdf.py in.md out.pdf")
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    try:
        build(src.read_text(encoding="utf-8"), dst)
    except ImportError:
        raise SystemExit("reportlab missing: uv pip install -e \".[docs]\"") from None
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()
