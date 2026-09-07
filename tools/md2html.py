"""Conversor Markdown -> HTML minimo e sem dependencias.

Cobre o subconjunto usado em docs/tutorial.md: titulos, paragrafos, listas,
blocos de codigo cercados, regua, **negrito** e `codigo`.

Uso:  python tools/md2html.py entrada.md saida.html
"""

from __future__ import annotations

import html
import re
import sys
from pathlib import Path

_STYLE = """
:root { color-scheme: light dark; }
body { max-width: 46rem; margin: 2rem auto; padding: 0 1.1rem;
       font: 16px/1.65 system-ui, -apple-system, Segoe UI, Roboto, sans-serif; }
h1 { font-size: 1.9rem; margin: 1.6rem 0 .6rem; }
h2 { font-size: 1.4rem; margin: 1.8rem 0 .5rem;
     border-bottom: 1px solid #8884; padding-bottom: .2rem; }
h3 { font-size: 1.15rem; margin: 1.4rem 0 .4rem; }
code { font-family: ui-monospace, Consolas, monospace; font-size: .92em;
       background: #8882; padding: .1em .35em; border-radius: 4px; }
pre { background: #8881; border: 1px solid #8883; border-radius: 8px;
      padding: .8rem 1rem; overflow-x: auto; }
pre code { background: none; padding: 0; }
hr { border: none; border-top: 1px solid #8884; margin: 2rem 0; }
li { margin: .2rem 0; }
a { color: #3b82f6; }
.bar { position: sticky; top: 0; background: Canvas; padding: .5rem 0;
       border-bottom: 1px solid #8884; margin-bottom: 1rem; font-size: .9rem; }
.bar a, .bar button { margin-right: 1rem; }
button { font: inherit; cursor: pointer; }
"""

def _bar(pdf_name: str) -> str:
    return (
        "<div class=bar>"
        '<a href="index.html">&#9664; App</a>'
        f'<a href="{html.escape(pdf_name)}">Download PDF</a>'
        '<button onclick="window.print()">Print / Save as PDF</button>'
        "</div>"
    )


def _inline(text: str) -> str:
    text = html.escape(text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"`([^`]+?)`", r"<code>\1</code>", text)
    return text


def convert(md: str, pdf_name: str = "tutorial.pdf") -> str:
    out: list[str] = []
    lines = md.splitlines()
    i = 0
    para: list[str] = []
    list_open = False

    def flush_para():
        if para:
            out.append("<p>" + " ".join(_inline(p) for p in para) + "</p>")
            para.clear()

    def close_list():
        nonlocal list_open
        if list_open:
            out.append("</ul>")
            list_open = False

    while i < len(lines):
        line = lines[i]
        if line.strip().startswith("```"):
            flush_para()
            close_list()
            i += 1
            buf = []
            while i < len(lines) and not lines[i].strip().startswith("```"):
                buf.append(html.escape(lines[i]))
                i += 1
            out.append("<pre><code>" + "\n".join(buf) + "</code></pre>")
            i += 1
            continue

        s = line.strip()
        heading = next((n for n, p in ((1, "# "), (2, "## "), (3, "### ")) if s.startswith(p)), 0)
        if not s:
            flush_para()
            close_list()
        elif heading:
            flush_para()
            close_list()
            out.append(f"<h{heading}>{_inline(s[heading + 1:])}</h{heading}>")
        elif s == "---":
            flush_para()
            close_list()
            out.append("<hr>")
        elif s.startswith("- ") or re.match(r"\d+\.\s+", s):
            flush_para()
            if not list_open:
                out.append("<ul>")
                list_open = True
            item = s[2:] if s.startswith("- ") else re.sub(r"^\d+\.\s+", "", s)
            out.append(f"<li>{_inline(item)}</li>")
        else:
            para.append(s)
        i += 1

    flush_para()
    close_list()
    title = next((ln[2:] for ln in lines if ln.startswith("# ")), "Tutorial")
    return (
        "<!doctype html><html lang=en><head><meta charset=utf-8>"
        "<meta name=viewport content='width=device-width,initial-scale=1'>"
        f"<title>{html.escape(title)}</title><style>{_STYLE}</style></head>"
        f"<body>\n{_bar(pdf_name)}\n" + "\n".join(out) + "\n</body></html>\n"
    )


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: python tools/md2html.py in.md out.html")
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    pdf = dst.with_suffix(".pdf").name
    dst.write_text(convert(src.read_text(encoding="utf-8"), pdf), encoding="utf-8")
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()
