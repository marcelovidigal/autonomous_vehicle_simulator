import os
import subprocess
import sys

import pytest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame  # noqa: E402


def test_docview_builds_and_scrolls():
    pygame.init()
    from app.docview import DocView, load_doc

    screen = pygame.Surface((1200, 820))
    dv = DocView(pygame.Rect(20, 20, 1000, 720), load_doc("tutorial_ga"))
    assert dv._total > 800          # the markdown loaded and rendered
    dv.draw(screen)

    dv.handle(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_END))
    assert dv.scroll == dv._max_scroll > 0
    dv.handle(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_HOME))
    assert dv.scroll == 0
    dv.draw(screen)


@pytest.mark.parametrize("name", ["tutorial_ga", "tutorial_dqn", "sobre"])
def test_load_doc_finds_all(name):
    from app.docview import load_doc

    assert "not found" not in load_doc(name)
    assert len(load_doc(name)) > 400


def test_load_doc_language_variants():
    from app.docview import load_doc

    en = load_doc("tutorial_ga", "en-US")
    pt = load_doc("tutorial_ga", "pt-BR")
    assert "genetic algorithm" in en.lower()
    assert "algoritmo gen" in pt.lower()


def test_tutorials_are_technique_exclusive():
    from app.docview import load_doc

    ga = load_doc("tutorial_ga").lower()
    dqn = load_doc("tutorial_dqn").lower()
    assert "genetic algorithm" in ga and "replay" not in ga
    assert "replay" in dqn and "crossover" not in dqn


@pytest.mark.parametrize("doc", ["tutorial_ga", "tutorial_dqn"])
def test_md2html_generates_page(tmp_path, doc):
    out = tmp_path / f"{doc}.html"
    subprocess.run(
        [sys.executable, "tools/md2html.py", f"docs/{doc}.md", str(out)], check=True
    )
    page = out.read_text(encoding="utf-8")
    assert "<h1>" in page and "<pre><code>" in page


@pytest.mark.parametrize("doc", ["tutorial_ga", "tutorial_dqn"])
def test_md2pdf_generates_pdf(tmp_path, doc):
    pytest.importorskip("reportlab", reason="reportlab not installed (extra 'docs')")
    out = tmp_path / f"{doc}.pdf"
    subprocess.run(
        [sys.executable, "tools/md2pdf.py", f"docs/{doc}.md", str(out)], check=True
    )
    data = out.read_bytes()
    assert data[:4] == b"%PDF" and len(data) > 2000
