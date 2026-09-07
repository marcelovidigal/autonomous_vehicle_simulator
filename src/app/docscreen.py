"""Screen showing a Markdown document (per-technique Tutorial, or About).

Opened from inside a simulation; the Back button returns to the origin simulation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pygame

from . import links, theme
from .chrome import TOPBAR_H, BackBar
from .docview import DocView, load_doc
from .i18n import get_lang, t
from .widgets import Button

_IS_WEB = sys.platform == "emscripten"


class DocScreen:
    def __init__(self, size, doc_name: str, title: str, back: str,
                 with_pdf: bool = False, with_link: bool = False):
        self.w, self.h = size
        self.goto: str | None = None
        self.back = back
        self.doc_name = doc_name
        self.f = pygame.font.Font(None, 19)
        self._msg = ""
        self._msg_t = 0.0

        self.bar = BackBar(title)
        self.b_pdf = Button(0, 7, 132, 26, t("btn.pdf")) if with_pdf else None
        self.b_link = (Button(0, 7, 120, 26, t("btn.copylink"))
                       if with_link and links.PUBLISHED_URL else None)

        rect = pygame.Rect(60, TOPBAR_H + 10, self.w - 120, self.h - TOPBAR_H - 20)
        self.view = DocView(rect, load_doc(doc_name, get_lang()))

    def handle(self, ev) -> None:
        self.bar.handle(ev)
        if self.b_pdf is not None:
            self.b_pdf.handle(ev)
        if self.b_link is not None:
            self.b_link.handle(ev)
        self.view.handle(ev)
        if ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE:
            self.goto = self.back

    def update(self, dt) -> None:
        if self.bar.poll():
            self.goto = self.back
        if self.b_pdf is not None and self.b_pdf.poll():
            self._open_pdf()
        if self.b_link is not None and self.b_link.poll():
            self._copy_link()
        if self._msg:
            self._msg_t -= dt
            if self._msg_t <= 0:
                self._msg = ""

    def _open_pdf(self) -> None:
        import webbrowser

        if _IS_WEB:
            webbrowser.open(f"{self.doc_name}.pdf")      # served next to the site
            return
        if getattr(sys, "frozen", False):               # PyInstaller bundle
            pdf = Path(getattr(sys, "_MEIPASS", ".")) / "app" / f"{self.doc_name}.pdf"
            if pdf.is_file():
                webbrowser.open(pdf.as_uri())
                self._flash(t("doc.pdf.opening", f=pdf.name))
            else:
                self._flash(t("doc.pdf.missing"))
            return
        repo = Path(__file__).resolve().parents[2]
        pdf = repo / "docs" / f"{self.doc_name}.pdf"
        if not pdf.is_file():
            md = repo / "docs" / f"{self.doc_name}.md"
            try:
                import subprocess

                subprocess.run([sys.executable, str(repo / "tools" / "md2pdf.py"),
                                str(md), str(pdf)], check=True, capture_output=True)
            except Exception:
                self._flash(t("doc.pdf.missing"))
                return
        if pdf.is_file():
            webbrowser.open(pdf.as_uri())
            self._flash(t("doc.pdf.opening", f=pdf.name))

    def _copy_link(self) -> None:
        url = links.PUBLISHED_URL
        ok = False
        try:
            if _IS_WEB:
                import platform  # pygbag runtime

                platform.window.navigator.clipboard.writeText(url)
                ok = True
            else:
                pygame.scrap.put_text(url)   # pygame-ce 2.5+, no scrap.init() needed
                ok = True
        except Exception:
            ok = False
        self._flash(t("doc.link.copied", u=url) if ok else t("doc.link.show", u=url))

    def _flash(self, text: str) -> None:
        self._msg, self._msg_t = text, 3.0

    def draw(self, surf) -> None:
        surf.fill(theme.BG)
        self.view.draw(surf)
        self.bar.draw(surf, self.w)
        right = self.w - 12
        if self.b_pdf is not None:
            self.b_pdf.rect.topleft = (right - self.b_pdf.rect.w, 7)
            self.b_pdf.draw(surf, self.f)
            right -= self.b_pdf.rect.w + 8
        if self.b_link is not None:
            self.b_link.rect.topleft = (right - self.b_link.rect.w, 7)
            self.b_link.draw(surf, self.f)
        if self._msg:
            surf.blit(self.f.render(self._msg, True, theme.ACCENT), (70, self.h - 26))
