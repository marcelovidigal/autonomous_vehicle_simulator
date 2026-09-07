"""Minimal scrollable Markdown renderer, used by the Tutorial and About screens.

Supports: headings, paragraphs, lists, fenced code blocks, rule (`---`).
Scroll with wheel, arrows, PageUp/PageDown, Home/End.
"""

from __future__ import annotations

import re
from pathlib import Path

import pygame

from . import theme

PAD = 26
HEADER_H = 8  # the top nav bar is drawn by the screen


def load_doc(name: str, lang: str = "en-US") -> str:
    """Read a doc by name, preferring the `<name>.<lang>.md` variant then `<name>.md`.

    `<name>.md` is the English (default) copy; `<name>.pt-BR.md` is the Portuguese one.
    """
    here = Path(__file__).resolve()
    bases = (here.parent, here.parents[2] / "docs", Path.cwd() / "docs")
    variants = [f"{name}.{lang}.md", f"{name}.md"] if lang != "en-US" else [f"{name}.md"]
    for base in bases:
        for v in variants:
            cand = base / v
            try:
                if cand.is_file():
                    return cand.read_text(encoding="utf-8")
            except OSError:
                pass
    return f"# {name}\n\n`docs/{name}.md` not found."


def _strip_inline(text: str) -> str:
    return text.replace("**", "").replace("`", "").replace("*", "")


def _parse(md: str):
    blocks: list[tuple[str, str]] = []
    para: list[str] = []
    in_code = False

    def flush():
        if para:
            blocks.append(("p", " ".join(para)))
            para.clear()

    for line in md.splitlines():
        if line.strip().startswith("```"):
            flush()
            in_code = not in_code
            blocks.append(("code_gap", ""))
            continue
        if in_code:
            blocks.append(("code", line))
            continue
        s = line.strip()
        if not s:
            flush()
            blocks.append(("gap", ""))
        elif s.startswith("### "):
            flush()
            blocks.append(("h3", s[4:]))
        elif s.startswith("## "):
            flush()
            blocks.append(("h2", s[3:]))
        elif s.startswith("# "):
            flush()
            blocks.append(("h1", s[2:]))
        elif s == "---":
            flush()
            blocks.append(("hr", ""))
        elif s.startswith("- "):
            flush()
            blocks.append(("li", s[2:]))
        elif re.match(r"\d+\.\s+", s):
            flush()
            blocks.append(("li", s))
        else:
            para.append(s)
    flush()
    return blocks


class DocView:
    def __init__(self, rect: pygame.Rect, markdown: str):
        self.rect = rect
        self.scroll = 0
        self.f_h1 = pygame.font.Font(None, 36)
        self.f_h2 = pygame.font.Font(None, 28)
        self.f_h3 = pygame.font.Font(None, 23)
        self.f_p = pygame.font.Font(None, 20)
        self.f_code = pygame.font.Font(None, 18)
        self._rows: list[tuple] = []
        self._build(markdown)
        self._max_scroll = max(0, self._total - (self.rect.h - HEADER_H - PAD))

    def set_rect(self, rect: pygame.Rect) -> None:
        self.rect = rect
        self._build(self._md)
        self._max_scroll = max(0, self._total - (self.rect.h - HEADER_H - PAD))
        self.scroll = min(self.scroll, self._max_scroll)

    def _wrap(self, text: str, font: pygame.font.Font, max_w: int) -> list[str]:
        out, cur = [], ""
        for word in text.split(" "):
            trial = f"{cur} {word}".strip()
            if not cur or font.size(trial)[0] <= max_w:
                cur = trial
            else:
                out.append(cur)
                cur = word
        if cur:
            out.append(cur)
        return out or [""]

    def _build(self, markdown: str) -> None:
        self._md = markdown
        width = self.rect.w - 2 * PAD
        rows: list[tuple] = []
        spec = {
            "h1": (self.f_h1, theme.ACCENT, 12, 0),
            "h2": (self.f_h2, theme.TEXT, 10, 0),
            "h3": (self.f_h3, theme.TEXT, 7, 0),
            "p": (self.f_p, theme.TEXT, 5, 0),
            "li": (self.f_p, theme.TEXT, 3, 18),
        }
        for kind, text in _parse(markdown):
            if kind == "gap":
                rows.append((None, 0, 7, False))
            elif kind == "code_gap":
                rows.append((None, 0, 3, False))
            elif kind == "hr":
                rows.append(("hr", 0, 12, False))
            elif kind == "code":
                surf = self.f_code.render(_strip_inline(text) or " ", True, (168, 208, 176))
                rows.append((surf, 8, 0, True))
            else:
                font, col, gap, indent = spec[kind]
                if kind in ("h1", "h2", "h3"):
                    rows.append((None, 0, 8, False))
                body = ("•  " + _strip_inline(text)) if kind == "li" else _strip_inline(text)
                wrapped = self._wrap(body, font, width - indent)
                for i, wl in enumerate(wrapped):
                    surf = font.render(wl, True, col)
                    rows.append((surf, indent if i == 0 else indent + 14,
                                 gap if i == len(wrapped) - 1 else 1, False))
        self._rows = rows
        self._total = sum(
            (0 if r[0] is None else (2 if r[0] == "hr" else r[0].get_height())) + r[2]
            for r in rows
        )

    def handle(self, ev) -> None:
        page = self.rect.h - HEADER_H - 80
        if ev.type == pygame.MOUSEWHEEL:
            self.scroll -= ev.y * 60
        elif ev.type == pygame.KEYDOWN:
            self.scroll += {
                pygame.K_DOWN: 60, pygame.K_UP: -60,
                pygame.K_PAGEDOWN: page, pygame.K_PAGEUP: -page, pygame.K_SPACE: page,
            }.get(ev.key, 0)
            if ev.key == pygame.K_HOME:
                self.scroll = 0
            elif ev.key == pygame.K_END:
                self.scroll = self._max_scroll
        self.scroll = max(0, min(self.scroll, self._max_scroll))

    def draw(self, surf) -> None:
        r = self.rect
        pygame.draw.rect(surf, theme.BG, r)
        top = r.y + HEADER_H
        clip = pygame.Rect(r.x + 1, top, r.w - 2, r.bottom - top - 1)
        surf.set_clip(clip)
        y = top - self.scroll
        for item, indent, gap, is_code in self._rows:
            h = 0 if item is None else (2 if item == "hr" else item.get_height())
            if clip.top - 40 <= y <= clip.bottom:
                if item == "hr":
                    pygame.draw.line(surf, theme.PANEL_LINE, (r.x + PAD, y),
                                     (r.right - PAD, y), 1)
                elif item is not None:
                    if is_code:
                        pygame.draw.rect(surf, theme.PANEL,
                                         (r.x + PAD - 6, y - 1, r.w - 2 * PAD + 12,
                                          item.get_height() + 2))
                    surf.blit(item, (r.x + PAD + indent, y))
            y += h + gap
        surf.set_clip(None)
        if self._max_scroll > 0:
            th = clip.h
            knob = max(24, int(th * th / (th + self._max_scroll)))
            ky = top + int((th - knob) * self.scroll / self._max_scroll)
            pygame.draw.rect(surf, theme.PANEL_LINE, (r.right - 6, ky, 3, knob), border_radius=2)
