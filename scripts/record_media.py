"""Capture screenshots + animated GIFs of the running web app, for the README.

    uv pip install playwright pillow  &&  python -m playwright install chromium
    bash scripts/serve_web.sh &          # serve build/web on 127.0.0.1:8080
    python scripts/record_media.py       # -> docs/media/

Uses the local static server (http://127.0.0.1:8080). The captures work the same on the
published URL; live-URL badges/links in the README are added after deploy.

The top-bar tab x-centres are computed from the *same* pygame font the app uses
(chrome.MenuBar), then scaled to the CSS viewport -- no hand-measured pixels.
About and Settings have no tab bar, so each is captured after a fresh reload.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

from PIL import Image
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

URL = "http://127.0.0.1:8080"
OUT = ROOT / "docs" / "media"
OUT.mkdir(parents=True, exist_ok=True)

CANVAS_W = 1300                      # app internal size (app/main.py: W, H)
VIEW = {"width": 1320, "height": 880}
SCALE = VIEW["width"] / CANVAS_W


def _tab_centres() -> dict[str, float]:
    """Reproduce chrome.MenuBar's layout to get each tab's click x (CSS px)."""
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    import pygame

    from app.i18n import set_lang, t

    set_lang("en-US")
    pygame.font.init()
    f = pygame.font.Font(None, 20)
    out: dict[str, float] = {}
    x = 12
    for key in ("ga", "dqn", "about", "settings"):
        w = f.size(t(f"menu.{key}"))[0] + 22
        out[key] = (x + w / 2) * SCALE
        x += w + 5
    return out


TAB_X = _tab_centres()
TAB_Y = 20 * (VIEW["height"] / 864)


def _wait_ready(page, timeout=180):
    end = time.time() + timeout
    while time.time() < end:
        st = page.evaluate(
            "() => { const c=document.getElementById('canvas');"
            " const tr=document.getElementById('transfer');"
            " return {w:c?c.width:0, hidden: tr?tr.hasAttribute('hidden'):true}; }"
        )
        if st["w"] > 100 and st["hidden"]:
            time.sleep(3)  # let the first screen settle
            return
        time.sleep(1)
    raise RuntimeError("app did not become ready")


def _shot(page, name):
    p = OUT / name
    page.screenshot(path=str(p))
    print("wrote", p)
    return p


def _gif(frames: list[Path], name: str, width=940, ms=550):
    imgs = []
    for f in frames:
        im = Image.open(f).convert("RGB")
        im = im.resize((width, round(width * im.height / im.width)))
        imgs.append(im)
    dst = OUT / name
    imgs[0].save(dst, save_all=True, append_images=imgs[1:], duration=ms, loop=0, optimize=True)
    print("wrote", dst, f"({dst.stat().st_size // 1024} KB)")


def _click_tab(page, key):
    page.mouse.click(TAB_X[key], TAB_Y)
    time.sleep(2.5)


def main():
    with sync_playwright() as p:
        br = p.chromium.launch(headless=True, args=["--use-gl=swiftshader"])
        ctx = br.new_context(viewport=VIEW)
        page = ctx.new_page()
        page.goto(URL, wait_until="domcontentloaded", timeout=60000)
        _wait_ready(page)

        # --- GA screen: still + training GIF
        _shot(page, "ga.png")
        ga_frames = []
        for i in range(26):
            ga_frames.append(_shot(page, f"_ga_{i:02d}.png"))
            time.sleep(0.7)
        _gif(ga_frames, "ga.gif")

        # --- DQN screen (tab bar present on the GA screen)
        _click_tab(page, "dqn")
        _shot(page, "dqn.png")
        dqn_frames = []
        for i in range(18):
            dqn_frames.append(_shot(page, f"_dqn_{i:02d}.png"))
            time.sleep(0.7)
        _gif(dqn_frames, "dqn.gif")

        # --- Settings and About: reload first (those screens have no tab bar)
        for key in ("settings", "about"):
            page.goto(URL, wait_until="domcontentloaded", timeout=60000)
            _wait_ready(page)
            _click_tab(page, key)
            _shot(page, f"{key}.png")

        for f in OUT.glob("_*.png"):
            f.unlink()
        br.close()


if __name__ == "__main__":
    main()
