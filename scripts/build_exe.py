"""Build a native executable of the app with PyInstaller (current OS only).

PyInstaller does NOT cross-compile: run this on each target OS, or let the CI matrix
(.github/workflows/release.yml) build Windows / macOS / Linux.

    uv pip install -e ".[build,docs]"
    python scripts/build_exe.py            # -> dist/AutonomousVehicleSimulator[.exe]
    python scripts/build_exe.py --console  # keep a console window (debugging)
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAME = "AutonomousVehicleSimulator"
DOCS = ("sobre", "tutorial_ga", "tutorial_dqn")
SEP = ";" if os.name == "nt" else ":"


def _md_files() -> list[Path]:
    out: list[Path] = []
    for d in DOCS:
        out.append(ROOT / "docs" / f"{d}.md")
        pt = ROOT / "docs" / f"{d}.pt-BR.md"
        if pt.is_file():
            out.append(pt)
    return out


def _build_pdfs(tmp: Path) -> list[Path]:
    pdfs: list[Path] = []
    try:
        import reportlab  # noqa: F401
    except ImportError:
        print("note: reportlab missing -> the in-app 'Download PDF' will be disabled")
        return pdfs
    for d in ("tutorial_ga", "tutorial_dqn"):
        dst = tmp / f"{d}.pdf"
        subprocess.run([sys.executable, str(ROOT / "tools" / "md2pdf.py"),
                        str(ROOT / "docs" / f"{d}.md"), str(dst)], check=True)
        pdfs.append(dst)
    return pdfs


def main() -> None:
    console = "--console" in sys.argv
    import PyInstaller.__main__

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        assets = [(p, "app") for p in _md_files() + _build_pdfs(tmp)]
        assets.append((ROOT / "src" / "tracks", "tracks"))

        args = [
            str(ROOT / "main.py"),
            "--name", NAME,
            "--onefile",
            "--noconfirm",
            "--clean",
            "--paths", str(ROOT / "src"),
            "--collect-submodules", "app",
            "--collect-submodules", "core",
            "--hidden-import", "pygame",
            "--distpath", str(ROOT / "dist"),
            "--workpath", str(tmp / "work"),
            "--specpath", str(tmp),
        ]
        args += [] if console else ["--windowed"]
        for src, dest in assets:
            args += ["--add-data", f"{src}{SEP}{dest}"]

        print("pyinstaller", " ".join(args))
        PyInstaller.__main__.run(args)

    exe = ROOT / "dist" / (NAME + (".exe" if os.name == "nt" else ""))
    print(f"\nOK -> {exe}" if exe.exists() else f"\nbuilt in {ROOT / 'dist'}")


if __name__ == "__main__":
    main()
