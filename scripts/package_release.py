"""Zip the built executable into a named, downloadable release asset.

Run after `scripts/build_exe.py`, on each target OS (PyInstaller does not
cross-compile). The CI matrix in `.github/workflows/release.yml` calls this.

    python scripts/build_exe.py
    python scripts/package_release.py
    # -> dist/AutonomousVehicleSimulator-<version>-<os>-<arch>.zip
    #    e.g. ...-0.1.0-windows-x64.zip  /  ...-0.1.0-macos-arm64.zip  /  ...-0.1.0-linux-x64.zip

The executable inside the zip is renamed to carry the same os-arch tag, so
macOS and Linux binaries (both plain `AutonomousVehicleSimulator`) never collide
when unpacked side by side. The macOS `.app` bundle is kept intact.

Version defaults to `src/__init__.py:__version__`; `--version` overrides it (the
CI passes the tag, e.g. `v0.1.0` -> `0.1.0`). `--os-name` / `--arch` override the
auto-detected platform.
"""

from __future__ import annotations

import argparse
import platform
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAME = "AutonomousVehicleSimulator"
DIST = ROOT / "dist"
EXTRA = ("README.md", "CHANGELOG.md", "LICENSE")


def _version() -> str:
    ns: dict[str, str] = {}
    exec((ROOT / "src" / "__init__.py").read_text(encoding="utf-8"), ns)  # noqa: S102
    return ns["__version__"]


def _os_name() -> str:
    return {"windows": "windows", "darwin": "macos", "linux": "linux"}.get(
        platform.system().lower(), platform.system().lower()
    )


def _arch() -> str:
    m = platform.machine().lower()
    return {"amd64": "x64", "x86_64": "x64", "x64": "x64",
            "arm64": "arm64", "aarch64": "arm64"}.get(m, m or "unknown")


def _payload() -> list[Path]:
    """Executable artifacts PyInstaller produced for this OS (binary, .exe, .app)."""
    return sorted(p for p in DIST.glob(f"{NAME}*") if p.suffix != ".zip")


def _arcname(path: Path, root: str, tag: str) -> str:
    """Rename the plain executable to <NAME>-<tag>[.exe]; keep .app bundles as-is."""
    if path.is_dir() or path.suffix not in ("", ".exe"):
        return f"{root}/{path.name}"
    return f"{root}/{NAME}-{tag}{path.suffix}"


def _add(zf: zipfile.ZipFile, path: Path, root: str, tag: str) -> None:
    if path.is_dir():
        base = f"{root}/{path.name}"
        for sub in sorted(path.rglob("*")):
            if sub.is_file():
                zf.write(sub, f"{base}/{sub.relative_to(path).as_posix()}")
    else:
        zf.write(path, _arcname(path, root, tag))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default=None)
    ap.add_argument("--os-name", default=None, dest="os_name")
    ap.add_argument("--arch", default=None)
    args = ap.parse_args()

    version = (args.version or _version()).lstrip("v")
    tag = f"{args.os_name or _os_name()}-{args.arch or _arch()}"

    payload = _payload()
    if not payload:
        print(f"error: no {NAME}* artifact in {DIST} - run scripts/build_exe.py first")
        return 1

    root = f"{NAME}-{version}-{tag}"
    zip_path = DIST / f"{root}.zip"
    zip_path.unlink(missing_ok=True)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in payload:
            _add(zf, p, root, tag)
        for extra in EXTRA:
            src = ROOT / extra
            if src.is_file():
                zf.write(src, f"{root}/{extra}")

    size_mb = zip_path.stat().st_size / 1e6
    print(f"OK -> {zip_path}  ({size_mb:.1f} MB)")
    print("  packed:", ", ".join(p.name for p in payload))
    return 0


if __name__ == "__main__":
    sys.exit(main())
