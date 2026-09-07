#!/usr/bin/env bash
# Package the app as WebAssembly (static site) with pygbag.
#
# FLAT layout: pygbag packages the folder of main.py; we copy core/, app/, tracks/
# next to main.py (no 'src/'), so 'import app' works directly in the browser.
#
# Requires:  uv pip install -e ".[web,docs]"   (docs = reportlab, for the PDFs)
# Output:    build/web/  -> index.html + runtime + autonomous_vehicle_simulator.apk
#            + tutorial_ga.html/.pdf, tutorial_dqn.html/.pdf
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PYTHON:-python}"
[ -x .venv/Scripts/python.exe ] && PY=".venv/Scripts/python.exe"
[ -x .venv/bin/python ] && PY=".venv/bin/python"

APP="autonomous_vehicle_simulator"       # becomes the .apk name
APPDIR="build/pygbag/$APP"
OUT="build/web"
TMPL="scripts/assets/pygbag.tmpl"

rm -rf build/pygbag "$OUT"
mkdir -p "$APPDIR" "$OUT"

cp main.py "$APPDIR/"
cp -r src/core src/app src/tracks "$APPDIR/"             # FLAT (no src/)
cp docs/sobre*.md docs/tutorial_*.md "$APPDIR/app/"      # doc assets (all languages)
find "$APPDIR" -name '__pycache__' -type d -prune -exec rm -rf {} +

# standalone reading pages + PDFs served next to the site (English primary docs)
for d in tutorial_ga tutorial_dqn; do
  "$PY" tools/md2html.py "docs/$d.md" "$OUT/$d.html"
  "$PY" tools/md2pdf.py  "docs/$d.md" "$OUT/$d.pdf" || \
    echo "WARNING: reportlab missing, $d.pdf not generated (uv pip install -e \".[docs]\")"
done

"$PY" -m pygbag \
  --build \
  --ume_block 0 \
  --template "$TMPL" \
  --disable-sound-format-error \
  --title "Autonomous Vehicle Simulator" \
  "$APPDIR/main.py"

cp -r "$APPDIR/build/web/." "$OUT/"
rm -rf build/pygbag

echo
echo "OK -> $OUT"
echo "Local test:  $PY -m http.server 8080 --directory $OUT   then open  http://127.0.0.1:8080"
echo "  (NOT 'localhost' -- pygbag treats //localhost: as dev mode and looks for a local"
echo "   package server; 127.0.0.1 uses the public CDN, same as GitHub Pages / HF Spaces)"
du -sh "$OUT"/*.apk 2>/dev/null || true
