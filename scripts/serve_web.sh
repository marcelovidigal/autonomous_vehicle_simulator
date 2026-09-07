#!/usr/bin/env bash
# Local preview of the web version — same as GitHub Pages / HF Spaces:
# static files, NO COOP/COEP headers, served on 127.0.0.1.
#
# Do NOT use 'localhost' and do NOT use `python -m pygbag` to serve:
#  - on 'localhost' pygbag switches to "dev mode" and looks for the Python wheels
#    (NumPy) on a local /cdn/ server a static server does not have -> stuck.
#  - `python -m pygbag`'s own server forces COEP `require-corp` with a broken CORP
#    header (pygbag 0.9.3 bug) -> ERR_BLOCKED_BY_RESPONSE ...ByCoep.
#  - on 127.0.0.1 pygbag does NOT enter dev mode: it downloads runtime + NumPy +
#    pygame from the public CDN (which sends `Access-Control-Allow-Origin: *`),
#    exactly like the published site.
#
# Usage:  bash scripts/serve_web.sh    ->  open  http://127.0.0.1:8080
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PYTHON:-python}"
[ -x .venv/Scripts/python.exe ] && PY=".venv/Scripts/python.exe"
[ -x .venv/bin/python ] && PY=".venv/bin/python"

OUT="build/web"
[ -f "$OUT/index.html" ] || bash scripts/build_web.sh

echo ">>> open  http://127.0.0.1:8080   (NOT 'localhost')   -- Ctrl+C to stop"
exec "$PY" -m http.server 8080 --bind 127.0.0.1 --directory "$OUT"
