# /// script
# dependencies = ["numpy", "pygame-ce"]
# ///
"""Ponto de entrada — funciona no desktop e no navegador (pygbag).

Desktop:  python main.py
Web:      bash web/serve.sh            ->  abra  http://127.0.0.1:8080
          (estático, sem COOP/COEP, em 127.0.0.1 — igual ao GitHub Pages; ver README)

No desktop os pacotes ficam em `src/`; no bundle web `core/`, `app/` e `tracks/`
são copiados para o lado deste arquivo (layout achatado), então `import app` /
`import core` funcionam direto a partir do diretório de trabalho.

IMPORTANTE p/ pygbag: ele empacota os pacotes vendo os `import` **deste arquivo**
(não faz varredura recursiva). Por isso `pygame` e `numpy` são importados aqui.
"""

import asyncio
import os
import sys

import numpy  # noqa: F401  -- pygbag precisa ver este import para empacotar o numpy
import pygame  # noqa: F401  -- idem: sem isto, o pygame não é empacotado no bundle web

_HERE = os.path.dirname(os.path.abspath(__file__))
for _cand in (os.path.join(_HERE, "src"), _HERE):
    if os.path.isdir(os.path.join(_cand, "app")) and _cand not in sys.path:
        sys.path.insert(0, _cand)

from app.main import main  # noqa: E402

if __name__ == "__main__":
    asyncio.run(main())
