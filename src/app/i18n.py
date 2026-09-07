"""Tiny app settings: two-language i18n (en-US / pt-BR) + a simulation-speed cap.

Persists best-effort: browser localStorage under pygbag, a JSON dotfile on desktop.
"""

from __future__ import annotations

import json
import os
import sys

LANGS = ("en-US", "pt-BR")
LANG_LABEL = {"en-US": "English (en-US)", "pt-BR": "Portugues (pt-BR)"}

# Frame-rate cap for the render/sim loop. 0 = uncapped (as fast as the machine draws).
# Lower = the fixed-step simulation advances slower in wall-clock time: handy for
# lectures and for screen/GIF capture. 60 is the default feel.
FPS_CHOICES = (30, 60, 120, 0)

_DEFAULTS = {"lang": "en-US", "fps": 60}
_state = dict(_DEFAULTS)

_STR: dict[str, dict[str, str]] = {
    # window / general
    "app.title": {"en-US": "Autonomous Vehicle Simulator",
                  "pt-BR": "Simulador de Veiculo Autonomo"},

    # top menu bar
    "menu.ga": {"en-US": "Neural Net + GA", "pt-BR": "Rede Neural + AG"},
    "menu.dqn": {"en-US": "DQN", "pt-BR": "DQN"},
    "menu.about": {"en-US": "About", "pt-BR": "Sobre"},
    "menu.settings": {"en-US": "Settings", "pt-BR": "Configuracoes"},
    "menu.tutorial": {"en-US": "Tutorial", "pt-BR": "Tutorial"},
    "nav.back": {"en-US": "< Back", "pt-BR": "< Voltar"},

    # shared controls
    "btn.play": {"en-US": "Play/Pause", "pt-BR": "Play/Pause"},
    "btn.turbo": {"en-US": "Turbo", "pt-BR": "Turbo"},
    "btn.reset": {"en-US": "Reset", "pt-BR": "Reset"},
    "btn.track": {"en-US": "Track >", "pt-BR": "Pista >"},
    "btn.nextgen": {"en-US": "+1 Gen", "pt-BR": "+1 Geracao"},
    "btn.retrain": {"en-US": "Retrain", "pt-BR": "Re-treinar"},
    "btn.pdf": {"en-US": "Download PDF", "pt-BR": "Baixar PDF"},
    "btn.copylink": {"en-US": "Copy link", "pt-BR": "Copiar link"},

    # GA screen
    "ga.title": {"en-US": "Solution 1 - Neural Net + Genetic Algorithm",
                 "pt-BR": "Solucao 1 - Rede Neural + Algoritmo Genetico"},
    "ga.sl.hidden": {"en-US": "Hidden layer", "pt-BR": "Camada oculta"},
    "ga.sl.sensors": {"en-US": "Sensors", "pt-BR": "Sensores"},
    "ga.sl.pop": {"en-US": "Population", "pt-BR": "Populacao"},
    "ga.sl.mrate": {"en-US": "Mutation rate", "pt-BR": "Taxa mutacao"},
    "ga.sl.msigma": {"en-US": "Mutation strength", "pt-BR": "Forca mutacao"},
    "ga.sl.elit": {"en-US": "Elitism", "pt-BR": "Elitismo"},
    "ga.sl.wspeed": {"en-US": "Speed weight", "pt-BR": "Peso velocidade"},
    "ga.sl.crash": {"en-US": "Crash penalty", "pt-BR": "Penal. colisao"},
    "ga.hud.gen": {"en-US": "Generation {g}/{n}   [{s}]", "pt-BR": "Geracao {g}/{n}   [{s}]"},
    "ga.hud.fit": {"en-US": "best fitness {b:6.2f}   mean {m:6.2f}",
                   "pt-BR": "melhor fitness {b:6.2f}   media {m:6.2f}"},
    "ga.hud.laps": {"en-US": "best: {l} lap(s)   track: {t}",
                    "pt-BR": "melhor: {l} volta(s)   pista: {t}"},
    "ga.hud.keys": {"en-US": "space=play/pause  g=+1gen  t=turbo  h=tutorial",
                    "pt-BR": "space=play/pause  g=+1ger  t=turbo  h=tutorial"},
    "ga.hint.apply": {"en-US": "sliders apply on the next Retrain",
                      "pt-BR": "sliders so valem no proximo Re-treinar"},
    "ga.net.training": {"en-US": "training generation 0...", "pt-BR": "treinando geracao 0..."},
    "net.caption": {"en-US": "network:  node = activation   edge = weight x activation",
                    "pt-BR": "rede:  no = ativacao   aresta = peso x ativacao"},
    "net.out.steer": {"en-US": "steer", "pt-BR": "dir"},
    "net.out.accel": {"en-US": "accel", "pt-BR": "acel"},
    "net.in.speed": {"en-US": "spd", "pt-BR": "vel"},

    # DQN screen
    "dqn.title": {"en-US": "Solution 2 - DQN", "pt-BR": "Solucao 2 - DQN"},
    "dqn.sl.hidden": {"en-US": "Hidden units", "pt-BR": "Neuronios ocultos"},
    "dqn.sl.lr": {"en-US": "Learning rate", "pt-BR": "Taxa aprendizado"},
    "dqn.sl.gamma": {"en-US": "Gamma (discount)", "pt-BR": "Gamma (desconto)"},
    "dqn.sl.edecay": {"en-US": "Epsilon decay", "pt-BR": "Decaimento epsilon"},
    "dqn.hud.ep": {"en-US": "Episode {e}   [{s}]", "pt-BR": "Episodio {e}   [{s}]"},
    "dqn.hud.best": {"en-US": "best reward {r:6.1f}   {l} lap(s)",
                     "pt-BR": "melhor recompensa {r:6.1f}   {l} volta(s)"},
    "dqn.hud.stats": {"en-US": "epsilon {e:4.2f}   loss {ls:6.3f}   buffer {b}",
                      "pt-BR": "epsilon {e:4.2f}   loss {ls:6.3f}   buffer {b}"},
    "dqn.hud.greedy": {"en-US": "greedy action now: {a}   track: {t}",
                       "pt-BR": "acao gulosa agora: {a}   pista: {t}"},
    "dqn.hud.keys": {"en-US": "space=play/pause  t=turbo  h=tutorial",
                     "pt-BR": "space=play/pause  t=turbo  h=tutorial"},
    "dqn.hint.apply": {"en-US": "changes apply on Reset", "pt-BR": "mudancas: Reset aplica"},
    "dqn.net.caption": {"en-US": "Q network:  node = activation   output = Q(action)",
                        "pt-BR": "rede Q:  no = ativacao   saida = Q(acao)"},

    # plot
    "plot.fitness": {"en-US": "fitness x generation", "pt-BR": "fitness x geracao"},
    "plot.reward": {"en-US": "reward x episode", "pt-BR": "recompensa x episodio"},

    # doc screen
    "doc.tutorial_ga": {"en-US": "Tutorial - Neural Net + GA",
                        "pt-BR": "Tutorial - Rede Neural + AG"},
    "doc.tutorial_dqn": {"en-US": "Tutorial - DQN", "pt-BR": "Tutorial - DQN"},
    "doc.about": {"en-US": "About the project", "pt-BR": "Sobre o projeto"},
    "doc.pdf.opening": {"en-US": "opening {f}", "pt-BR": "abrindo {f}"},
    "doc.pdf.missing": {"en-US": "install the 'docs' extra: uv pip install -e \".[docs]\"",
                        "pt-BR": "instale o extra 'docs': uv pip install -e \".[docs]\""},
    "doc.link.copied": {"en-US": "link copied: {u}", "pt-BR": "link copiado: {u}"},
    "doc.link.show": {"en-US": "link: {u}", "pt-BR": "link: {u}"},

    # settings screen
    "set.title": {"en-US": "Settings", "pt-BR": "Configuracoes"},
    "set.language": {"en-US": "Language", "pt-BR": "Idioma"},
    "set.language.hint": {"en-US": "Applies immediately to the whole app.",
                          "pt-BR": "Aplica imediatamente a todo o app."},
    "set.speed": {"en-US": "Simulation speed", "pt-BR": "Velocidade da simulacao"},
    "set.speed.hint": {
        "en-US": "Frames per second the loop runs at. Lower = slower, steadier motion "
                 "(good for lectures and screen capture). Training stays deterministic.",
        "pt-BR": "Quadros por segundo do loop. Menor = movimento mais lento e estavel "
                 "(bom para aula e captura de tela). O treino continua deterministico."},
    "set.fps.value": {"en-US": "{v} FPS", "pt-BR": "{v} FPS"},
    "set.fps.uncapped": {"en-US": "Uncapped", "pt-BR": "Sem limite"},
}


# ---------------------------------------------------------------- language
def get_lang() -> str:
    return _state["lang"]


def set_lang(lang: str) -> None:
    if lang in LANGS and lang != _state["lang"]:
        _state["lang"] = lang
        _save()


def t(key: str, **fmt) -> str:
    row = _STR.get(key, {})
    s = row.get(_state["lang"]) or row.get("en-US") or key
    return s.format(**fmt) if fmt else s


# ---------------------------------------------------------------- sim speed
def get_fps() -> int:
    return int(_state["fps"])


def set_fps(fps: int) -> None:
    if fps in FPS_CHOICES and fps != _state["fps"]:
        _state["fps"] = int(fps)
        _save()


def fps_label(fps: int) -> str:
    return t("set.fps.uncapped") if fps == 0 else t("set.fps.value", v=fps)


# ---------------------------------------------------------------- persistence
def _dotfile() -> str:
    return os.path.join(os.path.expanduser("~"), ".avs_settings.json")


def _save() -> None:
    try:
        if sys.platform == "emscripten":
            import platform  # pygbag runtime

            platform.window.localStorage.setItem("avs_settings", json.dumps(_state))
        else:
            with open(_dotfile(), "w", encoding="utf-8") as f:
                json.dump(_state, f)
    except Exception:
        pass


def _coerce(raw: dict) -> None:
    if raw.get("lang") in LANGS:
        _state["lang"] = raw["lang"]
    try:
        if int(raw.get("fps")) in FPS_CHOICES:
            _state["fps"] = int(raw["fps"])
    except (TypeError, ValueError):
        pass


def _load() -> None:
    try:
        if sys.platform == "emscripten":
            import platform

            blob = platform.window.localStorage.getItem("avs_settings")
            if blob:
                _coerce(json.loads(blob))
            else:  # migrate the old single-key format
                old = platform.window.localStorage.getItem("avs_lang")
                if old in LANGS:
                    _state["lang"] = old
        else:
            path = _dotfile()
            if os.path.isfile(path):
                with open(path, encoding="utf-8") as fh:
                    _coerce(json.load(fh))
    except Exception:
        pass


_load()
