# About the project

**Autonomous Vehicle Simulator — Neural Networks with a Genetic Algorithm and DQN.** A
**didactic** project: a neural network learns to drive a car around a winding track using
only distance sensors, and you watch the car on the track and the neurons (and connections)
lighting up at the same time. You can change the hyperparameters and retrain, on the desktop
or in the browser, from the same Python code.

The app opens on **Solution 1 (Neural Net + GA)**; the top menu bar switches to **DQN**,
**About** and **Settings**. Each simulation has its own **Tutorial**.

## Goals

- Make the chain **sensors -> network -> action** and **hyperparameter -> learning** tangible.
- Show, side by side, **two ways to train the same network** for the same problem:
  - **Solution 1 — Genetic Algorithm (neuroevolution):** no gradient; a population of
    networks is selected by performance.
  - **Solution 2 — DQN (Deep Q-Network):** gradient-based; a Q network learns from reward
    with a replay buffer and a target network.
- Run **with nothing to install** (browser) and also offline (desktop).
- Small, readable code, usable as study material.

## How it works (in brief)

- **Sensors:** 5 rays (raycasts) leave the car in a fan and measure the distance to the
  wall. It is the car's "sight" — cheap and deterministic, instead of a real camera.
- **Network:** a shallow MLP in NumPy. Inputs = distances + speed; outputs = steering and
  throttle (Solution 1) or the Q value of each discrete action (Solution 2).
- **Track:** a centre line (spline) + width -> edges; collision and progress are measured by
  distance to the centre line.
- **Thread-free training:** it advances inside the render loop, in chunks — works the same
  in the browser (WebAssembly via pygbag).

## Tech stack

- **Python 3.11+**, **NumPy** (core), **Pygame-CE** (UI), **pygbag** (WebAssembly).
- **uv** (environment), **pytest** + **ruff** (tests and lint), **reportlab** (tutorial PDF),
  **PyInstaller** (native executable).
- Left out on purpose: shapely/scipy (the geometry is ~100 lines of NumPy and runs in
  Pyodide), PyTorch/TensorFlow (the network is small and even the DQN backprop is by hand),
  matplotlib.

## Layout

```
src/core/     config, geometry, track, sensors, network, simulation, evolution, trainer, dqn
src/app/      Pygame: router (main, opens on 'ga') + gascreen, dqnscreen, docscreen,
              settingsscreen + chrome (menu bar), scene, netview, plot, widgets, camera,
              docview, i18n, links
src/tracks/   circuito_1..3 (JSON)
docs/         tutorial_ga.md, tutorial_dqn.md, sobre.md (+ .pt-BR variants), plan, prd, arch
scripts/      build_web.sh, serve_web.sh, deploy_hf.sh, build_exe.py, train_headless.py
```

## How to run

- **Desktop:** `uv pip install -e ".[dev]"` then `uv run python main.py`.
- **Web (local):** `uv pip install -e ".[web,docs]"`, `bash scripts/serve_web.sh`,
  open `http://127.0.0.1:8080` (**not** `localhost`).
- **Native executable:** `uv pip install -e ".[build]"` then `python scripts/build_exe.py`
  (builds for the current OS; the CI builds Windows/macOS/Linux).
- **Publish:** GitHub Pages (workflow ready) or Hugging Face Spaces (Static SDK).

## Licence

Open source (MIT) — didactic material.
