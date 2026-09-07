# Changelog

All notable changes to **Autonomous Vehicle Simulator** are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the
project uses [Semantic Versioning](https://semver.org/).

The release workflow ([.github/workflows/release.yml](.github/workflows/release.yml))
turns the section for a tag `vX.Y.Z` into that GitHub Release's notes, and appends
the auto-generated commit/PR list.

## [Unreleased]

### Fixed
- The `build` extra now pulls in `pygame-ce` — it is needed at PyInstaller freeze
  time, so `pip install -e ".[build]"` alone previously produced an executable that
  crashed on `import pygame` (this also affected the release binaries).
  `scripts/build_exe.py` now fails fast if `pygame`/`numpy` are missing.
- CI test jobs use `actions/setup-python` + plain `pip` instead of
  `uv pip install --system` (no reliable target Python on the runner).
- Workflows bumped to `actions/checkout@v5` / `actions/setup-python@v6` (Node 24).
- `deploy-pages.yml`: dropped `configure-pages` `enablement: true` — the default
  `GITHUB_TOKEN` cannot create the Pages site; enable it once in repo Settings.

### Added
- GA and DQN screens: an **Apply** button (key `a`) that rebuilds the config from the
  sliders and restarts training, so a change like the number of hidden units takes effect
  at once. It highlights while there are unapplied slider changes. On the GA screen it
  replaces the old **Retrain** button (same behaviour); the DQN screen previously had no
  way to apply slider changes without resetting them to the defaults.
- Settings: **Simulation speed** (frame-rate cap: 30 / 60 / 120 / Uncapped), persisted
  like the language choice. Useful for lectures and screen capture.
- About screen: **Copy link** button, shown once `src/app/links.py:PUBLISHED_URL` is set.
- `.gitattributes` normalising line endings to LF.
- CI: `package-smoke` job builds the executable on every PR to catch packaging breakage
  early.

### Changed
- Release assets are named `AutonomousVehicleSimulator-<version>-<os>-<arch>.zip` and the
  executable inside carries the same `os-arch` tag, so macOS and Linux binaries no longer
  collide. Release notes list the actual asset files.
- `deploy-pages.yml` caches the pygbag runtime download.
- Settings persistence moved from a single `avs_lang` key to an `avs_settings` JSON blob
  (the old key is still read once, for migration).

## [0.1.0] - 2026-09-06

First public release.

### Added
- **Simulator core** (`src/core/`, NumPy + stdlib only, runs in Pyodide/WASM):
  raycast distance sensors, spline track model with edges/collision/progress,
  shallow MLP, deterministic incremental training.
- **Solution 1 — Neural Net + GA** (neuroevolution): population of MLPs, tournament
  selection, uniform crossover, Gaussian mutation, elitism. Opens by default.
- **Solution 2 — DQN**: Q network with hand-written backprop + Adam, replay buffer,
  target network, epsilon-greedy. Fixed-pace demo car decoupled from training.
- **Pygame app** (desktop + browser via pygbag): top menu bar (GA / DQN / About /
  Settings), live neuron + edge activation graph, hyperparameter sliders + retrain,
  fitness / reward curve.
- **Per-technique tutorials** opened from inside a simulation, with Back and a
  Download PDF button. Each starts with the raycast sensor model.
- **i18n**: English (default) and Portuguese (pt-BR), switchable in Settings,
  persisted (localStorage on web, `~/.avs_settings.json` on desktop).
- **Tooling**: `scripts/build_web.sh` + `serve_web.sh` (pygbag static bundle with a
  custom loader), `scripts/build_exe.py` (PyInstaller), `scripts/record_media.py`
  (Playwright screenshots/GIFs).
- **CI**: `ci.yml` (ruff + pytest incl. slow GA/DQN learning tests + headless screen
  smoke), `release.yml` (Windows/macOS/Linux executables on tag `v*`),
  `deploy-pages.yml` (GitHub Pages).

[Unreleased]: https://github.com/OWNER/autonomous_vehicle_simulator/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/OWNER/autonomous_vehicle_simulator/releases/tag/v0.1.0
