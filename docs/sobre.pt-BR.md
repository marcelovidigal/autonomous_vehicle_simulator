# Sobre o projeto

**Simulador de Veiculo Autonomo — Redes Neurais com Algoritmo Genetico e DQN.** Projeto
**didatico**: uma rede neural aprende a dirigir um carro num circuito sinuoso usando apenas
sensores de distancia, e voce ve ao mesmo tempo o carro na pista e os neuronios (e as
ligacoes) sendo ativados. Da para mexer nos hiperparametros e re-treinar, no desktop ou no
navegador, a partir do mesmo codigo Python.

O app abre na **Solucao 1 (Rede Neural + AG)**; a barra de menu superior alterna para
**DQN** e **Sobre**. Cada simulacao tem seu **Tutorial** proprio.

## Objetivos

- Tornar tangivel a relacao **sensores -> rede -> acao** e **hiperparametro -> aprendizado**.
- Mostrar, lado a lado, **duas formas de treinar a mesma rede** para o mesmo problema:
  - **Solucao 1 — Algoritmo Genetico (neuroevolucao):** sem gradiente; uma populacao de
    redes e selecionada por desempenho.
  - **Solucao 2 — DQN (Deep Q-Network):** por gradiente; uma rede Q aprende de recompensa
    com *replay buffer* e rede-alvo.
- Rodar **sem instalar nada** (navegador) e tambem offline (desktop).
- Codigo pequeno e legivel, aproveitavel como material de estudo.

## Como funciona (resumo)

- **Sensores:** 5 raios (raycasts) partem do carro em leque e medem a distancia ate a
  parede. E o "olhar" do carro — barato e determinista, no lugar de uma camera real.
- **Rede:** MLP rasa em NumPy. Entradas = distancias + velocidade; saidas = direcao e
  acelerador (Solucao 1) ou valor Q de cada acao discreta (Solucao 2).
- **Pista:** linha central (spline) + largura -> bordas; colisao e progresso medidos pela
  distancia a central.
- **Treino sem thread:** avanca dentro do loop de render, em pedacos — funciona igual no
  navegador (WebAssembly via pygbag).

## Pilha de ferramentas

- **Python 3.11+**, **NumPy** (nucleo), **Pygame-CE** (interface), **pygbag** (WebAssembly).
- **uv** (ambiente), **pytest** + **ruff** (testes e lint), **reportlab** (tutorial em PDF).
- Ficou de fora de proposito: shapely/scipy (geometria propria em NumPy roda em Pyodide),
  PyTorch/TensorFlow (a rede e pequena e o DQN tem backprop manual), matplotlib.

## Estrutura

```
src/core/     config, geometry, track, sensors, network, simulation, evolution, trainer, dqn
src/app/      Pygame: roteador (main, abre em 'ga') + gascreen, dqnscreen, docscreen,
              settingsscreen + chrome (barra de menu), scene, netview, plot, widgets,
              camera, docview, i18n, links
src/tracks/   circuito_1..3 (JSON)
docs/         tutorial_ga.md, tutorial_dqn.md, sobre.md (+ variantes .pt-BR), plano, prd, arquitetura
scripts/      build_web.sh, serve_web.sh, deploy_hf.sh, build_exe.py, train_headless.py
```

## Como rodar

- **Desktop:** `uv pip install -e ".[dev]"` e `uv run python main.py`.
- **Web (local):** `uv pip install -e ".[web,docs]"`, `bash scripts/serve_web.sh`,
  abra `http://127.0.0.1:8080` (**não** `localhost`).
- **Executável nativo:** `uv pip install -e ".[build]"` e `python scripts/build_exe.py`
  (gera para o SO atual; o CI gera Windows/macOS/Linux).
- **Publicar:** GitHub Pages (workflow pronto) ou Hugging Face Spaces (SDK Static).
  Ver `docs/publicacao.md`.

## Licenca

Codigo aberto (MIT) — material didatico.
