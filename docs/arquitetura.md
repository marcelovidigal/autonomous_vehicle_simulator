# Arquitetura — Autonomous Vehicle Simulator

> Documentos relacionados: [Plano de trabalho](plano-de-trabalho.md) · [PRD](prd.md)
> Status: rascunho · Data: 2026-09-06

## 1. Princípios

1. **Núcleo sem UI.** `core/` não importa `pygame`, `fastapi`, nem nada de I/O gráfico.
   Só `numpy` + stdlib. Assim roda em CPython, em Pyodide/WASM e num servidor sem mudança.
2. **Determinístico.** Todo aleatório passa por um `numpy.random.Generator` semeado. Mesma
   config + mesma seed ⇒ mesma curva de fitness.
3. **Treino incremental.** O `Trainer` avança em passos pequenos chamados pelo loop; nunca
   bloqueia, nunca cria thread. Compatível com o loop `async` do `pygbag`.
4. **Dados simples.** `dataclasses` e `dict`/`list`/`ndarray`. Serialização = JSON.
5. **Um código, dois alvos.** O app Pygame roda no desktop e, via `pygbag`, no navegador.

## 2. Visão de componentes

```mermaid
flowchart TB
  subgraph core["core/ (numpy + stdlib, sem UI)"]
    cfg[config.py<br/>Config + DQNCfg]
    geo[geometry.py<br/>raycast, offset, polígono]
    trk[track.py<br/>bordas, sensores, colisão, progresso]
    sim[simulation.py<br/>step_car · simulate · simulate_population]
    net[network.py<br/>MLP NumPy + ativações]
    evo[evolution.py<br/>GA]
    trn[trainer.py<br/>treino incremental do GA]
    dqn[dqn.py<br/>QNet + Replay + DQNTrainer]
  end

  subgraph appA["app/ — Pygame (desktop + pygbag)"]
    loop[main.py<br/>roteador · loop async]
    gas[gascreen.py<br/>Solução 1 · TELA PADRÃO]
    dqs[dqnscreen.py<br/>Solução 2]
    doc[docscreen.py<br/>tutorial_ga / tutorial_dqn / sobre · PDF]
    shared[scene · netview · plot · widgets · camera · chrome (menu bar)]
  end

  cfg --> trn & dqn
  geo --> trk
  trk --> sim & dqn
  net --> sim
  sim --> evo --> trn
  trn --> gas
  dqn --> dqs
  loop --> gas & dqs & doc
  gas & dqs --> shared
  gas -. Tutorial .-> doc
  dqs -. Tutorial .-> doc
```

## 3. Camadas e responsabilidades

| Camada | Pasta | Depende de | Responsabilidade |
|---|---|---|---|
| Núcleo | `src/core/` | `numpy`, stdlib | Simulação, sensores, e **dois** treinos: `trainer.py` (GA) e `dqn.py` (DQN). Puro, testável, determinístico. |
| Pistas | `src/tracks/` | — | Arquivos JSON de pista. |
| App | `app/` | `core`, `pygame-ce` | Roteador (`main.py`, abre em `gascreen`) + telas `gascreen` · `dqnscreen` · `docscreen`. Barra de menu superior (`chrome.MenuBar`). Sem regra de negócio. |
| Build web | `web/` | `pygbag`, (`reportlab`) | Empacota o app em site estático + `tutorial_ga`/`tutorial_dqn` em `.html` e `.pdf`. |
| Ferramentas | `tools/` | `numpy`; `reportlab` (`md2pdf`) | Autoria de pista, `md2html`, `md2pdf`. Nunca importado pelo app. |
| App B | `server/`, `frontend/` | `core`, `fastapi`; `PixiJS`, `D3` | Streaming de estado + render web dedicado. |

## 4. Modelos de dados

### 4.1 Configuração (`config.py`)

```python
@dataclass(frozen=True)
class NetworkCfg:
    hidden: int = 6              # 0 = linear
    activation: str = "tanh"     # "tanh" | "relu"

@dataclass(frozen=True)
class SensorCfg:
    count: int = 5
    fov_deg: float = 160.0
    range_u: float = 150.0

@dataclass(frozen=True)
class PhysicsCfg:
    v_max: float = 120.0
    accel: float = 80.0
    steer_deg_s: float = 160.0
    friction: float = 0.05
    dt: float = 1/60

@dataclass(frozen=True)
class GACfg:
    population: int = 40
    generations: int = 60
    mutation_rate: float = 0.15
    mutation_sigma: float = 0.2
    elitism: int = 2
    tournament_k: int = 3
    crossover: bool = True
    seed: int = 0

@dataclass(frozen=True)
class FitnessCfg:
    w_progress: float = 1.0
    w_speed: float = 0.1
    w_smooth: float = 0.05
    crash_penalty: float = 1.0

@dataclass(frozen=True)
class SimCfg:
    max_steps: int = 1500
    track: str = "circuito_1"

@dataclass(frozen=True)
class Config:
    network: NetworkCfg = NetworkCfg()
    sensors: SensorCfg = SensorCfg()
    physics: PhysicsCfg = PhysicsCfg()
    ga: GACfg = GACfg()
    fitness: FitnessCfg = FitnessCfg()
    sim: SimCfg = SimCfg()
    # to_dict()/from_dict() -> JSON
```

### 4.2 Pista (JSON) — `tracks/*.json`

```json
{
  "name": "circuito_1",
  "width_u": 34.0,
  "centerline": [[x0, y0], [x1, y1], "..."],
  "closed": true,
  "start": { "pos": [x, y], "heading_deg": 0.0 },
  "meta": { "author": "", "notes": "" }
}
```

Derivado em carga (`Track.load`): bordas interna/externa (offset ± `width_u/2` da
`centerline` reamostrada por spline), lista de **checkpoints** (segmentos perpendiculares à
central, igualmente espaçados) e uma *bounding grid* para acelerar o raycast.

### 4.3 Genoma e rede

- **Genoma** = `np.ndarray` 1-D `float32` (vetor plano de todos os pesos+bias).
- `Network.n_params(cfg, n_inputs)` → tamanho esperado.
- `Network.from_flat(vec, shapes)` / `Network.to_flat()`.

### 4.4 Estado de simulação

```python
@dataclass
class CarState:
    pos: np.ndarray          # (2,)
    heading: float           # rad
    speed: float
    alive: bool = True
    progress: float = 0.0    # dist. acumulada ao longo da central
    checkpoint_idx: int = 0
    laps: int = 0
    steps: int = 0

@dataclass
class ActivationSnapshot:
    inputs: np.ndarray       # (n_inputs,)  leituras dos sensores [+ speed]
    hidden: np.ndarray       # (hidden,)
    outputs: np.ndarray      # (2,)  steer, throttle  em [-1, 1]

@dataclass
class GenerationStats:
    gen: int
    best_fitness: float
    mean_fitness: float
    alive: int
    best_genome: np.ndarray
```

## 5. Interfaces principais (contratos)

```python
# geometry.py  — tudo vetorizável, sem estado
def ray_segments_hit(origin, direction, seg_a, seg_b) -> float          # distância ou +inf
def point_in_polygon(p, poly) -> bool
def offset_polyline(pts, dist) -> np.ndarray

# track.py
class Track:
    @classmethod
    def load(cls, name: str) -> "Track"
    def sensor_distances(self, pos, heading, cfg: SensorCfg) -> np.ndarray  # normalizado [0,1]
    def collides(self, pos, radius: float) -> bool
    def advance_progress(self, state: CarState) -> None                    # atualiza checkpoint/lap
    def start_state(self) -> CarState

# car.py
def step_car(state: CarState, steer: float, throttle: float, cfg: PhysicsCfg) -> None

# network.py
class Network:
    def __init__(self, n_inputs: int, cfg: NetworkCfg)
    def forward(self, x: np.ndarray) -> tuple[np.ndarray, ActivationSnapshot]
    def to_flat(self) -> np.ndarray
    def load_flat(self, vec: np.ndarray) -> None
    @staticmethod
    def n_params(n_inputs: int, cfg: NetworkCfg) -> int

# simulation.py
def evaluate(genomes: np.ndarray,        # (P, n_params)
             track: Track, cfg: Config,
             rng: np.random.Generator,
             record_focus: int | None = None
             ) -> tuple[np.ndarray, list[CarState], list[ActivationSnapshot] | None]
# retorna fitness (P,), estados finais, e o trace do carro em foco (para replay/erro)

# evolution.py
def next_population(genomes, fitness, cfg: GACfg, rng) -> np.ndarray

# trainer.py
class Trainer:
    def __init__(self, cfg: Config)
    state: Literal["idle","running","paused","done"]
    stats: GenerationStats | None
    def reset(self, cfg: Config | None = None) -> None      # re-treinar do zero
    def step(self, budget_ms: float = 6.0) -> None          # avança o que couber no orçamento
    def advance_generation(self) -> None                    # roda 1 geração inteira
    def best_genome(self) -> np.ndarray | None
    def save(self) -> dict                                  # genoma + config -> JSON
    def load(self, blob: dict) -> None
```

`Trainer.step()` é o coração do "sem threads": consome um **orçamento de milissegundos**,
avaliando genomas/gerações enquanto houver tempo, e devolve o controle ao loop.

## 6. Fluxos de execução

### 6.1 Loop do app (Trilha A, `main.py`)

```mermaid
sequenceDiagram
  participant Loop as main (async)
  participant Panel
  participant Trainer
  participant Sim as simulation
  participant Scene
  participant NetView

  loop cada frame (~1/60 s)
    Loop->>Panel: processa input (sliders, botões)
    Panel-->>Trainer: reset()/advance_generation()/flags
    alt treinando
      Loop->>Trainer: step(budget_ms = 4..8)
      Trainer->>Sim: evaluate(sub-lote de genomas)
      Sim-->>Trainer: fitness parcial
    end
    Loop->>Sim: 1 passo do carro em foco (replay do melhor)
    Sim-->>NetView: ActivationSnapshot
    Loop->>Scene: desenha pista, carros, sensores, HUD
    Loop->>NetView: desenha grafo (cor = ativação)
    Loop->>Loop: await asyncio.sleep(0)
  end
```

Modos: **ocioso** (só render), **treinando** (chama `step`), **acelerar** (chama
`advance_generation` várias vezes por frame, sem animar a população), **pausado**.

### 6.2 Um passo de simulação por carro

`sensor_distances` → `Network.forward` → `(steer, throttle)` → `step_car` →
`track.collides?` → `track.advance_progress` → acumula fitness parcial → `steps++`.
Encerra o carro em colisão, saída de pista, `laps ≥ 1` (volta completa) ou `steps ≥ max_steps`.

### 6.3 Fitness

```
fitness = w_progress * progress_norm
        + w_speed    * mean_speed_norm
        - w_smooth   * mean_abs_steer_delta
        - crash_penalty * (1 if crashed else 0)
        + lap_bonus  * laps
```

`progress_norm` = distância ao longo da central / comprimento da central (via checkpoints,
robusto a "cortar caminho").

## 7. Rede neural (detalhe)

- Entradas: `sensors.count` distâncias normalizadas `[0,1]` (+ opcional `speed/v_max`).
- Camadas: `Linear(n_in, hidden) → act → Linear(hidden, 2) → tanh`. Se `hidden == 0`:
  `Linear(n_in, 2) → tanh`.
- `forward` guarda `inputs`, `hidden`, `outputs` no `ActivationSnapshot` (custo desprezível).
- Pesos inicializados `N(0, 1/sqrt(fan_in))` com o `Generator` semeado.
- `to_flat`/`load_flat`: concatena/reparte `W1,b1,W2,b2` numa ordem fixa documentada.

## 8. Algoritmo genético (detalhe)

1. **Elitismo**: copia os `elitism` melhores sem alteração.
2. **Seleção**: torneio de tamanho `tournament_k`.
3. **Crossover** (se ligado): uniforme por gene entre 2 pais.
4. **Mutação**: com prob. `mutation_rate` por gene, soma `N(0, mutation_sigma)`.
5. Repete até completar `population`.

Sem *speciation* (isso seria NEAT — fora do MVP). Topologia é fixa ⇒ genomas sempre têm o
mesmo tamanho ⇒ crossover trivial.

## 8b. DQN (detalhe) — `core/dqn.py`

Mesma interface pública do `Trainer` (`state` · `step(budget_ms)` · `reset` · `curve` ·
`best_reward` · `best_laps`), então o app trata os dois treinos igual.

- **Ações**: `STEER_ACTIONS = [-1, -0.5, 0, 0.5, 1]`, acelerador fixo (`THROTTLE`).
- **Recompensa/passo**: `R_PROGRESS · Δprogresso − R_CRASH·[bateu]`. Densa (guia o gradiente).
- **`QNet`**: `x → ReLU(W1x+b1) → (W2h+b2)` (Q linear, 5 saídas). `train_step`: perda MSE
  sobre `Q(s,a)`, gradiente **recortado em [-1,1]** (estilo Huber), **Adam** manual.
- **Alvo**: `y = r + γ · maxₐ' Q_target(s') · (1 − done)`; `Q_target` sincroniza a cada
  `target_sync` passos.
- **Exploração**: ε-greedy, ε de `eps_start`→`eps_end` linear em `eps_decay_steps`.
- **`Replay`**: buffer circular em arrays NumPy; treino começa após `warmup` transições.
- Loop por `step`: `env_step` (ε-greedy → push) e, se `buf.n ≥ warmup`, um `_learn` por passo.
- **Determinístico** dada `DQNCfg.seed` (um `Generator` para init, exploração e amostragem).
- **`demo_step()`** — avança 1 passo um "carro-demonstração" independente com a política
  gulosa atual. A tela chama `demo_step` a ritmo fixo (1×/frame; 4×/frame no turbo)
  **desacoplado** de `step`, para a visualização não ficar acelerada quando o treino corre
  muito rápido depois de aprender.

## 9. Renderização

### 9.1 Camadas de desenho (App A)
1. Fundo + faixa da pista (polígono entre bordas) + linha central tracejada.
2. Checkpoints (sutis).
3. Carros "fantasma" (população, translúcidos) — opcional/limitado.
4. Carro em foco (opaco) + raios de sensor + pontos de interseção.
5. HUD (geração, fitness, vivos, voltas, FPS).
6. `netview` (painel lateral).
7. `plot` fitness × geração (painel lateral).
8. `panel` (sliders/botões) — `pygame_gui` ou widgets próprios.

### 9.2 `netview` — mapeamento
- Colunas x: entrada, oculta, saída. Nós distribuídos em y.
- **Nó** = `diverging(activation)` — `-1→azul`, `0→cinza`, `+1→vermelho`.
- **Aresta destacada pelo sinal** que a percorre: `s[dst,src] = w[dst,src] * activation[src]`.
  Normaliza por `max|s|` do frame; abaixo de 5% desenha fininha e escura, acima disso
  interpola da cor de fundo para vermelho (sinal +) / azul (sinal −) e engrossa
  (`1 + round(2·intensidade)` px). Assim vê-se *quais ligações estão conduzindo sinal*.
- Recebe a `MLP` do melhor genoma (pesos) + o `Activation` por frame.
- Rótulos: `S0..Sn` + `vel` nas entradas; `dir`/`acel` nas saídas.
- Atualiza todo frame com o `Activation` do carro em foco.

## 10. Determinismo e RNG

- Um `np.random.Generator` criado de `ga.seed` no `Trainer.reset`.
- Ordem de consumo fixa: init de pesos → seleção → crossover → mutação, por geração.
- Simulação é puramente determinística dado o genoma e a pista (sem ruído).
- Teste de regressão: hash da sequência de `best_fitness` para uma config canônica.

## 11. Estratégia de testes

| Alvo | Teste |
|---|---|
| `geometry` | raycast contra segmentos conhecidos; ponto-em-polígono; offset de reta/círculo |
| `track` | carga de JSON; `sensor_distances` em poses conhecidas; colisão dentro/fora |
| `car` | integração cinemática (linha reta, círculo com esterço constante) |
| `network` | `n_params` bate com `to_flat().size`; `forward` determinístico; `hidden==0` |
| `evolution` | elitismo preserva os melhores; tamanho da população; determinismo por seed |
| `simulation` | controlador heurístico completa 1 volta; `evaluate` reproduz fitness com mesma seed |
| `trainer` | `step` respeita orçamento; `save`/`load` round-trip; `reset` volta ao estado inicial |
| regressão | curva de `best_fitness` (config canônica) estável entre commits |

Ferramentas: `pytest`, `ruff`, `mypy` opcional. CI roda em CPython (o núcleo não precisa de browser).

## 12. Orçamento de performance

| Item | Alvo |
|---|---|
| `evaluate` de 40 genomas × 1500 passos (desktop) | ≤ ~150 ms |
| Mesmo, no browser (Pyodide) | ≤ ~600 ms (só no modo "acelerar") |
| Frame de render (desktop / web) | ≤ 16 ms / ≤ 40 ms |
| `Trainer.step` por frame durante treino animado | ≤ 8 ms |
| Download inicial (web) | ≤ ~15 MB |

Táticas: vetorizar o raycast (todos os sensores de um carro de uma vez; se couber, todos os
carros num tensor), *bounding grid* de segmentos da pista, limitar fantasmas desenhados,
"modo rápido" que pula o render durante o treino.

## 13. Empacotamento e deploy

### 13.1 Trilha A — site estático via `pygbag`
```
bash scripts/build_web.sh
  # monta um stage limpo (só main.py + src/) e roda:
  #   python -m pygbag --build --disable-sound-format-error stage/main.py
  # saída: build/web/  (index.html + autonomous_vehicle_simulator.apk ~24 KB + favicon;
  #        runtime Python/NumPy/pygame vem do CDN do pygbag no 1º load)
```
- **Preview local:** `bash scripts/serve_web.sh` → **`http://127.0.0.1:8080`** (estático, sem
  COOP/COEP; igual ao GitHub Pages). Regras que descobrimos na marra:
  - a origem **não** pode ser `localhost` — ali o `pythons.js` entra em "modo dev" e busca
    os *wheels* (NumPy) num `localhost:<porta>/cdn/` inexistente → trava;
  - **não** usar o servidor do próprio pygbag (`python -m pygbag`) — ele força
    `Cross-Origin-Embedder-Policy: require-corp` com um header CORP quebrado (bug 0.9.3) →
    `ERR_BLOCKED_BY_RESPONSE…ByCoep`;
  - em `127.0.0.1` + servidor estático **sem COEP**, o pygbag baixa runtime + NumPy do CDN
    público (`Access-Control-Allow-Origin: *`) — cross-origin normal, sem COEP travando.
- **Empacotamento de pacotes:** o pygbag lê os `import` do **`main.py` raiz** (não varre a
  árvore). Por isso `main.py` importa `pygame`/`numpy` no topo + bloco PEP 723
  `dependencies = ["numpy", "pygame-ce"]`. Sem isso o `pygame` não entra no bundle
  (`AttributeError: module 'pygame' has no attribute 'init'`).
- **CSS da tela de carregamento:** `#transfer` usa `display:flex`; foi preciso
  `#transfer[hidden]{display:none!important}` para o runtime conseguir escondê-la.
- **`app/main.py`:** pinta + `await asyncio.sleep(0)` algumas vezes logo após `set_mode`, e
  só começa a treinar depois de ~20 frames — garante o 1º paint no navegador.
- Verificado por Playwright (Chromium headless): app carrega, canvas 1300×864, tela GA
  renderiza, treino roda.
- **GitHub Pages**: [`.github/workflows/deploy-pages.yml`](../.github/workflows/deploy-pages.yml)
  roda o build e publica `build/web/` (Source = GitHub Actions). *Single-thread*;
  não precisa de headers COOP/COEP. O domínio publicado não é `localhost`, então funciona.
- **Hugging Face Spaces**: Space **Static** com o conteúdo de `build/web/` na raiz
  (`scripts/deploy_hf.sh`).
- Requisitos do código para `pygbag`: `async def main()`, `await asyncio.sleep(0)` por
  frame, sem `threading`, sem libs C fora do Pyodide (só `numpy`).
- **Portas**: o app **não abre nenhuma porta**. Só o preview `python -m pygbag ...` sobe
  um servidor — use `--port 8080` (o padrão `8000` pode colidir com outro serviço).
- Extras gerados junto, um por técnica: `tutorial_ga.html`/`tutorial_dqn.html`
  (`tools/md2html.py`) e `.pdf` (`tools/md2pdf.py`, requer `reportlab`).

### 13.2 Desktop
`uv run python main.py` (ou `pyinstaller` para sala de aula).

### 13.3 Trilha B — FastAPI + PixiJS (opcional)
- `server/`: `uvicorn server.main:app`; endpoint `/ws` faz *streaming* do estado.
- `frontend/`: build estático (Vite) hospedado à parte (Vercel/Netlify).
- Deploy do backend: Fly.io/Render/HF Space Docker.

## 14. Contrato de mensagens — Trilha B (`server/schema.py`)

```jsonc
// servidor -> cliente, ~30 Hz
{
  "type": "frame",
  "gen": 12, "best_fitness": 3.4, "mean_fitness": 1.1, "alive": 27,
  "focus": {
    "pos": [x, y], "heading": 1.23, "speed": 88.0,
    "rays": [[x1,y1],[x2,y2], "..."],
    "act": { "inputs": [/*...*/], "hidden": [/*...*/], "outputs": [s, t] }
  },
  "ghosts": [[x,y,heading], "..."],          // opcional, limitado
  "fitness_curve": [/* por geração */]
}

// cliente -> servidor
{ "type": "set_params", "config": { /* Config.to_dict() parcial */ } }
{ "type": "command", "name": "apply" | "pause" | "resume" | "step_gen" | "reset" | "fast" }
{ "type": "load_genome", "blob": { /* save() */ } }
```

O mesmo `Trainer` do núcleo alimenta os dois fronts; só muda quem chama `step()` e quem
desenha.

## 15. Decisões (resumo)

| # | Decisão | Alternativa recusada | Motivo |
|---|---|---|---|
| D1 | Neuroevolução (GA) sobre MLP rasa | Backprop supervisionado; RL | Sem dataset, sem gradiente; melhora visível por geração. GA treina pesos, rede continua rasa. |
| D2 | Núcleo só `numpy`+stdlib | Usar `shapely`/`scipy` no núcleo | Compatibilidade com Pyodide/WASM; geometria cabe em ~100 linhas. |
| D3 | Trilha A = Pygame + `pygbag` | Front JS desde já; Streamlit/Gradio | 100% Python, um código, publicação estática grátis, controle total do canvas (inclui o grafo de neurônios). |
| D4 | Treino incremental (`Trainer.step`) | Thread de treino | Sem threads no WASM; mantém a UI fluida. |
| D5 | Topologia fixa | NEAT | Visualização estável dos neurônios; crossover trivial. |
| D6 | Gráfico de fitness com primitivas do Pygame | `matplotlib` no bundle | Reduz o download WASM. |
| D7 | Trilha B só se necessário | Construir cliente-servidor já | Evita duas linguagens e custo de servidor antes de haver demanda. |
