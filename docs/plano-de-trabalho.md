# Plano de Trabalho — Autonomous Vehicle Simulator

> Documentos relacionados: [PRD](prd.md) · [Arquitetura](arquitetura.md) · [Log de prompts](../prompts/prompts_gerais_v1.md)
>
> **Status (2026-09-06):** app Pygame multi-tela — **barra de menu superior** (abas
> Rede Neural + AG · DQN · Sobre); abre por padrão na **Solução 1 (GA)**; cada simulação
> tem seu **Tutorial exclusivo** (RNA+AG / DQN), acionado de dentro dela, com **Voltar** e
> **Baixar PDF**. Núcleo GA + **Solução 2 DQN** (backprop manual). 59+ testes verdes (GA e
> DQN aprendendo uma volta), `ruff` limpo. Falta: verificação visual no navegador e
> publicação — ver [plano de publicação](publicacao.md).
>
> **Sobre a porta 8000:** o projeto **não abre nenhuma porta**. Só o comando opcional de
> preview `python -m pygbag ...` sobe um servidor — documentado agora com `--port 8080`
> para não colidir com um serviço já rodando em 8000.

## 1. Objetivo

Construir, em **Python**, uma rede neural artificial que aprende a dirigir um carro num
circuito sinuoso (tipo pista de corrida / autódromo) **sem sair da pista e sem colidir**,
guiada por **sensores ao redor do carro**. O foco é **didático**. A aplicação deve:

1. mostrar visualmente a pista e o carro trafegando nela, em tempo real;
2. mostrar, em paralelo, os **neurônios da rede sendo ativados**;
3. oferecer uma **interface para alterar hiperparâmetros e re-treinar** a rede.

**Entrega alvo: aplicação web**, publicável como site estático e também executável no
desktop a partir do mesmo código.

## 2. Decisões de arquitetura (o essencial)

| Tema | Decisão | Por quê |
|---|---|---|
| Paradigma de aprendizado | **Neuroevolução** (algoritmo genético sobre uma MLP rasa) por padrão; modo *behavioral cloning* opcional | Sem dados rotulados, sem gradiente, sem framework pesado. Treino = simulação; a melhora aparece **por geração** (didático). O GA treina os **pesos**; a rede continua sendo uma MLP rasa. |
| Rede | **MLP rasa em NumPy**: ~5–7 entradas → 1 camada oculta (6–8 neurônios, `tanh`) → 2 saídas (`tanh`: direção, acelerar/frear). Camada oculta é configurável (inclusive 0 = linear) | Pequena o suficiente para desenhar **todos** os neurônios. Forward em NumPy = snapshot de ativações trivial. Pesos como vetor plano = fácil para o GA. |
| Física | 2D top-down, modelo cinemático simples (posição, ângulo, velocidade; atrito; `dt` fixo) | Realismo não é o objetivo; clareza e estabilidade sim. |
| Pista | Linha central (spline Catmull-Rom sobre pontos de controle) + largura → bordas interna/externa como polilinhas/segmentos; *checkpoints* ao longo da central | Colisão = ponto/círculo vs. segmentos de borda. Sensor = raycast vs. segmentos. *Checkpoints* dão progresso (fitness) e contagem de voltas. |
| Sensores | 5–7 *raycasts* em leque (ex.: -90°…+90°), distância normalizada até a borda; opcional: velocidade atual como entrada extra | Entrada compacta e suficiente para manter o carro na pista. |
| Fitness | progresso na central (checkpoints) + bônus leve de velocidade − penalidade por colisão/sair da pista − penalidade leve por zigue-zague | Recompensa "andar longe sem bater". |
| **Front-end / entrega** | **Trilha A (MVP):** um app **Pygame-CE** único, empacotado para a web com **`pygbag`** (WebAssembly) e também rodável no desktop. **Trilha B (evolução):** **núcleo Python via FastAPI + WebSocket** com front **PixiJS** (pista) + **D3** (neurônios/gráficos). | Trilha A = 100% Python, um código só, publicação estática grátis, todos os elementos visuais sob nosso controle. Trilha B = melhor UX/escala quando fizer sentido, reusando o mesmo núcleo. |
| Separação | **Núcleo (`sim` + `nn` + `train`) sem nenhuma dependência de UI**; cada front-end é um cliente do núcleo | Permite Trilha A hoje e Trilha B depois sem reescrever lógica. |
| Concorrência | Sem threads reais: **treino cooperativo** (N passos de GA por frame) dentro do loop `async` | Threads não portam bem para WASM; o loop `async` do `pygbag` exige `await asyncio.sleep(0)` a cada frame. |

### Alternativas de treino (não são o padrão do MVP)
- **Behavioral cloning** (a MLP imita um controlador heurístico/PID): loop de treino mais
  simples e o mais *lean* se aceitarmos escrever a heurística-professora. Fica como **modo
  opcional** no app (didático: comparar "imitar" vs. "evoluir").
- **Aprendizado por reforço (DQN/PPO + Gymnasium)**: mais pesado, instável e lento —
  ruim para didática. Extensão opcional.
- **NEAT (`neat-python`)**: topologia evolui junto; ótimo didaticamente, mas complica a
  visualização fixa dos neurônios. Fase opcional.

## 3. Ferramentas e bibliotecas

### Núcleo (obrigatório — compatível com Pyodide/WASM)
- **Python 3.12+**
- **numpy** — rede neural e matemática vetorizada (roda em Pyodide)
- **stdlib** (`dataclasses`, `json`, `math`, `asyncio`) — configuração, persistência de genomas, loop

> Geometria **sem `shapely`** na Trilha A: `shapely` (libgeos) não está no Pyodide padrão.
> As operações necessárias (interseção raio–segmento, ponto-em-polígono, *offset* de
> polilinha) são ~100 linhas de NumPy. `shapely`/`scipy` ficam disponíveis **apenas** para
> ferramentas *offline* de autoria de pista (rodam no desktop, geram JSON).

### Front-end — Trilha A (MVP, web + desktop)
- **pygame-ce** — render 2D (pista, carro, sensores, **grafo de neurônios**, HUD, gráfico de fitness), tudo desenhado com `pygame.draw`
- **pygbag** — empacota o app em WebAssembly e gera o site estático (`index.html` + assets)
- Painel de hiperparâmetros: **`pygame_gui`** (funciona sob `pygbag`) **ou** widgets próprios desenhados no canvas (mais leves; decidir na Fase 6)
- Gráfico fitness × geração: **desenhado com primitivas do Pygame** (evita `matplotlib` no bundle WASM)

### Front-end — Trilha B (evolução, "produto web")
- **fastapi** + **uvicorn** + **websockets** — servidor que roda o núcleo e faz *streaming* do estado por frame
- **pydantic** — schema das mensagens (contrato servidor↔cliente)
- **PixiJS** (WebGL 2D) — pista, carro, sensores, muitos carros/fantasmas
- **D3.js** — grafo de neurônios (node-link, cor por ativação) e curva de fitness
- HTML/JS simples ou **React** para o painel

### Desenvolvimento
- **uv** — ambiente e dependências + **pyproject.toml**
- **pytest** — testes do núcleo (rodam em CPython normal)
- **ruff** (+ **black**) — lint/format; **mypy** *(opcional)* — tipagem

### Opcional / futuro
- **neat-python** (rota NEAT) · **torch** / **stable-baselines3 + gymnasium** (rota RL)
- **matplotlib**, **shapely**, **scipy** — apenas em *scripts* offline (autoria de pista, análises)
- **pyinstaller** — executável desktop para distribuição em sala de aula

## 4. Estrutura do projeto

```
autonomous_vehicle_simulator/
  main.py                     # ponto de entrada (desktop + pygbag)
  docs/                       # plano, PRD, arquitetura, publicação, CHANGELOG
                              #   traceback_v1.md — log local de entregas (git-ignored)
  prompts/                    # prompts_gerais_v1.md — log local de prompts (git-ignored)
  src/
    __init__.py
    core/                     # NÚCLEO — só numpy + stdlib (roda em Pyodide/WASM)
      config.py               # dataclasses de hiperparâmetros (Config + DQNCfg)
      geometry.py             # raycast, ponto-em-polígono, offset de polilinha (NumPy puro)
      track.py                # pista, bordas, sensores, colisão, progresso, load JSON
      sensors.py              # fachada de leitura dos sensores
      network.py              # MLP NumPy (Solução 1): forward, params planos, ativações
      simulation.py           # 1 carro (legível) + população vetorizada; fitness; step_car
      evolution.py            # GA: seleção, crossover, mutação, elitismo
      trainer.py              # treino incremental do GA (step-a-step, sem threads)
      dqn.py                  # Solução 2: QNet (backprop + Adam manual) + Replay + DQNTrainer
    app/                      # Pygame (desktop + pygbag)
      main.py                 # ROTEADOR de telas; abre em 'ga'; loop async
      gascreen.py             # tela da Solução 1 (GA) — padrão
      dqnscreen.py            # tela da Solução 2 (DQN) — carro-demo a ritmo fixo
      docscreen.py            # Tutorial da técnica / Sobre (docview + Voltar + Baixar PDF)
      chrome.py               # barra de menu superior (abas) + barra de Voltar
      scene.py · netview.py · plot.py · widgets.py · camera.py · docview.py · theme.py
    tracks/                   # circuito_1..3 (JSON)
  docs/tutorial_ga.md · docs/tutorial_dqn.md · docs/sobre.md   # conteúdo das telas
  tools/make_track.py · md2html.py · md2pdf.py
  scripts/train_headless.py   # treino GA sem UI (valida o núcleo)
  scripts/build_web.sh · serve_web.sh · deploy_hf.sh   # pygbag + preview + deploy HF
  scripts/build_exe.py · record_media.py · assets/     # PyInstaller · captura Playwright
  .github/workflows/ci.yml · release.yml · deploy-pages.yml
  tests/                      # config, geometry, track, network, evolution, simulation,
                              # trainer, learning, dqn, tutorial
  pyproject.toml · README.md
```

Núcleo alvo: **< ~1500 linhas** — atual **~1050** (GA 831 + DQN ~220).

## 5. Requisitos

### Funcionais
- **RF1** — o carro se mantém na pista sob controle da rede, a partir das leituras dos sensores.
- **RF2** — o sistema detecta saída da pista / colisão e encerra a corrida daquele carro.
- **RF3** — sensores configuráveis (quantidade, abertura/FOV, alcance); leituras normalizadas.
- **RF4** — treinar a rede até completar **uma volta inteira** em ≥1 pista sem sair nem colidir.
- **RF5** — visualizar pista, carro e raios dos sensores em tempo real.
- **RF6** — visualizar a rede em tempo real (carro em foco): **ativação por neurônio** (cor do nó) **e sinal por aresta** (peso × ativação da origem — largura/cor).
- **RF6b** — **Tutorial por técnica**, acionado **de dentro da simulação**; um **Voltar**
  retorna àquela simulação. Conteúdo **exclusivo da técnica** (`tutorial_ga` só trata de
  RNA+AG; `tutorial_dqn` só de DQN), com a descrição dos **sensores por raycast** no início
  e detalhada no passo a passo do código. Botão **Baixar PDF** (browser: imprimir; desktop:
  `reportlab`).
- **RF7** — editar hiperparâmetros (GA: rede/física/GA/fitness; DQN: ocultos, lr, γ, ε).
- **RF8** — re-treinar sob demanda; resetar para *defaults*; acompanhar o progresso.
- **RF9** — salvar rede/config em JSON (GA e DQN).
- **RF10** — ≥2 pistas embutidas, com troca em tempo de execução (nas duas soluções).
- **RF11** — **barra de menu superior** com abas: Rede Neural + AG, DQN, Sobre. O app
  **abre na Solução 1 (GA)** por padrão.
- **RF12** — **Solução 2 (DQN)**: rede Q por gradiente (backprop manual) + *replay* +
  rede-alvo + ε-greedy; treina até completar uma volta; **carro-demonstração a ritmo fixo**
  (visualização não acelera após o treino) + grafo da rede Q + curva de recompensa/episódio.
- **RF13** — **tela Sobre** com a descrição do projeto (`docs/sobre.md`).

### Não funcionais
- **RNF1** — roda no **navegador** via WebAssembly (`pygbag`) **e** no desktop, do mesmo código.
- **RNF2** — **publicável como site estático** (GitHub Pages ou Hugging Face Spaces *Static*), sem backend.
- **RNF3** — visualização interativa a **≥30 fps** num notebook comum, somente CPU (desktop nativo; ~20–30 fps aceitável no browser).
- **RNF4** — poucas dependências; núcleo compatível com Pyodide (só `numpy` + stdlib).
- **RNF5** — **determinístico** dada uma *seed*.
- **RNF6** — núcleo **tipado e testado**; clareza didática acima de realismo/performance.

### Hiperparâmetros expostos na UI
- **Rede:** tamanho da camada oculta (0–16), função de ativação, nº de sensores, FOV, alcance.
- **Física:** velocidade máx., aceleração, taxa de esterço, atrito, `dt`.
- **GA:** tamanho da população, nº de gerações, taxa e força de mutação, elitismo, tamanho do torneio, crossover on/off, *seed*.
- **Fitness:** pesos de progresso × velocidade × suavidade; penalidade de colisão.
- **Simulação:** limite de passos por genoma (timeout), pista selecionada.
- **Botões:** Iniciar/Pausar · **Aplicar** (aplica os sliders + reinicia o treino; tecla `a`) · Reset · Avançar 1 geração · Salvar/Carregar melhor genoma · Acelerar (sim rápido, anima só o melhor).

## 6. Estratégia de front-end (web-first)

### Trilha A — MVP: Pygame-CE + `pygbag` (100% Python, web + desktop)
Um único app Pygame. No desktop roda direto; para a web, `pygbag` compila Python + NumPy +
Pygame para WebAssembly e gera um site **estático** (`index.html` + `.apk`/wasm). Requisitos
do modelo `pygbag`: `main` assíncrono e `await asyncio.sleep(0)` a cada frame; **sem
threads** (por isso o treino é incremental/cooperativo).

**Publicação — respostas diretas:**
- **GitHub Pages: sim.** A saída do `pygbag` é estática; funciona em modo *single-thread*
  (não depende de headers COOP/COEP nem de `SharedArrayBuffer`). Basta publicar a pasta de
  build (via GitHub Actions ou `docs/`). NumPy funciona no Pyodide.
- **Hugging Face Spaces: sim**, usando um Space do tipo **Static** (serve HTML/JS/wasm).
  Alternativamente um Space Docker/Gradio, mas o Static é o mais simples e gratuito.

**Grafo de neurônios + pista + carro estão garantidos na Trilha A?** **Sim.** Tudo é
desenhado por nós com `pygame.draw` sobre a mesma *surface*:
- pista, carro e raios de sensor → polígonos/linhas;
- **grafo de neurônios** → círculos posicionados em colunas (entrada · oculta · saída), com
  a **cor de cada nó vinda do vetor de ativação** que lemos após o `forward` da MLP em
  NumPy; arestas opcionais por peso.
O WASM não remove nenhuma primitiva de desenho — o que muda em relação ao desktop é só:
performance ~2–5× menor (irrelevante para MLP rasa + sim 2D), ausência de threads, e um
*download* inicial de alguns MB. Nenhum recurso visual é perdido.

### Trilha B — evolução: núcleo Python + FastAPI/WebSocket + PixiJS/D3
Quando quisermos melhor UX, mais carros na tela ou embutir em outra página: o **mesmo
núcleo** roda no servidor (FastAPI), faz *streaming* do estado por frame (JSON/MessagePack)
e um front dedicado renderiza — **PixiJS** para a cena, **D3** para o grafo de neurônios e
os gráficos. Contrato de mensagens tipado com Pydantic ↔ TypeScript. Custo: duas
linguagens e um servidor que gasta CPU por usuário (deploy em Fly.io/Render; front em
Vercel/Netlify).

### Comparativo
| Critério | Trilha A (`pygbag`) | Trilha B (FastAPI + PixiJS) |
|---|---|---|
| Linguagens | só Python | Python + JS/TS |
| Publicação | site estático grátis (Pages/Spaces) | backend hospedado + front estático |
| Escala (muitos usuários) | ótima (roda no cliente) | limitada por CPU do servidor |
| UX / animação | boa | melhor |
| Esforço | baixo | alto |
| Reuso do núcleo | total | total |

## 7. Visualização dos neurônios

- Após cada `forward` do carro em foco (o melhor da geração), guardar os vetores de ativação por camada.
- `netview` desenha nós em colunas (entrada · oculta · saída); **cor do nó = valor da ativação**
  (mapa divergente: negativo → azul, positivo → vermelho).
- **Arestas destacadas pelo sinal** `peso × ativação_da_origem`, normalizado pelo máximo do
  frame: ligações que conduzem sinal ficam mais grossas e coloridas (vermelho/azul),
  as demais ficam fininhas e escuras.
- Rótulos: entradas `S0..Sn` + `vel`; saídas `dir`, `acel`.
- Atualiza a cada frame de render, com os pesos do melhor genoma e o `Activation` do carro.

## 8. Plano de execução (fases / marcos)

| Fase | Entrega | Critério de "pronto" |
|---|---|---|
| **0 — Setup** | Repo, `uv`, deps, lint, esqueleto `docs/` + `prompts/` | `uv run pytest` roda; lint limpo |
| **1 — Simulação (sem UI)** | `geometry`, `track` (carrega JSON), `car`, `sensors`, colisão + checkpoints; run *headless* de um controlador heurístico | Carro heurístico completa 1 volta *headless*; testes de `geometry`/`track`/`sensors` passam |
| **2 — Rede** | MLP NumPy: `forward`, get/set de parâmetros planos, snapshot de ativações | Testes de formato/limites; forward determinístico |
| **3 — Evolução** | GA (torneio, crossover, mutação, elitismo), fitness, `trainer` incremental (step-a-step), treino *headless*; salvar melhor genoma; log de métricas | Um genoma evoluído completa 1 volta sem bater; curva de fitness sobe |
| **4 — App Pygame (desktop)** | Loop `async`, `scene` (pista/carro/sensores), HUD; anima o melhor genoma | Roda a ≥30 fps; melhor genoma visível dirigindo |
| **5 — Grafo da rede** | `netview` sincronizado com o carro em foco; `plot` de fitness | Ativações mudam visivelmente nas curvas |
| **6 — Painel de controle** | `panel` de hiperparâmetros + botões; treino cooperativo no loop; aplicar/reset/salvar/carregar (JSON) | Alterar slider + "Aplicar" muda o comportamento sem travar a UI |
| **7 — Build web (`pygbag`) + publicação** | `scripts/build_web.sh`; GitHub Action publicando no Pages; espelho num HF Space Static | URL pública abre e treina no browser a ≥20 fps |
| **8 — Polimento didático** | 2–3 pistas + gerador procedural opcional; *presets*; *tooltips*; README + `docs/`; roteiro guiado; GIF; (opcional) exe PyInstaller | Terceiro roda pelo README, local e pela URL |
| **9 — Trilha B (opcional)** | `server/` FastAPI + `frontend/` PixiJS/D3 reusando o núcleo | Mesma demo, UX de "produto web" |

**Ordem de dependências:** 0 → 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8 → (9).
Fases 1–3 entregam a "prova de que aprende" antes de qualquer UI.

## 9. Riscos e mitigações

| Risco | Mitigação |
|---|---|
| GA não converge | Fitness por progresso na central (checkpoints); normalizar entradas; sensores mais abertos; elitismo + diversidade inicial; heurística como *baseline* e possível *warm-start* por *cloning*. |
| Performance no browser (WASM) | MLP rasa; população ~20–40; geometria vetorizada em NumPy; animar só herói + poucos fantasmas; "modo rápido" sem render durante o treino. |
| Sem threads no `pygbag` | `trainer` incremental desde a Fase 3 (nunca dependeu de thread); N passos de GA por frame + `await asyncio.sleep(0)`. |
| Dependência fora do Pyodide (`shapely`, `matplotlib`) | Núcleo só com `numpy` + stdlib; geometria própria em `geometry.py`; gráfico com primitivas do Pygame; libs pesadas só em `tools/` offline. |
| `pygame_gui` pesado/limitado sob `pygbag` | Fallback: widgets próprios desenhados no canvas (sliders/botões simples). |
| Tamanho do *download* inicial | Enxugar assets; loader amigável; documentar o "primeiro carregamento demora". |
| Geometria da pista (auto-interseção, *offset*) | Pistas feitas à mão + validadas; `tools/make_track.py` (offline, com `shapely`) gera JSON revisado. |
| *Scope creep* (NEAT, RL, Trilha B) | Explicitamente fases 9+; MVP = fases 0–8. |

## 10. Resumo

- **Stack:** Python 3.12 · numpy · pygame-ce · pygbag · pytest/ruff · uv · reportlab (extra `docs`).
- **Abordagem:** sensores raycast + MLP rasa em NumPy, treinada por **duas soluções**:
  neuroevolução (GA) e **DQN** (backprop manual + replay). **Núcleo sem dependência de UI**;
  treino incremental (sem threads); app Pygame multi-tela.
- **Entrega:** app Pygame único → `pygbag` → **site estático (GitHub Pages ou HF Spaces
  Static)**, com `tutorial_ga`/`tutorial_dqn` em `.html`/`.pdf` ao lado; roda igual no desktop.
- **Evolução opcional (Trilha B):** FastAPI + WebSocket + PixiJS/D3, reusando o núcleo.
- **Primeiro valor entregável:** núcleo GA — um carro que aprende sozinho a completar uma
  volta; e a Solução 2 (DQN) fazendo o mesmo por gradiente.
