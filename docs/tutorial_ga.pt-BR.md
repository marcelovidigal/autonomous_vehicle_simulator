# Tutorial — Rede Neural + Algoritmo Genético

Esta é a **Solução 1**: uma **rede neural rasa** dirige o carro, e os pesos dela são
encontrados por um **algoritmo genético** (neuroevolução) — sem gradiente, sem "gabarito".
O tutorial cobre a teoria mínima, os **sensores por raycast**, a rede, o algoritmo genético
(com prós e contras) e o código passo a passo.

> A técnica de aprendizado por reforço (DQN) tem tutorial próprio, na tela dela.

---

## 1. O que esta solução faz, em 30 segundos

- O carro "enxerga" a pista com **raios de distância** (raycasts).
- Uma **MLP** recebe essas distâncias + a velocidade e devolve **direção** e **acelerador**.
- Uma **população** de MLPs tenta dirigir; as que vão mais longe e mais rápido sem bater
  **se reproduzem**, com mutações, geração após geração. O melhor genoma é animado na tela,
  junto do grafo de neurônios e da curva de fitness.

---

## 2. Teoria em 5 minutos

### 2.1 O "olhar" do carro: raios em vez de câmera
O carro não tem câmera. Ele "enxerga" com **sensores de distância** — *raycasts*: do centro
do carro saem `N` raios num leque (padrão 5, abertura 160°, alcance 150 unidades). Cada raio
devolve **a distância até a primeira parede**, **normalizada** para `[0, 1]` (1 = nada dentro
do alcance; perto de 0 = parede colada).

Por que raios e não uma câmera de verdade:

- **Barato**: cada leitura é uma interseção geométrica, não um tensor de pixels — dá para
  simular dezenas de carros por milhares de passos em tempo real e rodar liso no navegador.
- **Determinístico e sem ruído**: a mesma pose sempre dá as mesmas leituras (treino
  reprodutível dada uma *seed*).
- **Entrada compacta**: 5–7 números bastam para se manter na pista; a rede fica pequena e o
  grafo de neurônios cabe na tela.
- **Custo**: o carro só "vê" ao longo das linhas dos raios; entre dois raios, ou além do
  alcance, ele é cego — por isso quantidade e abertura são hiperparâmetros.

A matemática do raycast está no passo a passo (§6.2).

### 2.2 Controle reativo
A cada instante o carro observa (sensores + velocidade) e decide uma ação. Como a observação
atual já contém o que importa, basta uma função `observação → ação` — uma rede neural é uma
forma flexível de representá-la. Não é preciso "lembrar" do passado.

### 2.3 Rede neural = função com parâmetros ajustáveis
Uma MLP *feedforward*: multiplica a entrada por uma matriz de pesos, soma um viés, aplica
uma não-linearidade, repete. Os **pesos** são os parâmetros; "treinar" = achar pesos bons.

### 2.4 Achar pesos sem gabarito → evolução
Não existe "a saída correta era 0,3" para cada frame. Solução usada aqui: **busca por
evolução** — gere muitas redes, meça o desempenho de cada uma (*fitness*), fique com as
melhores, gere variações. Não precisa de gradiente.

---

## 3. Ferramentas e por quê

- **Python 3.11+** — linguagem única.
- **NumPy** — rede, geometria, simulação vetorizada. Roda também no navegador (Pyodide).
- **Pygame-CE** — janela, desenho 2D, entrada (pista, grafo da rede, painel).
- **pygbag** — empacota o app em **WebAssembly** (site estático).
- **uv** (ambiente) · **pytest** + **ruff** (testes e lint) · **reportlab** (PDF, opcional).

Ficou **de fora** de propósito: **shapely/scipy** (a geometria cabe em ~100 linhas de NumPy
e roda em Pyodide); **PyTorch/TensorFlow** (o GA não usa gradiente); **matplotlib** (as
curvas são desenhadas com primitivas do Pygame).

---

## 4. A rede neural

### 4.1 MLP rasa (sem memória)
A saída depende **só** da entrada atual. Como o estado relevante está na observação (os
sensores), memória (uma RNN) não compensa a dificuldade extra de treino — e uma MLP mantém
o **grafo de neurônios estável e legível**, que é o objetivo didático.

### 4.2 Arquitetura
`n_entradas → camada oculta → 2 saídas`, com `tanh`.

- Entradas: `count` distâncias em `[0,1]` + `velocidade/v_max` (padrão 5 + 1 = 6).
- Oculta: `hidden` neurônios (padrão 6), `tanh` (ou `relu`).
- Saídas: 2 valores em `[-1,1]` via `tanh` — **direção** e **acelerador**.
- `hidden = 0` → rede **linear**: bom para mostrar que circuitos fáceis dispensam camada oculta.

Com 6 entradas e 6 ocultas são **56 números** (pesos + vieses) — é esse vetor que o GA otimiza.

### 4.3 Implementação — `core/network.py`
- `MLP.__init__` cria `w1,b1,w2,b2` (ou só `w1,b1` se `hidden=0`), pesos `~ N(0, 1/√fan_in)`.
- `forward(x)`: `h = act(w1@x + b1)` ; `out = tanh(w2@h + b2)` ; guarda `self.last = (x,h,out)`
  para a visualização.
- `to_flat()` / `load_flat(vec)` — serializam os pesos num **vetor 1‑D** (ordem `w1,b1,w2,b2`).
  Um **genoma É esse vetor**.
- `n_params(n_in, cfg)` — tamanho esperado do vetor.

### 4.4 Vantagens e desvantagens
**Vantagens:** simples de entender/testar; `forward` barato (dá para simular dezenas de
carros em tempo real); sem memória → grafo estável; pesos como vetor plano → crossover e
mutação triviais.
**Desvantagens:** não "antecipa" nada; capacidade limitada (mais neurônios ⇒ evolução mais
lenta); `tanh` satura com pesos grandes.

---

## 5. O algoritmo genético

### 5.1 A ideia
1. **População** — muitos genomas (vetores de peso) aleatórios.
2. **Avaliação** — cada genoma vira uma MLP, dirige na simulação, recebe um **fitness**.
3. **Seleção** — genomas melhores têm mais chance de "ter filhos".
4. **Variação** — **crossover** (o filho herda genes dos pais) + **mutação** (ruído gaussiano).
5. **Elitismo** — os melhores passam intactos.
6. Repete; o fitness (melhor e médio) sobe ao longo das gerações.

### 5.2 Por que GA aqui
- Não há alvo diferenciável → backprop não se aplica direto.
- GA precisa só de "sei dizer se A foi melhor que B": robusto a fitness ruidoso/descontínuo,
  fácil de visualizar (a população inteira tentando) e naturalmente paralelo.

### 5.3 Implementação — `core/evolution.py`
- `init_population(pop, n_params, rng)` → matriz `(pop, n_params)` com `N(0, 0.5)`.
- `_tournament(fitness, k, rng)` → sorteia `k` genomas e devolve o índice do melhor (`k`
  controla a pressão seletiva).
- `next_population(...)`: copia os `elitism` melhores; para o resto, torneio → pai 1; se
  `crossover`, torneio → pai 2 e **crossover uniforme** (50/50 por gene); **mutação**: com
  prob. `mutation_rate` por gene, soma `N(0, mutation_sigma)`.
- Toda aleatoriedade passa por um `Generator` **semeado** → mesma seed, mesmo resultado.

### 5.4 A função de fitness — `core/simulation.py`
```
fitness =  w_progress * progresso_ao_longo_da_central     # ir longe (conta voltas)
         + w_speed    * velocidade_média_normalizada       # ir rápido
         - w_smooth   * zigue_zague_médio_do_volante       # dirigir liso
         - crash_penalty * (bateu ? 1 : 0)                 # não bater
         + lap_bonus  * voltas_completas                   # cruzou a linha
```
O **progresso** é distância ao longo da linha central (via ponto mais próximo), o que impede
"cortar caminho". Um carro parado/girando é encerrado por *stall*.

### 5.5 Vantagens e desvantagens
**Vantagens:** sem gradiente nem dados rotulados; fácil de mostrar; robusto a fitness
"quebrado"; paraleliza de graça.
**Desvantagens:** **amostra ineficiente** (cada avaliação é uma simulação inteira); "tateia"
o espaço de pesos; sensível a `mutation_sigma`/`elitism`; pode convergir cedo para um ótimo
local.

---

## 6. Tutorial do código, passo a passo

Tudo em `src/`. Ordem recomendada de leitura.

### 6.1 `core/config.py`
`dataclasses` **congeladas** com os hiperparâmetros: `NetworkCfg`, `SensorCfg`, `PhysicsCfg`,
`GACfg`, `FitnessCfg`, `SimCfg` (em `Config`). `to_dict`/`from_dict` = JSON.

### 6.2 `core/geometry.py` — geometria e o raycast
- `resample_closed(pts, n)` — reamostra a central em `n` pontos igualmente espaçados por arco.
- `offset_closed(pts, dist)` — desloca a central pela normal de cada vértice → **bordas**.
- `fan_angles(count, fov)` — ângulos relativos do leque de sensores.
- `ray_fan_distances(origin, dirs, seg_a, seg_b, max_dist)` — **o coração dos sensores**.
  Para cada raio (origem `O`, direção unitária `d`) contra cada segmento `A→B`:
  ```
  v1 = O − A     v2 = B − A     perp = (−d_y, d_x)
  t_raio = cross(v2, v1) / dot(v2, perp)     # distância ao longo do raio
  t_seg  = dot(v1, perp)  / dot(v2, perp)    # posição no segmento (precisa estar em [0,1])
  ```
  Vale o toque se `dot(v2, perp) ≠ 0`, `t_raio ≥ 0`, `0 ≤ t_seg ≤ 1`. A leitura é o **menor
  `t_raio` válido**, limitado a `max_dist`. Feito de uma vez para `M` raios × `N` segmentos
  com *broadcasting* NumPy — sem laço Python.

### 6.3 `core/track.py` — a pista e a leitura de sensores
- `__init__`: reamostra a central, gera `outer`/`inner` por offset (maior área = externa),
  concatena os **segmentos de parede** e a tabela de comprimento acumulado `cum`.
- `sensor_readings(pos, heading, cfg)` — direções `heading + fan_angles(...)` →
  `ray_fan_distances` contra as paredes → `distância / range` em `[0,1]`. É a entrada da rede.
- `locate(pos)` — ponto mais próximo na central + distância: serve para **colisão**
  (`> half_w + folga`) e **progresso**.
- `commit_progress(state, i)` — atualiza índice, detecta a **virada de volta** e marca `finished`.
- Pistas em `src/tracks/*.json` (`tools/make_track.py`: curvas polares suaves).

### 6.4 `step_car` em `core/simulation.py` — cinemática
```
heading += steer * taxa_esterço * dt * aderência   # aderência ~ velocidade: não vira parado
speed   += throttle * aceleração * dt
speed   -= speed * atrito
speed    = clip(speed, 0, v_max)
pos     += speed * dt * (cos heading, sin heading)
```

### 6.5 `core/network.py` — a MLP
Ver §4.3.

### 6.6 `core/simulation.py` — avaliar genomas
- `simulate(genome, track, cfg, record=False)` — **versão legível**: laço Python passo a
  passo (sensores → `forward` → `step_car` → colisão/progresso → fitness). `record=True`
  grava um `trace` (posição, heading, ativações por frame) para a animação.
- `simulate_population(genomes, track, cfg)` — **vetorizada**: roda a população em *lockstep*
  (`einsum`); quem bate/empaca sai da máscara `alive`. É o que o treino usa.
- As duas compartilham `step_car` e a fórmula `_score`; um teste garante que concordam.

### 6.7 `core/evolution.py`
Ver §5.3.

### 6.8 `core/trainer.py` — treino incremental (sem threads)
- `reset(cfg)` — semeia o RNG, carrega a pista, monta a população.
- `step(budget_ms)` — avança pelo tempo dado, avaliando a população em **sub-lotes**; ao
  fechar a geração, `_finish_generation` registra melhor/média e gera a próxima. É o que
  permite treinar **dentro do loop de render** (sem travar, sem thread — essencial p/ web).
- `advance_generation()` — geração inteira de uma vez. `best_trace()` — regrava o trace do
  melhor genoma. `save()` — genoma + config em JSON.

### 6.9 `app/gascreen.py` — esta tela
Painel de sliders (camada oculta, sensores, população, mutação, elitismo, pesos de fitness)
+ botões (Play/Pause, +1 Geração, Turbo, Re-treinar, Reset, Pista). A cada frame:
`trainer.step(...)`; se surgiu um novo melhor, recomputa o `trace` e a `MLP`; desenha a
cena (pista + carro + raios), o `netview` (nós = ativação, arestas = sinal) e a curva.
A barra superior troca de técnica; **Tutorial** abre este texto; **Voltar** retorna aqui.

---

## 7. Exercícios sugeridos

1. `Camada oculta = 0` e re-treine no `circuito_1`: ainda dirige? E no `circuito_3`?
2. `Força mutação` ~0,6: a curva de fitness fica mais errática — por quê?
3. Zere `Penal. colisão`: o que o carro passa a fazer?
4. `Sensores = 9`: melhora nas curvas fechadas?
5. No código: troque o crossover uniforme por "ponto único" em `evolution.py` e compare.
6. Adicione um 3º neurônio de saída ("freio") e ajuste `network.py` + `step_car`.

---

## 8. Referências

- Backprop e MLPs: *Deep Learning* (Goodfellow, Bengio, Courville), cap. 6.
- Neuroevolução: Stanley & Miikkulainen, *NEAT*, 2002.
- Algoritmos genéticos: Melanie Mitchell, *An Introduction to Genetic Algorithms*.
- Sensores por raycast + carro 2D: a clássica série de demos "self-driving car" em JS.
