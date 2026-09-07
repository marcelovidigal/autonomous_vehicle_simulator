# Tutorial — DQN (Deep Q-Network)

Esta é a **Solução 2**: um **único carro** aprende a dirigir **por recompensa**, ajustando
uma **rede Q** por *backpropagation*. O tutorial cobre a teoria mínima, os **sensores por
raycast**, a rede Q, o algoritmo DQN (com prós e contras) e o código passo a passo.

> A técnica de neuroevolução (rede neural + algoritmo genético) tem tutorial próprio, na
> tela dela.

---

## 1. O que esta solução faz, em 30 segundos

- O carro "enxerga" a pista com **raios de distância** (raycasts).
- Uma **rede Q** recebe essas distâncias + a velocidade e estima o **valor de cada ação**
  (`Q(estado, ação)`) — quanto retorno esperar de virar mais/menos para cada lado.
- O carro age escolhendo `argmax Q` (com um pouco de exploração aleatória). As transições
  vão para um *replay buffer*; a rede é ajustada por gradiente para que suas estimativas
  fiquem coerentes com o que aconteceu. O gráfico mostra a **recompensa por episódio**.

O carro que você vê é uma **demonstração** da política gulosa atual, rodando a ritmo fixo
(1 passo por frame) — o treino, por baixo, corre o mais rápido que couber.

---

## 2. Teoria em 5 minutos

### 2.1 O "olhar" do carro: raios em vez de câmera
Igual à Solução 1. Do centro do carro saem `N` raios num leque (padrão 5, abertura 160°,
alcance 150). Cada raio devolve a **distância até a primeira parede**, normalizada para
`[0, 1]` (1 = livre; ~0 = parede colada). Vantagens: barato, determinístico, entrada
compacta. Custo: o carro só "vê" nas linhas dos raios. A matemática está em §6.2.

### 2.2 A ideia de "valor de ação" (Q)
Em vez de aprender diretamente "o que fazer", aprende-se **quanto vale cada opção**:
`Q(s, a)` = soma esperada das recompensas futuras se eu tomar a ação `a` no estado `s` e
depois seguir jogando bem. Sabendo `Q`, a política é trivial: **escolha a ação de maior Q**.

### 2.3 Aprender Q de recompensa
`Q` obedece à **equação de Bellman**:
```
Q(s, a)  ≈  r  +  γ · maxₐ' Q(s', a')
```
`r` é a recompensa imediata, `s'` o próximo estado, `γ ∈ [0,1)` o **desconto** (o quanto o
futuro importa). Treinar = empurrar `Q(s,a)` para esse alvo, repetidamente, com gradiente.

---

## 3. Ferramentas e por quê (foco DQN)

- **NumPy** — a rede Q, o **backprop manual** e o otimizador **Adam** (~10 linhas cada).
  Sem framework de deep learning.
- **Replay buffer** — arrays NumPy circulares.
- **Rede-alvo** — uma cópia da rede Q, sincronizada de tempos em tempos.
- **Pygame-CE / pygbag / uv / pytest / ruff** — como no resto do projeto.

Ficou **de fora** de propósito:

- **PyTorch / TensorFlow** — a rede Q é minúscula; *backprop* de uma MLP de 1 camada é
  ~30 linhas e deixa tudo explícito (e o download web fica pequeno).
- **Gymnasium / stable-baselines3** — o ambiente (a pista) e o laço de RL são simples o
  bastante para escrever direto; assim o DQN inteiro cabe em um arquivo legível.

---

## 4. A rede Q — `QNet` em `core/dqn.py`

### 4.1 Formato
`entrada → 1 camada oculta (ReLU) → Q linear`.

- Entradas: `count` distâncias `[0,1]` + `velocidade/v_max` (padrão 6).
- Oculta: `hidden` neurônios (padrão 32), **ReLU**.
- Saída: **5 valores lineares** (sem `tanh`) — o `Q` de cada ação discreta:
  virar `{-1, -0,5, 0, +0,5, +1}` com acelerador fixo.

### 4.2 forward, backprop, Adam
```
z1 = x @ W1ᵀ + b1 ;  h = relu(z1) ;  q = h @ W2ᵀ + b2      # forward (em lote)
```
`train_step(estados, ações, alvos, lr)`:
- erro só na ação tomada: `e = clip(Q(s,a) − alvo, −1, 1)` (recorte estilo **Huber** — evita
  passos gigantes quando o alvo está muito longe);
- retropropaga `e` por `W2, b2`, depois pela ReLU (`z1 > 0`), depois por `W1, b1`;
- **Adam** (médias móveis do gradiente e do seu quadrado) aplica o passo.

`copy_from(outra)` — copia os pesos (para sincronizar a rede-alvo).

### 4.3 Vantagens e desvantagens da rede Q
**Vantagens:** saída **interpretável** (dá para ver o Q de cada ação e a ação gulosa);
ReLU + linear treina bem por gradiente; minúscula e rápida.
**Desvantagens:** **ações discretas** → controle de volante em degraus (5 níveis), acelerador
fixo; a estimativa de Q pode ficar super/subestimada e desestabilizar o treino.

---

## 5. O algoritmo DQN — `DQNTrainer`

### 5.1 Peças
1. **Ações discretas** (5, acima).
2. **Recompensa por passo** — **densa**: `R_PROGRESS · Δprogresso − R_CRASH·[bateu]`. É o
   sinal que guia o gradiente; sem densidade, o DQN não sai do lugar.
3. **Replay buffer** — guarda `(s, a, r, s', fim)`; o treino amostra *minibatches*
   aleatórios (quebra a correlação temporal entre passos vizinhos).
4. **Rede-alvo** — cópia "congelada" de Q usada no lado direito de Bellman; sincronizada a
   cada `target_sync` passos. Sem ela, o alvo "corre atrás" da própria rede e diverge.
5. **ε-greedy** — com probabilidade `ε` a ação é aleatória (exploração); `ε` decai de 1,0
   até 0,05 ao longo de `eps_decay_steps`.

### 5.2 Laço (`core/dqn.py`)
- `reset(cfg, dqn)` — semeia o RNG, carrega a pista, cria `QNet` online e alvo, o `Replay`.
- `step(budget_ms)` — enquanto houver tempo: `_env_step` (escolhe ε-greedy, anda um passo,
  calcula recompensa, `Replay.push`) e, passado o `warmup`, `_learn` (amostra um lote,
  monta o alvo `r + γ·max Q_alvo(s')·(1−fim)`, um `train_step`; a cada `target_sync`
  sincroniza a rede-alvo). Fim de episódio (bateu / *stall* / limite) → registra a
  recompensa acumulada em `curve`.
- `demo_step()` — avança **1 passo** um "carro-demonstração" independente com a política
  gulosa atual; é o que a tela desenha, a ritmo fixo, para a visualização não ficar
  acelerada quando o treino está rápido.
- **Determinístico** dada `DQNCfg.seed` (init, exploração e amostragem do mesmo `Generator`).

### 5.3 Hiperparâmetros na tela
Nº de **neurônios ocultos**, **taxa de aprendizado** (`lr`), **γ (desconto)** e
**decaimento de ε**. `Reset` aplica as mudanças.

### 5.4 Vantagens e desvantagens do DQN
**Vantagens:** **amostra mais eficiente** que o GA quando funciona (reaproveita cada
transição via replay); política de **valor interpretável**; o gradiente aponta a direção de
melhora.
**Desvantagens:** **instável** — sensível a `lr`, `γ`, tamanho do buffer e sincronização do
alvo; pode divergir ou "esquecer" (a curva de recompensa oscila bem mais que a do GA);
precisa de **recompensa densa e bem escalada**; menos "visual" (um carro só).

### 5.5 GA × DQN, lado a lado
| | Algoritmo Genético | DQN |
|---|---|---|
| Usa gradiente? | não | sim (backprop manual) |
| Unidade de treino | geração (população) | passo / episódio |
| Recompensa densa? | não (fitness no fim) | sim (por passo) |
| Estabilidade | alta | média (oscila) |
| Eficiência de amostra | baixa | maior (replay) |
| Saídas da rede | direção/acelerador (contínuas) | Q de 5 ações discretas |
| Bom para mostrar… | seleção natural, muitas tentativas | valor de ação, exploração/explotação |

---

## 6. Tutorial do código, passo a passo

### 6.1 `core/config.py` — `DQNCfg`
`dataclass` congelada com: `hidden`, `lr`, `gamma`, `batch`, `buffer`, `warmup`,
`target_sync`, `eps_start`/`eps_end`/`eps_decay_steps`, `max_steps`, `seed`. Os sensores e a
física vêm do `Config` comum.

### 6.2 Sensores e pista (compartilhados) — `core/geometry.py`, `core/track.py`
O DQN usa exatamente a mesma leitura de sensores da Solução 1:
```
v1 = O − A     v2 = B − A     perp = (−d_y, d_x)
t_raio = cross(v2, v1) / dot(v2, perp)     # distância ao longo do raio
t_seg  = dot(v1, perp)  / dot(v2, perp)    # ∈ [0,1] para valer o toque
```
A leitura de cada sensor é o menor `t_raio` válido, limitado a `max_dist`, dividido por
`range`. `track.locate(pos)` dá colisão e progresso; `commit_progress` detecta a volta.

### 6.3 `step_car` em `core/simulation.py` — cinemática (compartilhada)
```
heading += steer * taxa_esterço * dt * aderência
speed   += throttle * aceleração * dt   ;   speed -= speed * atrito   ;   speed = clip(...)
pos     += speed * dt * (cos heading, sin heading)
```
No DQN, `steer` vem da ação discreta e `throttle` é fixo.

### 6.4 `core/dqn.py` — o coração desta solução
- **`QNet`** — §4. `forward` em lote, `train_step` (backprop + Adam), `copy_from`.
- **`Replay`** — `push(s,a,r,s2,done)` circular; `sample(batch)` uniforme.
- **`DQNTrainer`** — §5. Mesma interface do `Trainer` do GA (`state`, `step`, `reset`,
  `curve`, `best_reward`, `best_laps`), então a tela trata os dois iguais.
- Constantes de recompensa: `STEER_ACTIONS`, `THROTTLE`, `R_PROGRESS`, `R_CRASH`, `STALL_STEPS`.

### 6.5 `app/dqnscreen.py` — esta tela
Painel: neurônios ocultos, `lr`, `γ`, decaimento de ε; botões Play/Pause, Turbo, Reset,
Pista. A cada frame: `trainer.step(...)` (treina rápido) e `trainer.demo_step()` (1 passo
da política gulosa, para a visualização a ritmo fixo). Desenha o carro-demonstração + raios,
o `netview` (nós = ativação; **saídas = Q de cada ação**), a curva de recompensa e o HUD
(episódio, ε, loss, buffer, recompensa/voltas, ação gulosa atual).

---

## 7. Exercícios sugeridos

1. Dobre a **taxa de aprendizado** — a curva de recompensa fica mais instável? Diverge?
2. Baixe **γ** para ~0,85 — o carro fica "míope" (só pensa a curto prazo)?
3. Reduza o **decaimento de ε** (decai rápido) — ele para de explorar cedo demais?
4. Aumente os **neurônios ocultos** para 64 — aprende melhor ou só mais devagar?
5. No código: aumente `R_CRASH` em `dqn.py` — o carro fica mais medroso?
6. No código: passe o acelerador a ser também uma dimensão de ação (3 níveis × 5 de volante
   = 15 ações) e veja o efeito no tempo de aprendizado.

---

## 8. Referências

- Q-learning e DQN: Mnih et al., *Human-level control through deep reinforcement learning*,
  Nature 2015; Sutton & Barto, *Reinforcement Learning: An Introduction*, cap. 6 e 9.
- Adam: Kingma & Ba, *Adam: A Method for Stochastic Optimization*, 2015.
- Backprop: *Deep Learning* (Goodfellow, Bengio, Courville), cap. 6.
- Sensores por raycast + carro 2D: a clássica série de demos "self-driving car" em JS.
