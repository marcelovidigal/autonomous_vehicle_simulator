# Tutorial — DQN (Deep Q-Network)

This is **Solution 2**: a **single car** learns to drive **from reward**, tuning a **Q
network** by *backpropagation*. This tutorial covers the minimal theory, the **raycast
sensors**, the Q network, the DQN algorithm (with pros and cons) and a step-by-step
walkthrough of the code.

> The neuroevolution technique (neural network + genetic algorithm) has its own tutorial,
> on its screen.

---

## 1. What this solution does, in 30 seconds

- The car "sees" the track with **distance rays** (raycasts).
- A **Q network** takes those distances + the speed and estimates the **value of each
  action** (`Q(state, action)`) — how much return to expect from steering more/less to each
  side.
- The car acts by picking `argmax Q` (with a bit of random exploration). Transitions go into
  a *replay buffer*; the network is tuned by gradient so its estimates become consistent
  with what happened. The chart shows the **reward per episode**.

The car you see is a **demonstration** of the current greedy policy, running at a fixed pace
(1 step per frame) — training, underneath, runs as fast as it fits.

---

## 2. Theory in 5 minutes

### 2.1 The car's "sight": rays instead of a camera
Same as Solution 1. From the car's centre, `N` rays fan out (default 5, 160° spread, range
150). Each ray returns the **distance to the first wall**, normalised to `[0, 1]` (1 = clear;
~0 = wall touching). Pros: cheap, deterministic, compact input. Cost: the car only "sees"
along the ray lines. The maths is in §6.2.

### 2.2 The idea of "action value" (Q)
Instead of learning directly "what to do", we learn **how much each option is worth**:
`Q(s, a)` = the expected sum of future rewards if I take action `a` in state `s` and then
keep playing well. Knowing `Q`, the policy is trivial: **pick the highest-Q action**.

### 2.3 Learning Q from reward
`Q` satisfies the **Bellman equation**:
```
Q(s, a)  ~  r  +  gamma * max_a' Q(s', a')
```
`r` is the immediate reward, `s'` the next state, `gamma` in `[0,1)` the **discount** (how
much the future matters). Training = pushing `Q(s,a)` towards that target, repeatedly, with
gradient descent.

---

## 3. Tools and why (DQN focus)

- **NumPy** — the Q network, the **manual backprop** and the **Adam** optimiser (~10 lines
  each). No deep-learning framework.
- **Replay buffer** — circular NumPy arrays.
- **Target network** — a copy of the Q network, synced from time to time.
- **Pygame-CE / pygbag / uv / pytest / ruff** — as in the rest of the project.

Deliberately **left out**:

- **PyTorch / TensorFlow** — the Q network is tiny; *backprop* of a 1-hidden-layer MLP is
  ~30 lines and keeps everything explicit (and the web download stays small).
- **Gymnasium / stable-baselines3** — the environment (the track) and the RL loop are simple
  enough to write directly; that way the whole DQN fits in one readable file.

---

## 4. The Q network — `QNet` in `core/dqn.py`

### 4.1 Shape
`input → 1 hidden layer (ReLU) → linear Q`.

- Inputs: `count` distances in `[0,1]` + `speed/v_max` (default 6).
- Hidden: `hidden` neurons (default 32), **ReLU**.
- Output: **5 linear values** (no `tanh`) — the `Q` of each discrete action: steer
  `{-1, -0.5, 0, +0.5, +1}` with a fixed throttle.

### 4.2 forward, backprop, Adam
```
z1 = x @ W1.T + b1 ;  h = relu(z1) ;  q = h @ W2.T + b2      # forward (batched)
```
`train_step(states, actions, targets, lr)`:
- error only on the taken action: `e = clip(Q(s,a) - target, -1, 1)` (**Huber**-style
  clipping — avoids huge steps when the target is far off);
- backprops `e` through `W2, b2`, then the ReLU (`z1 > 0`), then `W1, b1`;
- **Adam** (running averages of the gradient and its square) applies the step.

`copy_from(other)` — copies the weights (to sync the target network).

### 4.3 Pros and cons of the Q network
**Pros:** **interpretable** output (you can see the Q of each action and the greedy one);
ReLU + linear trains well by gradient; tiny and fast.
**Cons:** **discrete actions** → steering control in steps (5 levels), fixed throttle; the Q
estimate can become over/under-estimated and destabilise training.

---

## 5. The DQN algorithm — `DQNTrainer`

### 5.1 The pieces
1. **Discrete actions** (5, above).
2. **Per-step reward** — **dense**: `R_PROGRESS * delta_progress - R_CRASH*[crashed]`. It is
   the signal that guides the gradient; without density, DQN goes nowhere.
3. **Replay buffer** — stores `(s, a, r, s', done)`; training samples random *minibatches*
   (breaks the temporal correlation between neighbouring steps).
4. **Target network** — a "frozen" copy of Q used on the right-hand side of Bellman; synced
   every `target_sync` steps. Without it the target "chases" the network itself and diverges.
5. **epsilon-greedy** — with probability `epsilon` the action is random (exploration);
   `epsilon` decays from 1.0 to 0.05 over `eps_decay_steps`.

### 5.2 The loop (`core/dqn.py`)
- `reset(cfg, dqn)` — seeds the RNG, loads the track, creates the online `QNet` and the
  target, and the `Replay`.
- `step(budget_ms)` — while there is time: `_env_step` (pick epsilon-greedy, take one step,
  compute reward, `Replay.push`) and, past `warmup`, `_learn` (sample a batch, build the
  target `r + gamma*max Q_target(s')*(1-done)`, one `train_step`; every `target_sync` steps
  sync the target). End of episode (crash / *stall* / limit) → record the accumulated reward
  in `curve`.
- `demo_step()` — advances **1 step** an independent "demo car" with the current greedy
  policy; it is what the screen draws, at a fixed pace, so the visualisation is not
  fast-forwarded when training is quick.
- **Deterministic** given `DQNCfg.seed` (init, exploration and sampling from the same
  `Generator`).

### 5.3 On-screen hyperparameters
Number of **hidden units**, **learning rate** (`lr`), **gamma (discount)** and
**epsilon decay**. **Apply** (key `a`) rebuilds the Q network from the sliders and restarts
training; it lights up while a slider differs from the running config. `Reset` puts every
slider back to its default and applies that.

### 5.4 Pros and cons of DQN
**Pros:** **more sample-efficient** than the GA when it works (reuses each transition via
replay); an **interpretable value** policy; the gradient points towards improvement.
**Cons:** **unstable** — sensitive to `lr`, `gamma`, buffer size and target sync; can
diverge or "forget" (the reward curve wobbles much more than the GA's); needs a **dense,
well-scaled reward**; less "visual" (a single car).

### 5.5 GA vs DQN, side by side
| | Genetic Algorithm | DQN |
|---|---|---|
| Uses gradient? | no | yes (manual backprop) |
| Training unit | generation (population) | step / episode |
| Dense reward? | no (fitness at the end) | yes (per step) |
| Stability | high | medium (wobbles) |
| Sample efficiency | low | higher (replay) |
| Network outputs | steering/throttle (continuous) | Q of 5 discrete actions |
| Good for showing... | natural selection, many attempts | action value, exploration/exploitation |

---

## 6. Code walkthrough

### 6.1 `core/config.py` — `DQNCfg`
A frozen `dataclass` with: `hidden`, `lr`, `gamma`, `batch`, `buffer`, `warmup`,
`target_sync`, `eps_start`/`eps_end`/`eps_decay_steps`, `max_steps`, `seed`. The sensors and
physics come from the shared `Config`.

### 6.2 Sensors and track (shared) — `core/geometry.py`, `core/track.py`
DQN uses exactly the same sensor reading as Solution 1:
```
v1 = O - A     v2 = B - A     perp = (-d_y, d_x)
t_ray = cross(v2, v1) / dot(v2, perp)     # distance along the ray
t_seg = dot(v1, perp)  / dot(v2, perp)    # in [0,1] for the hit to count
```
Each sensor reading is the smallest valid `t_ray`, clamped to `max_dist`, divided by
`range`. `track.locate(pos)` gives collision and progress; `commit_progress` detects the lap.

### 6.3 `step_car` in `core/simulation.py` — kinematics (shared)
```
heading += steer * steer_rate * dt * grip
speed   += throttle * accel * dt   ;   speed -= speed * friction   ;   speed = clip(...)
pos     += speed * dt * (cos heading, sin heading)
```
In DQN, `steer` comes from the discrete action and `throttle` is fixed.

### 6.4 `core/dqn.py` — the heart of this solution
- **`QNet`** — §4. Batched `forward`, `train_step` (backprop + Adam), `copy_from`.
- **`Replay`** — circular `push(s,a,r,s2,done)`; uniform `sample(batch)`.
- **`DQNTrainer`** — §5. Same interface as the GA `Trainer` (`state`, `step`, `reset`,
  `curve`, `best_reward`, `best_laps`), so the screen treats both the same.
- Reward constants: `STEER_ACTIONS`, `THROTTLE`, `R_PROGRESS`, `R_CRASH`, `STALL_STEPS`.

### 6.5 `app/dqnscreen.py` — this screen
Panel: hidden units, `lr`, `gamma`, epsilon decay; buttons Play/Pause, Turbo, Apply, Reset, Track.
Each frame: `trainer.step(...)` (trains fast) and `trainer.demo_step()` (1 greedy-policy
step, for the fixed-pace visualisation). It draws the demo car + rays, the `netview` (nodes
= activation; **outputs = Q of each action**), the reward curve and the HUD (episode,
epsilon, loss, buffer, reward/laps, current greedy action).

---

## 7. Suggested exercises

1. Double the **learning rate** — does the reward curve get more unstable? Diverge?
2. Lower **gamma** to ~0.85 — does the car become "short-sighted" (only thinks near-term)?
3. Reduce the **epsilon decay** (decays fast) — does it stop exploring too early?
4. Raise the **hidden units** to 64 — does it learn better or just more slowly?
5. In the code: raise `R_CRASH` in `dqn.py` — does the car get more timid?
6. In the code: make the throttle a second action dimension (3 levels x 5 steering = 15
   actions) and see the effect on learning time.

---

## 8. References

- Q-learning and DQN: Mnih et al., *Human-level control through deep reinforcement learning*,
  Nature 2015; Sutton & Barto, *Reinforcement Learning: An Introduction*, ch. 6 and 9.
- Adam: Kingma & Ba, *Adam: A Method for Stochastic Optimization*, 2015.
- Backprop: *Deep Learning* (Goodfellow, Bengio, Courville), ch. 6.
- Raycast sensors + 2D car: the classic "self-driving car" JS demo series.
