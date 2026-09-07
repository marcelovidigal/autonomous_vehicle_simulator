# Tutorial — Neural Net + Genetic Algorithm

This is **Solution 1**: a **shallow neural network** drives the car, and its weights are
found by a **genetic algorithm** (neuroevolution) — no gradient, no "ground truth". This
tutorial covers the minimal theory, the **raycast sensors**, the network, the genetic
algorithm (with pros and cons) and a step-by-step walkthrough of the code.

> The reinforcement-learning technique (DQN) has its own tutorial, on its screen.

---

## 1. What this solution does, in 30 seconds

- The car "sees" the track with **distance rays** (raycasts).
- An **MLP** takes those distances + the speed and returns **steering** and **throttle**.
- A **population** of MLPs tries to drive; the ones that go farther and faster without
  crashing **reproduce**, with mutations, generation after generation. The best genome is
  animated on screen, alongside the neuron graph and the fitness curve.

---

## 2. Theory in 5 minutes

### 2.1 The car's "sight": rays instead of a camera
The car has no camera. It "sees" with **distance sensors** — *raycasts*: from the car's
centre, `N` rays fan out (default 5, 160° spread, 150-unit range). Each ray returns **the
distance to the first wall**, **normalised** to `[0, 1]` (1 = nothing within range; near 0 =
wall touching).

Why rays and not a real camera:

- **Cheap**: each reading is a geometric intersection, not a pixel tensor — you can simulate
  dozens of cars over thousands of steps in real time and it runs smoothly in the browser.
- **Deterministic and noise-free**: the same pose always gives the same readings (training
  is reproducible given a *seed*).
- **Compact input**: 5–7 numbers are enough to stay on the track; the network stays small
  and the neuron graph fits on screen.
- **Cost**: the car only "sees" along the ray lines; between two rays, or beyond range, it
  is blind — that is why the count and spread are hyperparameters.

The raycast maths is in the walkthrough (§6.2).

### 2.2 Reactive control
At every instant the car observes (sensors + speed) and picks an action. Since the current
observation already contains what matters, a plain function `observation → action` is
enough — a neural network is a flexible way to represent it. No need to "remember" the past.

### 2.3 A neural network = a function with adjustable parameters
A *feedforward* MLP: multiply the input by a weight matrix, add a bias, apply a
non-linearity, repeat. The **weights** are the parameters; "training" = finding good weights.

### 2.4 Finding weights without a target → evolution
There is no "the correct output was 0.3" for each frame. The approach used here is a
**search by evolution** — generate many networks, measure how well each one does
(*fitness*), keep the best, generate variations. No gradient needed.

---

## 3. Tools and why

- **Python 3.11+** — single language.
- **NumPy** — network, geometry, vectorised simulation. Also runs in the browser (Pyodide).
- **Pygame-CE** — window, 2D drawing, input (track, network graph, panel).
- **pygbag** — packages the app as **WebAssembly** (static site).
- **uv** (environment) · **pytest** + **ruff** (tests and lint) · **reportlab** (PDF, optional).

Deliberately **left out**: **shapely/scipy** (the geometry fits in ~100 lines of NumPy and
runs in Pyodide); **PyTorch/TensorFlow** (the GA uses no gradient); **matplotlib** (the
curves are drawn with Pygame primitives).

---

## 4. The neural network

### 4.1 Shallow MLP (no memory)
The output depends **only** on the current input. Since the relevant state is in the
observation (the sensors), memory (an RNN) does not pay for the extra training difficulty —
and an MLP keeps the **neuron graph stable and readable**, which is the didactic goal.

### 4.2 Architecture
`n_inputs → hidden layer → 2 outputs`, with `tanh`.

- Inputs: `count` distances in `[0,1]` + `speed/v_max` (default 5 + 1 = 6).
- Hidden: `hidden` neurons (default 6), `tanh` (or `relu`).
- Outputs: 2 values in `[-1,1]` via `tanh` — **steering** and **throttle**.
- `hidden = 0` → the network is **linear**: good for showing that easy tracks need no
  hidden layer.

With 6 inputs and 6 hidden units there are **56 numbers** (weights + biases) — that vector
is what the GA optimises.

### 4.3 Implementation — `core/network.py`
- `MLP.__init__` creates `w1,b1,w2,b2` (or just `w1,b1` if `hidden=0`), weights
  `~ N(0, 1/sqrt(fan_in))`.
- `forward(x)`: `h = act(w1@x + b1)` ; `out = tanh(w2@h + b2)` ; stores `self.last = (x,h,out)`
  for the visualisation.
- `to_flat()` / `load_flat(vec)` — serialise the weights into a **1-D vector** (order
  `w1,b1,w2,b2`). A **genome IS that vector**.
- `n_params(n_in, cfg)` — the expected vector size.

### 4.4 Pros and cons
**Pros:** simple to understand and test; `forward` is cheap (you can simulate dozens of
cars in real time); no memory → stable graph; weights as a flat vector → crossover and
mutation are trivial.
**Cons:** it "anticipates" nothing; limited capacity (more neurons ⇒ slower evolution);
`tanh` saturates with large weights.

---

## 5. The genetic algorithm

### 5.1 The idea
1. **Population** — many random genomes (weight vectors).
2. **Evaluation** — each genome becomes an MLP, drives in the simulation, gets a **fitness**.
3. **Selection** — better genomes are more likely to "have children".
4. **Variation** — **crossover** (the child inherits genes from the parents) + **mutation**
   (Gaussian noise).
5. **Elitism** — the best pass through unchanged.
6. Repeat; fitness (best and mean) rises over the generations.

### 5.2 Why GA here
- There is no differentiable target → backprop does not apply directly.
- GA only needs "I can tell whether A did better than B": robust to noisy/discontinuous
  fitness, easy to visualise (the whole population trying) and naturally parallel.

### 5.3 Implementation — `core/evolution.py`
- `init_population(pop, n_params, rng)` → a `(pop, n_params)` matrix with `N(0, 0.5)`.
- `_tournament(fitness, k, rng)` → picks `k` genomes and returns the index of the best one
  (`k` controls the selection pressure).
- `next_population(...)`: copies the `elitism` best; for the rest, tournament → parent 1; if
  `crossover`, tournament → parent 2 and **uniform crossover** (50/50 per gene); **mutation**:
  with probability `mutation_rate` per gene, add `N(0, mutation_sigma)`.
- All randomness goes through a **seeded** `Generator` → same seed, same result.

### 5.4 The fitness function — `core/simulation.py`
```
fitness =  w_progress * progress_along_the_centre_line   # go far (counts full laps)
         + w_speed    * mean_normalised_speed            # go fast
         - w_smooth   * mean_steering_wiggle             # drive smoothly
         - crash_penalty * (crashed ? 1 : 0)             # do not crash
         + lap_bonus  * completed_laps                   # crossed the line
```
**Progress** is distance along the centre line (via the nearest point), which prevents
"cutting the corner". A stopped/spinning car is ended by a *stall* check.

### 5.5 Pros and cons
**Pros:** no gradient and no labelled data; easy to show; robust to a "broken" fitness;
parallelises for free.
**Cons:** **sample-inefficient** (each evaluation is a whole simulation); it "gropes" the
weight space; sensitive to `mutation_sigma`/`elitism`; can converge early to a local optimum.

---

## 6. Code walkthrough

Everything is in `src/`. Recommended reading order.

### 6.1 `core/config.py`
**Frozen** `dataclasses` with the hyperparameters: `NetworkCfg`, `SensorCfg`, `PhysicsCfg`,
`GACfg`, `FitnessCfg`, `SimCfg` (inside `Config`). `to_dict`/`from_dict` = JSON.

### 6.2 `core/geometry.py` — geometry and the raycast
- `resample_closed(pts, n)` — resamples the centre line into `n` points evenly spaced by arc.
- `offset_closed(pts, dist)` — shifts the centre line along each vertex normal → **edges**.
- `fan_angles(count, fov)` — the relative angles of the sensor fan.
- `ray_fan_distances(origin, dirs, seg_a, seg_b, max_dist)` — **the heart of the sensors**.
  For each ray (origin `O`, unit direction `d`) against each segment `A->B`:
  ```
  v1 = O - A     v2 = B - A     perp = (-d_y, d_x)
  t_ray = cross(v2, v1) / dot(v2, perp)     # distance along the ray
  t_seg = dot(v1, perp)  / dot(v2, perp)    # position on the segment (must be in [0,1])
  ```
  A hit counts if `dot(v2, perp) != 0`, `t_ray >= 0`, `0 <= t_seg <= 1`. The reading is the
  **smallest valid `t_ray`**, clamped to `max_dist`. Computed at once for `M` rays x `N`
  segments with NumPy broadcasting — no Python loop.

### 6.3 `core/track.py` — the track and the sensor readings
- `__init__`: resamples the centre line, builds `outer`/`inner` by offset (larger area =
  outer), concatenates the **wall segments** and the cumulative length table `cum`.
- `sensor_readings(pos, heading, cfg)` — directions `heading + fan_angles(...)` →
  `ray_fan_distances` against the walls → `distance / range` in `[0,1]`. This is the network
  input.
- `locate(pos)` — nearest point on the centre line + distance: used for **collision**
  (`> half_w + cushion`) and **progress**.
- `commit_progress(state, i)` — updates the index, detects the **lap wrap** and sets
  `finished`.
- Tracks are in `src/tracks/*.json` (`tools/make_track.py`: smooth polar curves).

### 6.4 `step_car` in `core/simulation.py` — kinematics
```
heading += steer * steer_rate * dt * grip   # grip ~ speed: no turning while stopped
speed   += throttle * accel * dt
speed   -= speed * friction
speed    = clip(speed, 0, v_max)
pos     += speed * dt * (cos heading, sin heading)
```

### 6.5 `core/network.py` — the MLP
See §4.3.

### 6.6 `core/simulation.py` — evaluating genomes
- `simulate(genome, track, cfg, record=False)` — **the readable version**: a Python loop
  step by step (sensors → `forward` → `step_car` → collision/progress → fitness). With
  `record=True` it records a `trace` (position, heading, activations per frame) for the
  animation.
- `simulate_population(genomes, track, cfg)` — **vectorised**: runs the population in
  lockstep (`einsum`); whoever crashes/stalls drops out of the `alive` mask. This is what
  training uses.
- Both share `step_car` and the `_score` formula; a test checks they agree.

### 6.7 `core/evolution.py`
See §5.3.

### 6.8 `core/trainer.py` — incremental, thread-free training
- `reset(cfg)` — seeds the RNG, loads the track, builds the population.
- `step(budget_ms)` — advances for the given time budget, evaluating the population in
  **sub-batches**; when the generation closes, `_finish_generation` records best/mean and
  breeds the next one. This is what lets training run **inside the render loop** (no
  freeze, no thread — essential for the web).
- `advance_generation()` — a whole generation at once. `best_trace()` — re-records the best
  genome's trace. `save()` — genome + config as JSON.

### 6.9 `app/gascreen.py` — this screen
Slider panel (hidden layer, sensors, population, mutation, elitism, fitness weights) +
buttons (Play/Pause, +1 Gen, Turbo, Apply, Reset, Track). **Apply** (key `a`) rebuilds the
config from the sliders and restarts training; it lights up while a slider differs from the
running config. Each frame: `trainer.step(...)`;
if a new best appeared, it recomputes the `trace` and the `MLP`; it draws the scene (track +
car + rays), the `netview` (nodes = activation, edges = signal) and the curve. The top bar
switches technique; **Tutorial** opens this text; **Back** returns here.

---

## 7. Suggested exercises

1. Set `Hidden layer = 0`, hit **Apply** on `circuito_1`: does it still drive? What about `circuito_3`?
2. `Mutation strength` ~0.6: the fitness curve becomes more erratic — why?
3. Zero out `Crash penalty`: what does the car start doing?
4. `Sensors = 9`: does it improve on tight corners?
5. In the code: swap uniform crossover for "single-point" in `evolution.py` and compare.
6. Add a 3rd output neuron ("brake") and adjust `network.py` + `step_car`.

---

## 8. References

- Backprop and MLPs: *Deep Learning* (Goodfellow, Bengio, Courville), ch. 6.
- Neuroevolution: Stanley & Miikkulainen, *NEAT*, 2002.
- Genetic algorithms: Melanie Mitchell, *An Introduction to Genetic Algorithms*.
- Raycast sensors + 2D car: the classic "self-driving car" JS demo series.
