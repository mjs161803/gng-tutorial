# gng-tutorial

A Python app that visualises a **Growing Neural Gas (GNG)** network as it
learns a dataset consisting of samples of a scalar (1-D) random variable.

---

## Features

| Feature | Detail |
|---|---|
| **Animated visualisation** | Matplotlib `FuncAnimation` shows the GNG growing in real-time |
| **Colour-coded neurons** | Each neuron is coloured by its *total accumulated lifetime* (steps since creation) using the `plasma` colourmap |
| **Controllable frame rate** | `--fps` sets the animation speed |
| **User-defined hyperparameters** | All GNG parameters are exposed as CLI arguments |
| **Dataset looping** | `--loop` replays the dataset indefinitely |

---

## Quickstart

```bash
pip install -r requirements.txt
python gng_app.py
```

Pass `--help` for a full list of options:

```
python gng_app.py --help
```

### Example: fast animation with looping

```bash
python gng_app.py --fps 30 --steps-per-frame 20 --loop
```

### Example: custom hyperparameters

```bash
python gng_app.py \
  --n-samples 8000 \
  --epsilon-b 0.3 \
  --epsilon-n 0.01 \
  --max-age 30 \
  --lam 50 \
  --alpha 0.6 \
  --d 0.999 \
  --fps 15 \
  --steps-per-frame 30
```

---

## CLI reference

| Argument | Default | Description |
|---|---|---|
| `--n-samples N` | `5000` | Number of training samples |
| `--seed N` | `42` | Random seed |
| `--epsilon-b F` | `0.2` | Winner learning rate (ε_b) |
| `--epsilon-n F` | `0.006` | Neighbour learning rate (ε_n) |
| `--max-age N` | `50` | Maximum edge age before removal |
| `--lam N` | `100` | Neuron-insertion interval λ |
| `--alpha F` | `0.5` | Error-reduction factor on insertion |
| `--d F` | `0.995` | Per-step error-decay factor |
| `--fps F` | `10.0` | Animation frame rate |
| `--steps-per-frame N` | `50` | Dataset steps between frames |
| `--loop` | off | Loop over dataset indefinitely |

---

## Project layout

```
gng_app.py       # Main app: CLI, dataset generation, animation
gng.py           # Pure GNG algorithm (no UI dependencies)
requirements.txt # numpy, matplotlib
tests/
  test_gng.py    # Unit tests (pytest)
```

---

## Running tests

```bash
pytest tests/
```

---

## Algorithm overview

The GNG algorithm (Fritzke 1995) maintains a graph of neurons, each with a
1-D position.  On every training step it:

1. Finds the two nearest neurons **s1** and **s2** to the input signal.
2. Increments the *lifetime* counter of all neurons.
3. Accumulates squared-distance error at **s1**.
4. Moves **s1** towards the signal by ε_b; moves its graph-neighbours by ε_n.
5. Ages every edge from **s1**; removes edges older than `max_age`; prunes
   isolated neurons.
6. Creates (or resets) the edge between **s1** and **s2**.
7. Every λ steps, inserts a new neuron between the two highest-error
   connected neurons.
8. Decays all accumulated errors by factor *d*.
