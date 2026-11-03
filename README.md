### Experimental Neural Cellular Automata (NCA)

An experiment inspired by the Distill article “Growing Neural Cellular Automata” (Mordvintsev et al., 2020). Each pixel (cell) holds a small internal state and updates itself from local neighbourhood inputs using a shared, learned rule. With the right training, such systems can grow, persist, and regenerate target patterns from small seeds and survive damage.

- Reference: [Growing Neural Cellular Automata — Distill (2020)](https://distill.pub/2020/growing-ca/)

### What’s in this repo

- `NCA/model.py`
  - Keras/TensorFlow `CAModel` skeleton.
  - Uses `CHANNEL_N = 16` state channels.
  - Update rule is a 1×1 conv stack: `Conv2D(128, 1, relu) → Conv2D(16, 1, no activation)`, with the last layer’s kernel initialized to zeros.
  - Includes constants you’d expect in a training setup (e.g., `CELL_FIRE_RATE`, `TARGET_PADDING`, `TARGET_SIZE`, `POOL_SIZE`, etc.), but there is currently no training loop in this repository.
  - Intended to pair with a 3×3 perception stage (Sobel + identity) like in the Distill work, but that stage is not implemented here.

- `NCA/CA_test.py`
  - A forward-only, untrained demo built around `SimpleCA`.
  - Implements a fixed 3×3 perception filter bank with circular padding, followed by 1×1 conv layers.
  - Uses a stochastic update mask so only a random subset of cells update each step.
  - Loads `tst.npy` and runs 62 steps as a quick demo (no training).
  - Contains visualization helpers using Matplotlib.

- `NCA/sort_CA.py`
  - A small, separate 1D “sorting CA” experiment (not integrated with the 2D NCA above).
  - As written, it references an undefined variable `new_state` (see Known issues).

- Artifacts
  - `NCA/test.gif`, `NCA/video.mp4`: example animations.
  - `NCA/tst.npy`: sample multi-channel grid used by `CA_test.py`.

- Environment
  - `NCA/requirements.txt`: minimal Python dependencies for parts of the repo.

### Current status (honest overview)

- No training loop: there is no end-to-end training code, optimizer, or loss function wired up for `CAModel` or `SimpleCA`.
- No saved weights/checkpoints: nothing to load beyond the fixed filter bank in `CA_test.py`.
- No model export: no TF SavedModel, TF Lite, or TF.js export pipeline yet.
- Known issues:
  - `NCA/sort_CA.py` references an undefined variable `new_state`.
  - `NCA/requirements.txt` contains a typo `ipykernell` (should be `ipykernel`).
  - `NCA/CA_test.py` imports packages (e.g., `PyQt5`, `clip`) that are not listed in `requirements.txt`; these lines can be commented out or the packages installed if you use those parts.

### How to run what exists (local Python)

1) Create a virtual environment and install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r NCA/requirements.txt
# For the visualization path used in CA_test.py:
pip install matplotlib PyQt5
```

2) Run the forward-only demo (`CA_test.py`)

- Option A: run the script directly (if your environment supports GUI backends):
```bash
python NCA/CA_test.py
```
- Option B: import and run from a Python shell, which lets you skip optional imports:
```python
from NCA.CA_test import SimpleCA, np, tf, episode, render_episode
import numpy as np
import tensorflow as tf

ca = SimpleCA(8, zero_w2=False)
grid = tf.convert_to_tensor(np.load('NCA/tst.npy'))
arrays = episode(ca, grid, n=100, save_every=1)
render_episode(arrays)
```

Notes
- If you see an error from `import clip` or GUI backend issues, comment out those lines or make sure the dependency is installed and a display backend is available.
- This demo is untrained; it simply steps the CA forward from the provided `tst.npy` state.

### Next steps toward a trained, browser‑runnable NCA

- Training loop and sample pool
  - Implement a replay/sample pool of states; train with batches drawn from the pool so the CA learns to grow, persist, and regenerate.
  - Add damage augmentation (mask/cutout/erase regions) during training to encourage robustness/regeneration.
  - Use a stochastic update mask during training that matches the one used at inference (Cell Fire Rate).
- Perception stage
  - Compute a 3×3 filter bank per the Distill setup (e.g., identity + Sobel X/Y) to produce perceived channels.
  - Feed perceived channels into the 1×1 conv stack (e.g., 128 hidden → CHANNEL_N outputs with zero-initialized last layer).
- Alive masking
  - Maintain a dedicated “alive” channel and restrict updates/gradients to alive cells to stabilize training.
- Checkpoints and evaluation
  - Save/restore model weights; periodically evaluate survival/regeneration after randomized damage.
- Export to the web
  - Export to TF.js; optional 8‑bit quantization.
  - Expected weight size is small (on the order of tens of kilobytes; roughly 25–85 KB in float32 for typical NCA heads).
  - Build a browser demo that:
    - Uses the same perception filters and stochastic update mask as training.
    - Operates on a multi‑channel state.
    - Includes a simple “damage” tool to erase or perturb areas interactively.

### Context for later (legacy website note)

There was a temporary, hand‑coded, single‑channel “majority repair” stand‑in on `leopoldlacroix.github.io` under `/NCA`. That toy model was not this learned multi‑channel NCA. It is being removed until a trained model exists. Key mismatches vs. the target model here:
- Channels: 1 (stand‑in) vs. 16 (this repo’s target), or a reduced 4‑channel perception before the head.
- Perception: simple neighbor counting (stand‑in) vs. a proper 3×3 filter bank (identity + Sobel).
- Update rule: deterministic (stand‑in) vs. stochastic with a cell fire/update rate.

### References

- Mordvintsev, Niklasson, Randazzo. “Growing Neural Cellular Automata.” Distill, 2020. [distill.pub/2020/growing-ca](https://distill.pub/2020/growing-ca/)
- Related codebase from the original authors: [google-research/self-organising-systems](https://github.com/google-research/self-organising-systems)

