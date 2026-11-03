### Notes for maintainers

- `NCA/model.py`
  - `CAModel` is defined with a 1×1 conv head (128 hidden → `CHANNEL_N` outputs, zero‑initialized final kernel).
  - Constants including `CELL_FIRE_RATE`, `POOL_SIZE`, `TARGET_PADDING`, `TARGET_SIZE` are present but unused because there is no training harness yet.
  - Missing pieces to add: 3×3 perception (Sobel + identity), alive masking, optimizer/loss, and a training loop with a sample pool and damage augmentation.

- `NCA/CA_test.py`
  - Defines a fixed 3×3 filter bank with circular padding and a small 1×1 conv stack (`w1`, `w2`) inside `SimpleCA`.
  - Uses a stochastic update mask so that only some cells update at each step.
  - Demo runs for 62 steps from `tst.npy`; visualization uses Matplotlib. Optional imports (`PyQt5`, `clip`) can be removed or installed as needed.

- `NCA/sort_CA.py`
  - Prototype for a 1D “sorting CA” visualization using Plotly; currently references `new_state` without defining it, so it will raise at runtime.

- Artifacts
  - `test.gif`, `video.mp4` are example animations kept for reference.
  - `tst.npy` is a saved multi‑channel grid for the untrained forward demo.

- Dependencies
  - `NCA/requirements.txt` has a known typo: `ipykernell` → `ipykernel`.
  - `matplotlib`, `PyQt5`, and any optional extras (e.g., `clip`) are not listed; install them manually if you use those paths.

Planned milestones
- Add the 3×3 perception bank (identity + Sobel), alive masking, and the sample pool training loop.
- Implement damage augmentation and evaluation protocols (survival/regeneration after perturbation).
- Save checkpoints regularly and add a simple CLI to run/visualize from checkpoints.
- Export trained weights to TF.js with optional 8‑bit quantization and produce a small browser demo that mirrors training time behaviour (perception filters, stochastic update mask, multi‑channel state).

