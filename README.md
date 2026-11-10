# qgan_subspace

Quantum Wasserstein GANs (QWGANs) that learn the time evolution $U_T = e^{-iH_Tt}$ of a few-qubit Hamiltonian
through its Choi state, with an optional **auxiliary qubit (ancilla)** coupled to the generator: from the start, or
inserted midway through training, once the fidelity has stalled on a plateau. The code is written in
[PennyLane](https://pennylane.ai) and [PyTorch](https://pytorch.org) and simulates the circuits exactly.

**Branches.** `pennylane-claude` (this branch) holds the PennyLane/PyTorch reimplementation, which is much faster and
generated the tutorial. `main` holds the original NumPy implementation, with its own version of the tutorial.

It is the code of

- A. Usui, G. Abad-López, H. SV, A. Sanpera, S. S. Bhattacharya, *Entanglement-assisted Hamiltonian dynamics
  learning* (2026), [arXiv:2602.15931](https://arxiv.org/abs/2602.15931).

**New here? Start with the notebook
[`tutorial.ipynb`](tutorial.ipynb).** It shows step by step how to obtain the main results
of the paper with this code, in similar but smaller settings: the auxiliary qubit inserted at a plateau or present
from the start, the entanglement and the gradients during training, and four system qubits. Along the way it
introduces the configuration, the training modes and the files the code writes.

## Installation

You need Python 3.10 or newer (tested with 3.12).

```bash
git clone -b pennylane-claude https://github.com/ayaka-usui/qgan_subspace.git
cd qgan_subspace
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install jupyter              # only for the notebook
```

The code runs on the CPU. Without a GPU, the CPU build of PyTorch is a much smaller download:
`pip install torch --index-url https://download.pytorch.org/whl/cpu` before installing the requirements.

## Running experiments

1. **Configure.** Every setting is a field of the `Config` class in [`src/config.py`](src/config.py), documented
   there: the target Hamiltonian, the generator ansatz and number of layers, the auxiliary qubit (`ancilla_*` fields:
   its topology, and what the critic sees of it, `ancilla_mode = "pass" | "trace" | "project"`; the paper uses
   `"pass"`, the auxiliary qubit returned and compared with $|0\rangle$, its Eq. (4)), the critic (local, as in the
   paper, or global), the optimiser, the number of epochs, and which experiment to run:
   - a single training (`run_multiple_experiments = False`);
   - several settings trained from scratch, `N_reps_if_from_scratch` times each (`common_initial_plateaus = False`,
     one setting per entry of `reps_new_config`);
   - the plateau protocol (`common_initial_plateaus = True`): `N_initial_plateaus` trainings that stall below
     `max_fidelity`, each continued without change (the control) and with each setting of `reps_new_config` (for
     instance, an auxiliary qubit inserted).

   The paper's configurations (a1)–(a4) are `ancilla_topology = "ansatz"`, `"bridge"` with `ancilla_connect_to = 1`,
   `"bridge"`, and `"total"` (see the tutorial, §1).

   With `parallel_runs = True`, the trainings of a setting are simulated together in one batch: the same results, many
   times faster.

2. **Run**, from the repository root (results are written relative to the current folder):

   ```bash
   python src/main.py
   ```

   Each run is saved to `generated_data/<timestamp>/`: its log (`logs/log.txt`, which starts with the full
   configuration), the fidelity and cost records (`fidelities/log_fidelity_loss.txt`, one column: the fidelities, then
   the costs), the trained models (`saved_model/`, from which `load_timestamp` continues a run), the final angles, the
   auxiliary-qubit diagnostics, and plots. A multiple run adds summary plots comparing its settings.

3. **Replot** a saved multiple run with your own labels: set `time_stamp_to_replot`, `run_names` and `x_label` in
   [`src/replot.py`](src/replot.py), then run `python src/replot.py`.

The code can also be driven from your own Python script or notebook: import `CFG` from `config`, change it with
`CFG.update(...)`, and call `run_single_training()` or `run_multiple_trainings()` from `tools.training_init`, with
`src/` on the path, as the notebook does. With `parallel_runs = True` the plots are drawn in separate processes, so a
script must guard its entry point with `if __name__ == "__main__":`. Notebooks need no guard.

## Repository layout

| path | contents |
|---|---|
| `src/config.py` | every setting, documented (the only file to edit for experiments) |
| `src/main.py` | entry point |
| `src/qgan/` | generator (PennyLane circuit), the critic (local, as in the paper, or global), targets, ancilla modes, cost, training loop, batched training of many runs |
| `src/tools/` | experiment protocols (`training_init.py`), loading and saving, plots |
| `src/replot.py` | plots of a saved run with custom labels |
| `src/spec_gen.py`, `src/trotter.py` | standalone scripts: gate counts of the generator on hardware, and a Trotter baseline for the same targets (not part of the paper's results) |
| `tutorial.ipynb` | the guided tutorial: how to obtain the paper's main results |

## Citation

If you use this code, please cite the paper above (A. Usui and G. Abad-López contributed equally).

## Acknowledgements and license

The code started from the repository [qWGAN](https://github.com/yiminghwang/qWGAN) of yiminghwang, which implements
the quantum Wasserstein GAN of Chakrabarti *et al.*, *Quantum Wasserstein Generative Adversarial Networks*, NeurIPS 32
(2019).

Distributed under the MIT License. See [LICENSE](LICENSE).
