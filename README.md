# qgan_subspace

Quantum Wasserstein GANs (QWGANs) that learn the time evolution $U_T = e^{-iH_Tt}$ of a few-qubit Hamiltonian
through its Choi state, with an optional **auxiliary qubit (ancilla)** coupled to the generator: from the start, or
inserted midway through training, once the fidelity has stalled on a plateau.

This `main` branch keeps the original NumPy implementation, used for the paper by A. Usui, G. Abad-López, H. SV,
A. Sanpera and S. S. Bhattacharya, *Entanglement-assisted Hamiltonian dynamics learning* (2026),
[arXiv:2602.15931](https://arxiv.org/abs/2602.15931). The ancilla (the auxiliary qubit of the paper) is always
compared with $|0\rangle$, and the critic's gradient uses an approximation of the derivative of its matrix
exponentials. As a result, its training trajectories differ a bit from those of `pennylane-claude`.

> [!IMPORTANT]
> **Looking for the fastest implementation? Use the
> [`pennylane-claude`](https://github.com/ayaka-usui/qgan_subspace/tree/pennylane-claude) branch.**
>
> ```bash
> git clone -b pennylane-claude https://github.com/ayaka-usui/qgan_subspace.git
> ```
>
> Written in PennyLane and PyTorch, it computes the critic's gradients exactly and simulates many trainings together
> in one batch: roughly a hundred times faster than this branch.

**New here?** The notebook [`tutorial.ipynb`](tutorial.ipynb) shows step by step how to obtain results like the
paper's main ones. **Its saved outputs were generated with the `pennylane-claude` branch.** Run here, it trains
everything again with this branch's code, much more slowly: its first cell is set to a small demonstration of roughly
1.5 hours.

## Installation

You need Python 3.10 or newer (tested with 3.12).

```bash
git clone https://github.com/ayaka-usui/qgan_subspace.git
cd qgan_subspace
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install jupyter              # only for the notebook
```

## Running experiments

1. **Configure.** Every setting is a field of the `Config` class in [`src/config.py`](src/config.py), documented
   there: the target Hamiltonian, the generator ansatz and number of layers, the ancilla topology, the optimiser, the
   number of epochs, and which experiment to run:
   - a single training (`run_multiple_experiments = False`);
   - several settings trained from scratch, `N_reps_if_from_scratch` times each (`common_initial_plateaus = False`,
     one setting per entry of `reps_new_config`);
   - the plateau protocol (`common_initial_plateaus = True`): `N_initial_plateaus` trainings that stall below
     `max_fidelity`, each continued without change (the control) and with each setting of `reps_new_config`.
     The base configuration has `extra_ancilla = False`, so the plateaus are trained without the ancilla and each
     setting inserts it. A run with an ancilla cannot be continued in another ancilla topology.

2. **Run**, from the repository root (results are written relative to the current folder):

   ```bash
   python src/main.py
   ```

   Each run is saved to `generated_data/<timestamp>/`: its log (`logs/log.txt`, which starts with the full
   configuration), the fidelity and cost records (`fidelities/`), the trained models (`saved_model/`, from which
   `load_timestamp` continues a run), the final angles, and plots. A multiple run adds summary plots comparing its
   settings. Errors during a multiple run are written to its log.

3. **Replot** a saved multiple run with your own labels: set `time_stamp_to_replot`, `run_names` and `x_label` in
   [`src/replot.py`](src/replot.py), then run `python src/replot.py`.

## Citation

If you use this code, please cite the paper above (A. Usui and G. Abad-López contributed equally).

## Acknowledgements and license

The code started from the repository [qWGAN](https://github.com/yiminghwang/qWGAN) of yiminghwang, which implements
the quantum Wasserstein GAN of Chakrabarti *et al.*, *Quantum Wasserstein Generative Adversarial Networks*, NeurIPS 32
(2019).

Distributed under the MIT License. See [LICENSE](LICENSE).
