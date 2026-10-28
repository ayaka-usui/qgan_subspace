# Copyright 2025 GIQ, Universitat Autònoma de Barcelona
#
# Licensed under the MIT License. See the LICENSE file in the project root for the full license text.
# SPDX-License-Identifier: MIT
"""the configuration for hamiltonian simulation task"""

from datetime import datetime
from typing import Any, Literal, Optional

import numpy as np


################################################################
# CONFIGURATION CLASS
################################################################
class Config:
    def __init__(self):
        """Configuration for the QGAN experiment, which sets up all parameters required for training it."""

        ############################################################################################
        # ---------------------
        # RUNS CONFIGURATION
        # ---------------------
        #       Whether to run a single or multiple experiments, and whether to just run different experiments from
        #       the start, or starting from a common configuration find a certain number of plateaus and then
        #       apply changes continuing the run from them (and not adding the changes for controls comparisons).
        #       Each individual experiment lasting the specified number of epochs and iterations in CFG.
        #
        #   - run_multiple_experiments: Whether to run multiple experiments.
        #
        #   - common_initial_plateaus: Whether to start from common initial plateaus + change, or all from scratch.
        #
        #     If True:
        #       + N_initial_plateaus: Number of initial plateaus to find, with default CFG (default: 5).
        #       + N_reps_each_init_plateau: Num of reps for each initial experiment afterwards, with changes (default: 20).
        #
        #     If False:
        #       + N_reps_if_from_scratch: Number of repetitions for each new configuration (default: 100).
        #
        #   - reps_new_config: The configuration changes to run. If starting from common initial plateaus,
        #         then this represent the changes after the initial plateaus (+ control with no changes).
        #         If not starting from common initial plateaus, then this represents the full set of experiments.
        #
        #   - parallel_runs: Whether the runs of each phase (initial plateaus, controls, each changed configuration, or
        #         the repetitions from scratch) are trained together, in one batched computation (False: one after
        #         the other). As many as fit in half of the free memory are trained together (all of them, at small
        #         sizes), and a run that stops is replaced by the next one. Each run gives the same results and files as
        #         when trained alone (equal up to rounding errors), except with compute_entanglement: numpy's random
        #         numbers (the negativities' random inputs, and with type_of_warm_start = "some" the re-drawn
        #         parameters) are then drawn in another order. Only with use_choi = True and dis_eval = "spectral".
        #
        #############################################################################################
        self.run_multiple_experiments: bool = True
        self.common_initial_plateaus: bool = True
        # If common_initial_plateaus == true:
        self.N_initial_plateaus: int = 5
        self.N_reps_each_init_plateau: int = 1
        # If common_initial_plateaus == false:
        self.N_reps_if_from_scratch: int = 20

        self.reps_new_config: list[dict[str, Any]] = [
            {
                "extra_ancilla": False,
                "ancilla_mode": "pass",
                "ancilla_topology": "bridge",
                "ancilla_connect_to": None,
                "do_ancilla_1q_gates": True,
                "start_ancilla_gates_randomly": True,
                "ancilla_coupling_layers": "all",
                "ancilla_training": True,
            },
            {
                "extra_ancilla": True,
                "ancilla_mode": "pass",
                "ancilla_topology": "bridge",
                "ancilla_connect_to": None,
                "do_ancilla_1q_gates": True,
                "start_ancilla_gates_randomly": True,
                "ancilla_coupling_layers": "all",
                "ancilla_training": True,
            },
            {
                "extra_ancilla": True,
                "ancilla_mode": "pass",
                "ancilla_topology": "total",
                "ancilla_connect_to": None,
                "do_ancilla_1q_gates": True,
                "start_ancilla_gates_randomly": True,
                "ancilla_coupling_layers": "all",
                "ancilla_training": True,
            },
            # Add more configs here for comparison
        ]
        self.parallel_runs: bool = False

        #############################################################################################
        # ---------------------
        # LOADING CONFIGURATION
        # ---------------------
        #   - load_timestamp: Timestamp to load a previous run (ex. None, 2025-06-06__02-05-10").
        #       + For individual runs, it will load the models from the specified timestamp.
        #       + For multiple runs, it will move to the directory from the specified timestamp,
        #         and append the new configurations (if common init, it will first check the CFG matches).
        #     Loads gen (thetas) and dis (alphas/betas), and the momentum of their optimizers (see below).
        #     Supports loading when adding or removing an ancilla (one qubit difference).
        #     WARNING: Only load trusted pickle files! Untrusted files may be insecure.
        #
        #   - type_of_warm_start: Warm start type for loading models (only if load_timestamp != None).
        #       + "none": No warm start.
        #       + "all": Warm start all parameters, by a bit (strength).
        #       + "some": Warm start some parameters (strength), to completely random.
        #
        #   - warm_start_strength: Strength of warm start for generator (only if loading).
        #
        #   - reset_optimizer_on_load: Whether the momentum of the optimizers restarts from zero when loading (True),
        #       or is restored from the saved models (False). Adding or removing an ancilla always restarts it, so with
        #       True the control runs restart it too (the same for both arms of a control-vs-ancilla comparison).
        #       (Continuing a multiple run made with False needs False: the configuration check compares it.)
        #
        #############################################################################################
        self.load_timestamp: Optional[str] = None  # "2025-07-10__18-57-21"
        self.type_of_warm_start: Literal["none", "all", "some"] = "none"
        self.warm_start_strength: Optional[float] = 0.1
        self.reset_optimizer_on_load: bool = True

        #############################################################################################
        # ----------------------
        # TRAINING CONFIGURATION
        # ----------------------
        #   - epochs: Number of training epochs (default: ~10)
        #
        #   - iterations_epoch: Number of iterations per epoch (default: ~100)
        #
        #   - save_fid_and_loss_every_x_iter: Saving fidelity and loss every x iterations (default: ~10)
        #
        #   - log_every_x_iter: Logging every x iterations (default: ~10)
        #           (needs to be a multiple of save_fid_and_loss_every_x_iter)
        #
        #   - max_fidelity: Stopping criterion for fidelity (default: ~0.99)
        #
        #   - compute_entanglement: Whether to compute the ancilla entanglement entropy at each iteration
        #       and display it in the live iteration plots (only meaningful when `extra_ancilla` is enabled),
        #       and all the qubit pairwise negativities (meaningful both for `extra_ancilla` enabled and disabled).
        #
        #   - negativity_n_states: Number of random product input states the negativities are averaged over.
        #
        #   - steps_gen/dis: Discriminator (first) and Generator (second) update steps in each iter (1~5).
        #
        #   - use_choi: Whether to train on the Choi state of the generator (True), or on batches of input states:
        #       + batch_mode: "haar" (batch_size random states each iteration), or "comp" (batch_size states each
        #         iteration from the basis states |j> and their pairwise superpositions, all d(d+1)/2 if larger).
        #       + batch_size: Number of input states per iteration.
        #       + eval_batch_size, eval_seed: Fixed set of random states the fidelity is evaluated on.
        #
        #   - precision: "float32" or "float64" (needed when exp(phi/lamb) overflows, for discriminators with
        #       large norms, e.g. random ones at >= 4 system qubits).
        #
        #############################################################################################
        self.epochs: int = 10
        self.iterations_epoch: int = 300
        self.save_fid_and_loss_every_x_iter: int = 20
        self.log_every_x_iter: int = 20  # This needs to be a multiple of save_fid_and_loss_every_x_iter
        self.max_fidelity: float = 0.99
        self.compute_entanglement: bool = False
        self.negativity_n_states: int = 50
        self.steps_dis: int = 1
        self.steps_gen: int = 1
        self.use_choi: bool = True
        self.batch_mode: Literal["haar", "comp"] = "comp"
        self.batch_size: int = 8
        self.eval_batch_size = 20
        self.eval_seed = 1234
        self.precision: Literal["float32", "float64"] = "float32"

        #############################################################################################
        # ---------------------
        # QUBITS CONFIGURATION
        # ---------------------
        #   - system_size: Number of qubits to study (excluding choi or ancilla), (default: 2-4)
        #
        #   - extra_ancilla: Whether to include an extra ancilla.
        #
        #   - ancilla_mode: What the discriminator sees, writing the output as |g> = |g_0>|0>_a + |g_1>|1>_a:
        #       + "pass": The full state |g>, compared with the target ⊗ |0>_a.
        #       + "project": |g_0>, the ancilla projected on |0>, normalised ("re-norm") or not ("pass"),
        #         as set in ancilla_project_norm.
        #       + "trace": The ancilla traced out, rho = |g_0><g_0| + |g_1><g_1|.
        #
        #   - ancilla_topology: Topology for the ancilla connections:
        #           |-----------------|-------------------|---------------------|----------------------|
        #           |  "disconnected" |      "ansatz"     |       "bridge"      |        "total"       |
        #     |-----|-----------------|-------------------|---------------------|----------------------|
        #     | Q0: |  ───|     |───  |  ───|     |─────  |  ───|     |──■────  |  ───|     |──■────── |
        #     | Q1: |  ───|  G  |───  |  ───|  G  |─────  |  ───|  G  |──│────  |  ───|  G  |──│─■──── |
        #     | Q2: |  ───|     |───  |  ───|     |──x──  |  ───|     |──│─x──  |  ───|     |──│─│─■── |
        #     |     |                 |              │    |              │ │    |              │ │ │   |
        #     | A:  |  ────U...U────  |  ────U...U───■──  |  ────U...U───■─■──  |  ────U...U───■─■─■── |
        #     |     |                 |                   |                     |                      |
        #     |     |       or        |         or        |           or        |          or          |
        #     |     |                 |                   |                     |                      |
        #     |  M  |                 |     Q0──Q1──Q2    |      Q0──Q1──Q2     |      Q0──Q1──Q2      |
        #     |  A  |  Q0──Q1──Q2  A  |           "x"|    |      │     "x"│     |      │   │   │       |
        #     |  P  |                 |     A────────     |      A────────      |      A────────       |
        #     |-----|-----------------|-------------------|---------------------|----------------------|
        #
        #       + "fake": The ancilla is not connected, and Q0 is connected to the "x" qubit instead (the ring).
        #
        #   - ancilla_connect_to: If ancilla_topology is "ansatz" or "bridge" connect to this qubit index
        #       (starting at 0). If None, then the ancilla is connected to the last qubit.
        #       (In the diagrams above, you would basically choose where that "x" connection goes in those)
        #
        #   - do_ancilla_1q_gates: Whether to include 1-qubit gates for the ancilla qubit.
        #       (The `U...U` gate in the previous diagrams).
        #
        #   - start_ancilla_gates_randomly: Whether to start the ancilla gates with random angles (if not, at 0).
        #       (The ancilla connections use the 2-qubit gates of the ansatz.)
        #
        #   - ancilla_coupling_layers: Layers with ancilla gates ("all", or a list of layer indices).
        #
        #   - ancilla_training: Whether the ancilla gates are trained (if not, they keep their initial angles).
        #
        ###############################################################################################
        self.system_size: int = 3
        self.extra_ancilla: bool = False  # We begin with no extra qubits, but we add once we reach the Plateau
        self.ancilla_mode: Optional[Literal["pass", "project", "trace"]] = "pass"
        self.ancilla_project_norm: Optional[Literal["re-norm", "pass"]] = "re-norm"
        self.ancilla_topology: Optional[Literal["disconnected", "ansatz", "bridge", "total", "fake"]] = "total"
        self.ancilla_connect_to: Optional[int] = None  # None means connected to last one, otherwise to the specified.
        self.do_ancilla_1q_gates: bool = True  # Whether to include 1-qubit gates for ancilla qubit.
        self.start_ancilla_gates_randomly: bool = True  # Whether to start ancilla gates with random angles.
        self.ancilla_coupling_layers: Literal["all"] | list[int] = "all"
        self.ancilla_training: bool = True

        #############################################################################################
        # -----------------------
        # GENERATOR CONFIGURATION
        # -----------------------
        #   - gen_layers: Number of layers in the generator ansatz (default: ~4)
        #
        #   - gen_ansatz: Ansatz type for generator:
        #       + "ZZ_YY_XX_Z": First 1 body Z terms, then, 2 body Z, 2 body Y and 2 body X terms.
        #       + "ZZ_Z_X": First 1 body X and 1 body Z terms, and then 2 body Z.
        #       + "custom": Custom ansatz, with two qubit and single qubit interactions in the given order.
        #
        #   - custom_ansatz_terms: Custom Ansatz terms (only apply if gen_ansatz is "custom").
        #       + "X", "Y", "Z", "XX", "ZZ", "YY": Available custom ansatz terms.
        #
        #   (In each layer, the outer loop is the type of gate, and the inner loop is the qubits they apply to).
        #   (Every gate is a rotation exp(-i θ/2 P): RX, RY, RZ, and IsingXX, IsingYY, IsingZZ).
        #
        #############################################################################################
        self.gen_layers: int = 1  # 2, 3, 5, 10, 20 ...
        self.gen_ansatz: Literal["ZZ_YY_XX_Z", "ZZ_Z_X", "custom"] = "ZZ_Z_X"
        self.custom_ansatz_terms: Optional[list[str]] = ["ZZ", "XX", "YY", "Z"]  # "X", "Y", "Z", "XX", "ZZ", "YY"

        #############################################################################################
        # ---------------------------
        # DISCRIMINATOR CONFIGURATION
        # ---------------------------
        #   - dis_type: Type of the discriminator observables psi and phi:
        #       + "local": Tensor products of 1 qubit observables, psi = ⊗_i (a_i0 I + a_i1 X + a_i2 Y + a_i3 Z).
        #       + "global": Sums over all the Pauli strings, psi = sum_P a_P P (any observable).
        #
        #   - dis_global_max_weight: Only for "global", keep only the Pauli strings acting on at most this number of
        #       qubits (None: all of them).
        #
        #   - dis_module_cap: Maximum norm of psi and phi. They are scaled down to it whenever they exceed it (when
        #       created or loaded, and after every update). This stops the rare runs whose random initial
        #       discriminator is so large that the updates blow up (to inf and NaN):
        #       + "auto": 6 * lamb * (4/3)^(system_size - 3), i.e. 60 for 3 qubits, 80 for 4 (lamb = 10). Above the
        #         norms that normal training reaches, so normal runs are not changed by it.
        #       + A number: that maximum.
        #       + None: no maximum (runs started with it can only be continued with it).
        #
        #   - dis_new_ancilla_row: Discriminator term on the ancilla, when it is added to a loaded run:
        #       + "random": Random, like the other qubits.
        #       + "identity": The identity (psi -> psi ⊗ I), so the cost right after adding it is the same.
        #
        #   - dis_ancilla_row_trainable: Whether that ancilla term is trained (if not, it keeps its initial value).
        #
        #   - dis_eval: How the terms exp(-phi/lamb) and exp(psi/lamb) of the cost are computed (same results):
        #       + "spectral": In the eigenbasis of psi and phi, known from their 1 qubit factors, so no 2^n x 2^n
        #         matrix is built (much faster, and the only option for large sizes). Only for "local".
        #       + "exponentiation": Exponentiating the 2^n x 2^n matrices (the original computation; any dis_type).
        #
        #############################################################################################
        self.dis_type: Literal["local", "global"] = "local"
        self.dis_global_max_weight: Optional[int] = None
        self.dis_module_cap: Optional[float | Literal["auto"]] = "auto"
        self.dis_new_ancilla_row: Literal["random", "identity"] = "random"
        self.dis_ancilla_row_trainable: bool = True
        self.dis_eval: Literal["spectral", "exponentiation"] = "spectral"

        #############################################################################################
        # ---------------------
        # TARGET CONFIGURATION
        # ---------------------
        #   - time_to_evolve: Time t to evolve with the Hamiltonian exp^(-iHt), for the target state preparation (default: 1.0).
        #
        #   - target_hamiltonian: Target Hamiltonian type:
        #       + "cluster_h": Cluster Hamiltonian (default).
        #       + "rotated_surface_h": Rotated surface code (only for squared sizes: 4, 9, 16...).
        #       + "ising_h": Ising Hamiltonian.
        #       + "custom_h": Custom Hamiltonian terms.
        #
        #   - custom_hamiltonian_terms: Custom Hamiltonian terms (only apply if target_hamiltonian is "custom_h").
        #       + "I", "X", "Y", "Z", "XX", "XZ", "ZZ", "ZZZ", "ZZZZ", "XZX", "XXXX": Available custom Hamiltonian terms.
        #
        #   - custom_hamiltonian_strengths: Strengths for the custom H terms (only apply if target_hamiltonian is "custom_h").
        #
        #############################################################################################
        self.time_to_evolve: float = 1.0  # Time to evolve with the Hamiltonian, for the target state preparation.
        self.target_hamiltonian: Literal["cluster_h", "rotated_surface_h", "ising_h", "custom_h"] = "custom_h"
        self.custom_hamiltonian_terms: Optional[list[str]] = [
            "ZZZ",
            "XXX",
        ]  # "I", "X", "Y", "Z", "XX", "XZ", "ZZZ", ...
        self.custom_hamiltonian_strengths: Optional[list[float]] = [1.0, 0.4]  # Strengths for the above terms.

        #############################################################################################
        # -----------------------------------
        # MOMENTUM OPTIMIZATION CONFIGURATION
        # -----------------------------------
        #   - l_rate: Learning rate for optimizers (default: 0.01)
        #   - momentum_coeff: Momentum coefficient for optimizers (default: 0.9)
        #   - dis_l_rate: Learning rate for the discriminator (None: l_rate). A "global" one may need a smaller one,
        #       since its steps on psi are 2^n times larger (l_rate / 8 was stable with 6 discriminator qubits).
        #
        #############################################################################################
        self.l_rate: float = 0.01
        self.momentum_coeff: float = 0.9
        self.dis_l_rate: Optional[float] = None

        #############################################################################################
        # ------------------------------------
        # HYPERPARAMETERS for Wasserstein Cost
        # ------------------------------------
        #  - lamb, s, cst1, cst2, cst3: Constants for Wasserstein cost and gradient.
        #
        #############################################################################################
        self.lamb = float(10)  # (change it with CFG.update, so that s and cst1-3 follow)
        self._derive()

        #############################################################################################
        # ----------------------------------
        # SAVING AND LOGGING CONFIGURATION
        # ---------------------------------
        #   - several paths for saving outputs.
        #
        #############################################################################################
        # Datetime for current run - initialized once
        self.run_timestamp: str = datetime.now().strftime("%Y-%m-%d__%H-%M-%S")
        self.base_data_path: str = f"./generated_data/{self.run_timestamp}"
        # File path settings (dynamic based on run_timestamp and system_size)
        self.set_results_paths()
        self._validate()

    # Fields computed from others (never set directly), and run-specific paths (not part of a configuration)
    _DERIVED = ("s", "cst1", "cst2", "cst3")
    _PATHS = (
        "run_timestamp",
        "base_data_path",
        "figure_path",
        "model_gen_path",
        "model_dis_path",
        "log_path",
        "fid_loss_path",
        "entropy_path",
        "gen_final_params_path",
    )

    def _derive(self) -> None:
        """Constants of the Wasserstein cost from lamb."""
        self.s = np.exp(-1 / (2 * self.lamb)) - 1
        self.cst1 = (self.s / 2 + 1) ** 2
        self.cst2 = (self.s / 2) * (self.s / 2 + 1)
        self.cst3 = (self.s / 2) ** 2

    def dis_cap(self) -> Optional[float]:
        """The maximum norm of psi and phi that dis_module_cap sets (None: no maximum)."""
        if self.dis_module_cap == "auto":
            return 6 * self.lamb * (4 / 3) ** (self.system_size - 3)
        return self.dis_module_cap

    def fields(self) -> list[str]:
        """Names of the settable fields (every public attribute except the derived ones)."""
        return [k for k in vars(self) if not k.startswith("_") and k not in self._DERIVED]

    def update(self, **kw) -> None:
        """Set fields by name (unknown names raise); derived constants and paths are recomputed."""
        unknown = sorted(k for k in kw if k not in self.fields())
        if unknown:
            raise KeyError(f"Unknown CFG field(s): {unknown}")
        for k, v in kw.items():
            setattr(self, k, v)
        self._derive()
        if "base_data_path" in kw:
            self.set_results_paths()
        self._validate()

    def as_dict(self) -> dict:
        """Settable fields except the run-specific paths (a configuration that reproduces this one)."""
        return {k: getattr(self, k) for k in self.fields() if k not in self._PATHS}

    def _validate(self) -> None:
        if self.log_every_x_iter % self.save_fid_and_loss_every_x_iter != 0:
            raise ValueError("log_every_x_iter must be a multiple of save_fid_and_loss_every_x_iter.")
        if self.dis_type not in ("local", "global"):
            raise ValueError(f"dis_type must be 'local' or 'global', got {self.dis_type!r}.")
        if self.dis_global_max_weight is not None and self.dis_type != "global":
            raise ValueError("dis_global_max_weight only applies to dis_type='global' (set it to None).")
        if self.dis_module_cap not in (None, "auto") and not (
            isinstance(self.dis_module_cap, (int, float)) and self.dis_module_cap > 0
        ):
            raise ValueError(f"dis_module_cap must be 'auto', None or > 0, got {self.dis_module_cap!r}.")
        if self.dis_eval not in ("spectral", "exponentiation"):
            raise ValueError(f"dis_eval must be 'spectral' or 'exponentiation', got {self.dis_eval!r}.")
        if self.dis_eval == "spectral" and self.dis_type != "local":
            raise ValueError("dis_eval='spectral' needs dis_type='local' (set dis_eval='exponentiation' for 'global').")
        if not isinstance(self.parallel_runs, (bool, np.bool_)):
            raise ValueError(f"parallel_runs must be True or False, got {self.parallel_runs!r}.")
        if self.parallel_runs and not (self.use_choi and self.dis_eval == "spectral"):
            raise ValueError("parallel_runs=True needs use_choi=True and dis_eval='spectral'.")
        if self.dis_l_rate is not None and not self.dis_l_rate > 0:
            raise ValueError(f"dis_l_rate must be None or > 0, got {self.dis_l_rate!r}.")

    def set_results_paths(self) -> None:
        """Set the paths for saving results based on the base data path."""
        base = self.base_data_path
        self.figure_path = f"{base}/figures"
        self.model_gen_path = f"{base}/saved_model/model-gen(hs).pkl"
        self.model_dis_path = f"{base}/saved_model/model-dis(hs).pkl"
        self.log_path = f"{base}/logs/log.txt"
        self.fid_loss_path = f"{base}/fidelities/log_fidelity_loss.txt"
        self.entropy_path = f"{base}/fidelities/log_entropy.txt"
        self.gen_final_params_path = f"{base}/gen_final_params/gen_final_params.txt"

    def negativity_path(self, pair: str) -> str:
        """Path of the negativity log for a qubit pair label such as '1-2' or '3-a'."""
        return f"{self.base_data_path}/fidelities/log_negativity_{pair.replace('-', '')}.txt"

    def show_data(self) -> str:
        """Return a string with the current configuration data."""
        sep = "─" * 50
        return (
            f"{'═' * 50}\n"
            f"run_timestamp: {self.run_timestamp}\n"
            f"{sep}\n"
            f"use_choi: {self.use_choi}\n"
            f"batch_size: {self.batch_size}\n"
            f"batch_mode:{self.batch_mode}\n"
            # (how the results are computed, not what: above type_of_warm_start, so that earlier logs still match in
            # training_init._check_for_previous_multiple_runs)
            f"dis_eval: {self.dis_eval}\n"
            f"parallel_runs: {self.parallel_runs}\n"
            f"{sep}\n"
            f"load_timestamp: {self.load_timestamp}\n"
            f"type_of_warm_start: {self.type_of_warm_start}\n"
            f"warm_start_strength: {self.warm_start_strength}\n"
            f"{sep}\n"
            f"system_size: {self.system_size}\n"
            f"extra_ancilla: {self.extra_ancilla}\n"
            f"ancilla_mode: {self.ancilla_mode}\n"
            f"ancilla_project_norm: {self.ancilla_project_norm}\n"
            f"ancilla_topology: {self.ancilla_topology}\n"
            f"ancilla_connect_to: {self.ancilla_connect_to}\n"
            f"do_ancilla_1q_gates: {self.do_ancilla_1q_gates}\n"
            f"start_ancilla_gates_randomly: {self.start_ancilla_gates_randomly}\n"
            f"ancilla_coupling_layers: {self.ancilla_coupling_layers}\n"
            f"ancilla_training: {self.ancilla_training}\n"
            f"{sep}\n"
            f"gen_layers: {self.gen_layers}\n"
            f"gen_ansatz: {self.gen_ansatz}\n"
            f"custom_ansatz_terms: {self.custom_ansatz_terms}\n"
            f"reset_optimizer_on_load: {self.reset_optimizer_on_load}\n"
            f"dis_new_ancilla_row: {self.dis_new_ancilla_row}\n"
            f"dis_ancilla_row_trainable: {self.dis_ancilla_row_trainable}\n"
            # printed only when set (dis_type other than "local", dis_module_cap other than None), so that the
            # logs of earlier runs, which have neither, still match in training_init._check_for_previous_multiple_runs
            + (f"dis_type: {self.dis_type}\ndis_global_max_weight: {self.dis_global_max_weight}\n"
               if self.dis_type != "local" else "")
            + (f"dis_module_cap: {self.dis_module_cap} ({self.dis_cap():g})\n"
               if self.dis_module_cap is not None else "")
            + f"precision: {self.precision}\n"
            f"{sep}\n"
            f"time_to_evolve: {self.time_to_evolve},\n"
            f"target_hamiltonian: {self.target_hamiltonian}\n"
            f"custom_hamiltonian_terms: {self.custom_hamiltonian_terms}\n"
            f"custom_hamiltonian_strengths: {self.custom_hamiltonian_strengths}\n"
            f"{sep}\n"
            f"epochs: {self.epochs}\n"
            f"iterations_epoch: {self.iterations_epoch}\n"
            f"log_every_x_iter: {self.log_every_x_iter}\n"
            f"save_fid_and_loss_every_x_iter: {self.save_fid_and_loss_every_x_iter}\n"
            f"max_fidelity: {self.max_fidelity}\n"
            f"steps_dis: {self.steps_dis}\n"
            f"steps_gen: {self.steps_gen}\n"
            f"compute_entanglement: {self.compute_entanglement}\n"
            f"negativity_n_states: {self.negativity_n_states}\n"
            f"{sep}\n"
            f"l_rate: {self.l_rate}\n"
            # printed only when set (no line = the shared l_rate)
            + (f"dis_l_rate: {self.dis_l_rate}\n" if self.dis_l_rate is not None else "")
            + f"momentum_coeff: {self.momentum_coeff}\n"
            f"lamb: {self.lamb}\n"
            f"{'═' * 50}\n"
        )


####################################################################
# Global instance of the Configuration class
####################################################################
CFG = Config()
