# Copyright 2025 GIQ, Universitat Autònoma de Barcelona
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Configuration for the QGAN experiment
No big changed from original"""

from datetime import datetime
from typing import Any, Literal, Optional

import numpy as np


class Config:
    """Central configuration for all QGAN experiment parameters

    Single global instance (CFG) imported by all modules
    """

    def __init__(self):
        # -- Run mode ----------------------------
        # Single run or multiple experiments with optional shared initial plateaus.

        self.run_multiple_experiments: bool = True
        self.common_initial_plateaus: bool = True

        # If common_initial_plateaus:
        self.N_initial_plateaus: int = 5
        self.N_reps_each_init_plateau: int = 1

        # If not common_initial_plateaus:
        self.N_reps_if_from_scratch: int = 20

        # -- Training mode ----------------------------
        # use_choi: True = Choi representation, False = Haar random batching
        self.use_choi: bool = True
        self.batch_size: int = 8  # Only for Haar/Comp random 
        self.batch_mode: Literal["haar", "comp"] = "comp"

        self.eval_batch_size = 20
        self.eval_seed = 1234   

        # Configurations to compare (each dict overrides CFG attributes):
        self.reps_new_config: list[dict[str, Any]] = [
            {
                "extra_ancilla": False,
                "ancilla_mode": "pass",
                "ancilla_topology": "bridge",
                "ancilla_connect_to": None,
                "do_ancilla_1q_gates": True,
                "start_ancilla_gates_randomly": True,
                "ancilla_coupling_layers": "all",
                "ancilla_training": True
            },
            {
                "extra_ancilla": True,
                "ancilla_mode": "pass",
                "ancilla_topology": "bridge",
                "ancilla_connect_to": None,
                "do_ancilla_1q_gates": True,
                "start_ancilla_gates_randomly": True,
                "ancilla_coupling_layers": "all",
                "ancilla_training": True
            },
            {
                "extra_ancilla": True,
                "ancilla_mode": "pass",
                "ancilla_topology": "total",
                "ancilla_connect_to": None,
                "do_ancilla_1q_gates": True,
                "start_ancilla_gates_randomly": True,
                "ancilla_coupling_layers": "all",
                "ancilla_training": True
            },
        ]

        # -- Loading and warm start ----------------------------
        # Load a previous run by timestamp. Supports \pm 1 qubit (ancilla add/remove).
        self.load_timestamp: Optional[str] = None # -- "" <------------
        self.type_of_warm_start: Literal["none", "all", "some"] = "none"
        self.warm_start_strength: Optional[float] = 0.1

        # -- Training ------------------------------------------
        self.epochs: int = 10
        self.iterations_epoch: int = 300
        self.save_fid_and_loss_every_x_iter: int = 20
        self.log_every_x_iter: int = 20  # Must be a multiple of save_fid_and_loss_every_x_iter
        self.max_fidelity: float = 0.99  # Stop button
        # In GANs, we can choose that the Discriminador learn faster than the Generator, or vice versa.
        self.steps_dis: int = 1
        self.steps_gen: int = 1

        # -- Entanglement diagnostics (ported from upstream `entanglement_entropy` + `negativity`) --
        #   compute_entanglement: every `save_fid_and_loss_every_x_iter` iterations compute
        #     - the von Neumann entropy (log2) of the ancilla in the Choi output state (only if extra_ancilla),
        #     - the mean pairwise negativity of U_gen acting on `negativity_n_states` random product
        #       states (ancilla fixed to |0>), for every pair of generator qubits ("1-2", ..., "1-a", ...).
        self.compute_entanglement: bool = False
        self.negativity_n_states: int = 50

        # -- Qubits and ancilla ----------------------------
        #
        # Ancilla topologies: (Fig. 4.3 TFM Guille)
        #   disconnected : ancilla has 1q gates only, no 2q coupling
        #   ansatz       : ancilla couples to one system qubit (connect_to)
        #   bridge       : ancilla couples to qubit 0 AND connect_to
        #   total        : ancilla couples to ALL system qubits
        #   fake         : extra 2q gate between qubit 0 and connect_to (no actual ancilla coupling). With
        #                  do_ancilla_1q_gates=False and ancilla_mode="project" the idle extra qubit stays in |0>
        #                  and is removed (p0 = 1): the system line closed into a ring.
        #
        # Ancilla modes (post-generator, pre-discriminator), see qgan/ancilla.py. Write the generator output
        # as |g> = |g_0>|0>_a + |g_1>|1>_a (Choi: K_j0 = <j|U_gen|0>_a); the target is always t ⊗ |0>_a:
        #   pass    : the critic sees the full state |g> (one discriminator row on the ancilla)
        #   project : the critic sees |g_0> (post-selection on ancilla |0>; no ancilla row);
        #             ancilla_project_norm = "re-norm" -> |g_0>/sqrt(p0) (post-selected state, p0 = <g_0|g_0>)
        #                                    "pass"    -> |g_0> unnormalised (the cost then depends on p0)
        #   trace   : the critic sees rho = Tr_a |g><g| = |g_0><g_0| + |g_1><g_1| (no ancilla row; the cost is
        #             linear in rho, so it is evaluated exactly on the two branches; no sampling)
        self.system_size: int = 3
        self.extra_ancilla: bool = False # We begin with no extra qubits, but we add once we reach the Plateau
        self.gen_layers: int = 1
        # Ancilla mode define what happens to the ancilla before Discriminator (ancilla.py)
        self.ancilla_mode: Optional[Literal["pass", "project", "trace"]] = "pass"
        self.ancilla_project_norm: Optional[Literal["re-norm", "pass"]] = "re-norm"
        self.ancilla_topology: Optional[Literal["disconnected", "ansatz", "bridge", "total", "fake"]] = "total"
        self.ancilla_connect_to: Optional[int] = None
        self.do_ancilla_1q_gates: bool = True
        # 2q gate types of the ancilla couplings: None -> the 2q terms of the ansatz (default), or a sublist
        # such as ["ZZ"]. With ["ZZ"], do_ancilla_1q_gates=False and the ancilla in |0>, the ancilla never
        # leaves |0> (ZZ is diagonal on it): the circuit is the system ansatz + RZ on the coupled qubits.
        self.ancilla_coupling_terms: Optional[list[str]] = None
        self.start_ancilla_gates_randomly: bool = True
        # Ancilla angles at creation when start_ancilla_gates_randomly is False: 0 (None, default), or
        # N(0, scale^2) ("nudged" insertion).
        self.ancilla_init_angle_scale: Optional[float] = None
        # If all layer have 1q and 2q coupling gates or not
        self.ancilla_coupling_layers: Literal["all"] | list[int] = "all"
        # If False, ancilla params are randomly initialised but FROZEN,
        # autograd does not propagate the losvs
        self.ancilla_training: bool = True

        # -- Generator ansatz ----------------------------
        #
        # Per layer, outer loop = gate type, inner loop = qubits.
        #
        # Predefined:
        #   ZZ_YY_XX_Z : Z(1q) -> XX(2q) + YY(2q) + ZZ(2q)
        #   ZZ_Z_X     : X(1q) + Z(1q) -> ZZ(2q)
        #
        # Custom: specify gate order in custom_ansatz_terms.
        #   Available: "X", "Y", "Z", "XX", "YY", "ZZ"
        self.gen_ansatz: Literal["ZZ_YY_XX_Z", "ZZ_Z_X", "custom"] = "ZZ_Z_X"
        self.custom_ansatz_terms: Optional[list[str]] = ["ZZ", "XX", "YY", "Z"]

        # Gate convention of the 2-qubit rotations and order of the ancilla couplings:
        #   "uniform"  : every rotation is exp(-i θ/2 P) (RX, RY, RZ, IsingXX, IsingYY, IsingZZ), so all
        #                generators have the same norm and per-parameter gradients are comparable across
        #                gate types; ancilla couplings ordered gate-type-outer, connection-inner (like the
        #                system 2q gates: commuting terms grouped as in a Trotter step).          [default]
        #   "upstream" : identical circuit to the numpy code (ayaka-usui/qgan_subspace):
        #                XX(θ)=exp(-iθ XX), YY(θ)=exp(-iθ YY), ZZ(θ)=exp(+iθ/2 ZZ), gate-type-outer order.
        #   "fork"     : the PennyLane fork before the merge: exp(-i θ/2 P), connection-outer order.
        #   The convention is stored with the generator and checked when loading.
        self.gate_convention: Literal["uniform", "upstream", "fork"] = "uniform"

        # -- Loading / continuing from a plateau --
        #   reset_optimizer_on_load: False -> exact-match loads restore the saved SGD momentum (fork);
        #       True -> momentum restarts from zero (upstream). Partial loads (ancilla added/removed)
        #       always restart it, so False makes control and ancilla-added continuations asymmetric.
        #   dis_new_ancilla_row: discriminator factor on the qubit added when an ancilla is inserted:
        #       "random"   -> alpha, beta ~ U[-1,1] like any other row (current behaviour);
        #       "identity" -> alpha = beta = (1,0,0,0): the cost right after insertion (with zero ancilla
        #                     angles) equals the plateau cost, so changes come from the ancilla gates only.
        #   dis_gradient: "exact" -> autograd gradient of W w.r.t. alpha, beta (fork);
        #       "upstream" -> the numpy code's formula, which uses d e^{X} ~ e^{X} dX (ignores [X, dX] != 0).
        #   dis_ancilla_row_trainable (pass mode, ancilla present): False -> the critic's factor on the ancilla qubit
        #       keeps its value (its gradient is zeroed and the row restored after every step), e.g. the identity row of
        #       dis_new_ancilla_row = "identity" stays the identity ("frozen identity"); True -> trained like any other
        #       row (current behaviour).
        self.reset_optimizer_on_load: bool = False
        self.dis_new_ancilla_row: Literal["random", "identity"] = "random"
        self.dis_ancilla_row_trainable: bool = True
        self.dis_gradient: Literal["exact", "upstream"] = "exact"
        # -- Critic initialisation scale --
        #   None  -> alpha, beta ~ U[-1,1]^4 per row (current); the operator norms ||psi||, ||phi|| =
        #            prod_i (|c_i0| + |c_i|) then grow with the number of rows.
        #   kappa -> same random directions, every row rescaled so that ||psi|| = ||phi|| = kappa exactly.
        self.dis_init_scale: Optional[float] = None

        # -- Numerical precision: "float32" (complex64 states, default) or "float64" (complex128).
        #    float64 avoids the float32 overflow of exp(phi/lambda) for discriminators with large norms
        #    (e.g. random discriminators at >= 4 system qubits).
        self.precision: Literal["float32", "float64"] = "float32"

        # -- Target Hamiltonian ----------------------------
        #
        # Predefined: cluster_h, ising_h, rotated_surface_h (squared qubits only)
        # Custom: specify terms and strengths.
        #   Available: I, X, Y, Z, XX, XZ, ZZ, ZZZ, ZZZZ, XZX, XXXX
        self.time_to_evolve: float = 1.0  # Time to evolve with the Hamiltonian, for the target state preparation.
        self.target_hamiltonian: Literal["cluster_h", "rotated_surface_h", "ising_h", "custom_h"] = "custom_h"
        self.custom_hamiltonian_terms: Optional[list[str]] = ["ZZZ","XXX"]
        self.custom_hamiltonian_strengths: Optional[list[float]] = [1.0,0.4]
        # -- Optimiser --------------------------------------
        self.l_rate: float = 0.01
        self.momentum_coeff: float = 0.9

        # -- Wasserstein cost hyperparameters ----------------------------
        # (lamb must be set before qgan.discriminator is imported: it reads it at import)
        self.lamb = float(10)
        self._derive()

        # -- Paths --------------------------------------------
        self.run_timestamp: str = datetime.now().strftime("%Y-%m-%d__%H-%M-%S")
        self.base_data_path: str = f"./generated_data/{self.run_timestamp}"
        self.set_results_paths()
        self._validate()

    # Fields computed from others (never set directly) and run-specific paths (not part of a preset)
    _DERIVED = ("s", "cst1", "cst2", "cst3")
    _PATHS = ("run_timestamp", "base_data_path", "figure_path", "model_gen_path", "model_dis_path", "log_path",
              "fid_loss_path", "entropy_path", "gen_final_params_path")

    def _derive(self) -> None:
        """Constants of the regulariser from lamb (e^{-C/2λ} = (1-u) I + u SWAP)."""
        self.s = np.exp(-1 / (2 * self.lamb)) - 1
        self.cst1 = (self.s / 2 + 1) ** 2
        self.cst2 = (self.s / 2) * (self.s / 2 + 1)
        self.cst3 = (self.s / 2) ** 2

    def fields(self) -> list[str]:
        """Names of the settable fields (every public attribute except the derived ones)."""
        return [k for k in vars(self) if not k.startswith("_") and k not in self._DERIVED]

    def update(self, **kw) -> None:
        """Set fields by name (for scripts that copy their settings into CFG).

        Unknown names raise (a typo never passes silently); derived constants and paths are recomputed.
        """
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
        """Settable fields except the run-specific paths: a preset that reproduces this configuration."""
        return {k: getattr(self, k) for k in self.fields() if k not in self._PATHS}

    def _validate(self) -> None:
        """Sanity check for every iter that we plot or save"""
        if self.log_every_x_iter % self.save_fid_and_loss_every_x_iter != 0:
            raise ValueError("log_every_x_iter must be a multiple of save_fid_and_loss_every_x_iter.")

    def set_results_paths(self) -> None:
        """Update all output paths based on current base_data_path"""
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
        """Readable summary of current configuration"""
        sep = "─" * 50
        return (
            f"{'═' * 50}\n"
            f"run_timestamp: {self.run_timestamp}\n"
            f"{sep}\n"
            f"use_choi: {self.use_choi}\n"
            f"batch_size: {self.batch_size}\n"
            f"batch_mode:{self.batch_mode}\n"
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
            f"ancilla_coupling_terms: {self.ancilla_coupling_terms}\n"
            f"start_ancilla_gates_randomly: {self.start_ancilla_gates_randomly}\n"
            f"ancilla_init_angle_scale: {self.ancilla_init_angle_scale}\n"
            f"ancilla_coupling_layers: {self.ancilla_coupling_layers}\n"
            f"ancilla_training: {self.ancilla_training}\n"
            f"{sep}\n"
            f"gen_layers: {self.gen_layers}\n"
            f"gen_ansatz: {self.gen_ansatz}\n"
            f"custom_ansatz_terms: {self.custom_ansatz_terms}\n"
            f"gate_convention: {self.gate_convention}\n"
            f"reset_optimizer_on_load: {self.reset_optimizer_on_load}\n"
            f"dis_new_ancilla_row: {self.dis_new_ancilla_row}\n"
            f"dis_ancilla_row_trainable: {self.dis_ancilla_row_trainable}\n"
            f"dis_gradient: {self.dis_gradient}\n"
            f"dis_init_scale: {self.dis_init_scale}\n"
            f"precision: {self.precision}\n"
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
            f"momentum_coeff: {self.momentum_coeff}\n"
            f"lamb: {self.lamb}\n"
            f"{'═' * 50}\n"
        )


CFG = Config()
