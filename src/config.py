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
No big changed from original """

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
        self.N_reps_if_from_scratch: int = 3

        # Configurations to compare (each dict overrides CFG attributes):
        self.reps_new_config: list[dict[str, Any]] = [
            {
                "extra_ancilla": True,
                "ancilla_mode": "pass",
                "ancilla_topology": "ansatz",
                "ancilla_connect_to": None,
                "do_ancilla_1q_gates": True,
                "start_ancilla_gates_randomly": False,
            },
            {
                "extra_ancilla": True,
                "ancilla_mode": "pass",
                "ancilla_topology": "bridge",
                "ancilla_connect_to": 1,
                "do_ancilla_1q_gates": True,
                "start_ancilla_gates_randomly": False,
            },
            {
                "extra_ancilla": True,
                "ancilla_mode": "pass",
                "ancilla_topology": "bridge",
                "ancilla_connect_to": None,
                "do_ancilla_1q_gates": False,
                "start_ancilla_gates_randomly": False,
            },
        ]

        # -- Loading and warm start ----------------------------
        # Load a previous run by timestamp. Supports \pm 1 qubit (ancilla add/remove).

        self.load_timestamp: Optional[str] = None # "2026-03-10__18-57-21"
        self.type_of_warm_start: Literal["none", "all", "some"] = "none"
        self.warm_start_strength: Optional[float] = 0.1

        # -- Training ------------------------------------------

        self.epochs: int = 10
        self.iterations_epoch: int = 300
        self.save_fid_and_loss_every_x_iter: int = 1
        self.log_every_x_iter: int = 10  # Must be a multiple of save_fid_and_loss_every_x_iter
        self.max_fidelity: float = 0.99  # Stop button 
        # In GANs, we can choose that the Discriminador learn faster than the Generator, or vice versa.
        self.steps_dis: int = 1
        self.steps_gen: int = 1

        # -- Qubits and ancilla ----------------------------
        #
        # Ancilla topologies: (Fig. 4.3 TFM Guille)
        #   disconnected : ancilla has 1q gates only, no 2q coupling
        #   ansatz       : ancilla couples to one system qubit (connect_to)
        #   bridge       : ancilla couples to qubit 0 AND connect_to
        #   total        : ancilla couples to ALL system qubits
        #   fake         : extra 2q gate between qubit 0 and connect_to (no actual ancilla coupling)
        #
        # Ancilla modes (post-generator, pre-discriminator):
        #   pass    : ancilla wire reaches the discriminator
        #   project : project ancilla to |0>, remove it
        #   trace   : trace out ancilla, sample pure state

        self.system_size: int = 3
        self.extra_ancilla: bool = False # We begin with no extra qubits, but we add once we reach the Plateau
        # Ancilla mode define what happens to the ancilla before Discriminator (ancilla.py)
        self.ancilla_mode: Optional[Literal["pass", "project", "trace"]] = "pass" 
        self.ancilla_project_norm: Optional[Literal["re-norm", "pass"]] = "re-norm"
        self.ancilla_topology: Optional[Literal["disconnected", "ansatz", "bridge", "total", "fake"]] = "bridge"
        self.ancilla_connect_to: Optional[int] = None
        self.do_ancilla_1q_gates: bool = True
        self.start_ancilla_gates_randomly: bool = True

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

        self.gen_layers: int = 3

        self.gen_ansatz: Literal["ZZ_YY_XX_Z", "ZZ_Z_X", "custom"] = "ZZ_Z_X"
        self.custom_ansatz_terms: Optional[list[str]] = ["ZZ", "XX", "Y", "X"]

        # -- Target Hamiltonian ----------------------------
        #
        # Predefined: cluster_h, ising_h, rotated_surface_h (squared qubits only)
        # Custom: specify terms and strengths.
        #   Available: I, X, Y, Z, XX, XZ, ZZ, ZZZ, ZZZZ, XZX, XXXX

        self.target_hamiltonian: Literal["cluster_h", "rotated_surface_h", "ising_h", "custom_h"] = "custom_h"
        self.custom_hamiltonian_terms: Optional[list[str]] = ["ZZZ"]
        self.custom_hamiltonian_strengths: Optional[list[float]] = [1.0]

        # -- Optimiser --------------------------------------
        self.l_rate: float = 0.01
        self.momentum_coeff: float = 0.9

        # -- Wasserstein cost hyperparameters ----------------------------
        self.lamb = float(10)
        self.s = np.exp(-1 / (2 * self.lamb)) - 1
        self.cst1 = (self.s / 2 + 1) ** 2
        self.cst2 = (self.s / 2) * (self.s / 2 + 1)
        self.cst3 = (self.s / 2) ** 2

        # -- Paths --------------------------------------------
        self.run_timestamp: str = datetime.now().strftime("%Y-%m-%d__%H-%M-%S")
        self.base_data_path: str = f"./generated_data/{self.run_timestamp}"
        self.set_results_paths()
        self._validate()

    
    def _validate(self) -> None:
        """Sanity check for every iter that we plot or save"""
        if self.log_every_x_iter % self.save_fid_and_loss_every_x_iter != 0:
            raise ValueError(
                "log_every_x_iter must be a multiple of save_fid_and_loss_every_x_iter."
            )

    def set_results_paths(self) -> None:
        """Update all output paths based on current base_data_path"""
        base = self.base_data_path
        self.figure_path = f"{base}/figures"
        self.model_gen_path = f"{base}/saved_model/model-gen(hs).pkl"
        self.model_dis_path = f"{base}/saved_model/model-dis(hs).pkl"
        self.log_path = f"{base}/logs/log.txt"
        self.fid_loss_path = f"{base}/fidelities/log_fidelity_loss.txt"
        self.gen_final_params_path = f"{base}/gen_final_params/gen_final_params.txt"

    def show_data(self) -> str:
        """Readable summary of current configuration"""
        sep = "─" * 50
        return (
            f"{'═' * 50}\n"
            f"run_timestamp: {self.run_timestamp}\n"
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
            f"{sep}\n"
            f"gen_layers: {self.gen_layers}\n"
            f"gen_ansatz: {self.gen_ansatz}\n"
            f"custom_ansatz_terms: {self.custom_ansatz_terms}\n"
            f"{sep}\n"
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
            f"{sep}\n"
            f"l_rate: {self.l_rate}\n"
            f"momentum_coeff: {self.momentum_coeff}\n"
            f"{'═' * 50}\n"
        )


CFG = Config()