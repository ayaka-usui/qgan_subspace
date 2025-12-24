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
        #       Whether to run a single or multiple experiments, and whether to just run different experiments from the start 
        #       or starting from a common configuration find a certain number of plateaus and then
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
        #############################################################################################
        self.run_multiple_experiments: bool = True
        self.common_initial_plateaus: bool = True
        # If common_initial_plateaus == true:
        self.N_initial_plateaus: int = 100
        self.N_reps_each_init_plateau: int = 1
        # If common_initial_plateaus == false:
        self.N_reps_if_from_scratch: int = 100

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
            # {"extra_ancilla": True, "ancilla_topology": "bridge", "target_hamiltonian": "ising_h"},
            # Add more configs here for comparison
        ]

        #############################################################################################
        # ---------------------
        # LOADING CONFIGURATION
        # ---------------------
        #   - load_timestamp: Timestamp to load a previous run (ex. None, 2025-06-06__02-05-10").
        #       + For individual runs, it will load the models from the specified timestamp.
        #       + For multiple runs, it will move to the directory from the specified timestamp,
        #         and append the new configurations (if common init, it will first check the CFG matches).
        #     Loads gen (thetas) and dis (alphas/betas), but not the momentum from their optimizers.
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
        #############################################################################################
        self.load_timestamp: Optional[str] = None  # "2025-07-10__18-57-21"
        self.type_of_warm_start: Literal["none", "all", "some"] = "none"
        self.warm_start_strength: Optional[float] = 0.1

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
        #   - steps_gen/dis: Discriminator and Generator update steps in each iter (1~5).
        #
        #############################################################################################
        self.epochs: int = 10
        self.iterations_epoch: int = 300
        self.save_fid_and_loss_every_x_iter: int = 1
        self.log_every_x_iter: int = 10  # This needs to be a multiple of save_fid_and_loss_every_x_iter
        self.max_fidelity: float = 0.99
        self.steps_gen: int = 1
        self.steps_dis: int = 1

        #############################################################################################
        # ---------------------
        # QUBITS CONFIGURATION
        # ---------------------
        #   - system_size: Number of qubits to study (excluding choi or ancilla), (default: 2-4)
        #
        #   - extra_ancilla: Whether to include an extra ancilla.
        #
        #   - ancilla_mode: How ancilla is handled, between gen to dis.
        #       + "pass": Pass the ancilla qubit to the discriminator, after passes through gen.
        #       + "project": Project the ancilla qubit to the |0> state after gen (doesn't arrive to dis).
        #       + "trace": Trace out the ancilla qubit after gen (doesn't arrive to dis).
        #
        #   - ancilla_project_norm: How to handle the ancilla norm after projection:
        #       + "re-norm": Re-normalize after project (losses norm info, easier train, less effective Ham).
        #       + "pass": Pass state with its norm after project (keeps norm info, harder train, more effective Ham).
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
        #   - ancilla_connect_to: If ancilla_topology is "ansatz" or "bridge" connect to this qubit index
        #       (starting at 0). If None, then the ancilla is connected to the last qubit.
        #       (In the diagrams above, you would basically choose where that "x" connection goes in those)
        #
        #   - do_ancilla_1q_gates: Whether to include 1-qubit gates for the ancilla qubit.
        #       (The `U...U` gate in the previous diagrams).
        #
        #   - start_ancilla_gates_randomly: Whether to start the ancilla gates with random angles.
        #
        ###############################################################################################
        self.system_size: int = 3
        self.extra_ancilla: bool = False
        self.ancilla_mode: Optional[Literal["pass", "project", "trace"]] = "pass"
        self.ancilla_project_norm: Optional[Literal["re-norm", "pass"]] = "re-norm"
        self.ancilla_topology: Optional[Literal["disconnected", "ansatz", "bridge", "total", "fake"]] = "bridge"
        self.ancilla_connect_to: Optional[int] = None  # None means connected to last one, otherwise to the specified.
        self.do_ancilla_1q_gates: bool = True  # Whether to include 1-qubit gates for ancilla qubit.
        self.start_ancilla_gates_randomly: bool = True  # Whether to start ancilla gates with random angles.

        #############################################################################################
        # -----------------------
        # GENERATOR CONFIGURATION
        # -----------------------
        #   - gen_layers: Number of layers in the generator ansatz (default: ~4)
        #
        #   - gen_ansatz: Ansatz type for generator:
        #       + "XX_YY_ZZ_Z": 2 body X, 2 body Y, 2 body Z and 1 body Z terms.
        #       + "ZZ_X_Z": 2 body Z, 1 body X and 1 body Z terms.
        #
        #############################################################################################
        self.gen_layers: int = 3  # 2, 3, 5, 10, 20 ...
        self.gen_ansatz: Literal["XX_YY_ZZ_Z", "ZZ_X_Z"] = "ZZ_X_Z"

        #############################################################################################
        # ---------------------
        # TARGET CONFIGURATION
        # ---------------------
        #   - target_hamiltonian: Target Hamiltonian type:
        #       + "cluster_h": Cluster Hamiltonian (default).
        #       + "rotated_surface_h": Rotated surface code (only for squared sizes: 4, 9, 16...).
        #       + "ising_h": Ising Hamiltonian.
        #       + "custom_h": Custom Hamiltonian terms.
        #
        #   - custom_hamiltonian_terms: Custom Hamiltonian terms (only apply if target_hamiltonian is "custom_h").
        #       + "I", "X", "Y", "Z", "XX", "XZ", "ZZ", "ZZZ", "ZZZZ", "XZX", "XXXX": Available custom Hamiltonian terms.
        #
        #############################################################################################
        self.target_hamiltonian: Literal["cluster_h", "rotated_surface_h", "ising_h", "custom_h"] = "custom_h"
        self.custom_hamiltonian_terms: Optional[list[str]] = ["ZZZ"]  # "I", "X", "Y", "Z", "XX", "XZ", "ZZZ", ...
        self.custom_hamiltonian_strengths: Optional[list[float]] = [1.0]  # Strengths for the above terms.

        #############################################################################################
        # -----------------------------------
        # MOMENTUM OPTIMIZATION CONFIGURATION
        # -----------------------------------
        #   - l_rate: Learning rate for optimizers (default: 0.01)
        #   - momentum_coeff: Momentum coefficient for optimizers (default: 0.9)
        #
        #############################################################################################
        self.l_rate: float = 0.01
        self.momentum_coeff: float = 0.9

        #############################################################################################
        # ------------------------------------
        # HYPERPARAMETERS for Wasserstein Cost
        # ------------------------------------
        #  - lamb, s, cst1, cst2, cst3: Constants for Wasserstein cost and gradient.
        #
        #############################################################################################
        self.lamb = float(10)
        self.s = np.exp(-1 / (2 * self.lamb)) - 1
        self.cst1 = (self.s / 2 + 1) ** 2
        self.cst2 = (self.s / 2) * (self.s / 2 + 1)
        self.cst3 = (self.s / 2) ** 2

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
        self.__post_init_checks__()

    def __post_init_checks__(self) -> None:
        if self.log_every_x_iter % self.save_fid_and_loss_every_x_iter != 0:
            raise ValueError("log_every_x_iter must be a multiple of save_fid_and_loss_every_x_iter.")

    def set_results_paths(self) -> None:
        """Set the paths for saving results based on the base data path."""
        self.figure_path: str = f"{self.base_data_path}/figures"
        self.model_gen_path: str = f"{self.base_data_path}/saved_model/model-gen(hs).pkl"
        self.model_dis_path: str = f"{self.base_data_path}/saved_model/model-dis(hs).pkl"
        self.log_path: str = f"{self.base_data_path}/logs/log.txt"
        self.fid_loss_path: str = f"{self.base_data_path}/fidelities/log_fidelity_loss.txt"
        self.gen_final_params_path: str = f"{self.base_data_path}/gen_final_params/gen_final_params.txt"

    def show_data(self) -> str:
        """Return a dictionary with the current configuration data."""
        return (
            "================================================== \n"
            f"run_timestamp: {self.run_timestamp},\n"
            "----------------------------------------------\n"
            f"load_timestamp: {self.load_timestamp},\n"
            f"type_of_warm_start: {self.type_of_warm_start},\n"
            f"warm_start_strength: {self.warm_start_strength},\n"
            "----------------------------------------------\n"
            f"system_size: {self.system_size},\n"
            f"extra_ancilla: {self.extra_ancilla},\n"
            f"ancilla_mode: {self.ancilla_mode},\n"
            f"ancilla_project_norm: {self.ancilla_project_norm},\n"
            f"ancilla_topology: {self.ancilla_topology},\n"
            f"ancilla_connect_to: {self.ancilla_connect_to},\n"
            f"do_ancilla_1q_gates: {self.do_ancilla_1q_gates},\n"
            f"start_ancilla_gates_randomly: {self.start_ancilla_gates_randomly},\n"
            "----------------------------------------------\n"
            f"gen_layers: {self.gen_layers},\n"
            f"gen_ansatz: {self.gen_ansatz},\n"
            "----------------------------------------------\n"
            f"target_hamiltonian: {self.target_hamiltonian},\n"
            f"custom_hamiltonian_terms: {self.custom_hamiltonian_terms},\n"
            f"custom_hamiltonian_strengths: {self.custom_hamiltonian_strengths},\n"
            "----------------------------------------------\n"
            f"epochs: {self.epochs},\n"
            f"iterations_epoch: {self.iterations_epoch},\n"
            f"log_every_x_iter: {self.log_every_x_iter},\n"
            f"save_fid_and_loss_every_x_iter: {self.save_fid_and_loss_every_x_iter},\n"
            f"max_fidelity: {self.max_fidelity},\n"
            f"steps_gen: {self.steps_gen},\n"
            f"steps_dis: {self.steps_dis},\n"
            "----------------------------------------------\n"
            f"l_rate: {self.l_rate},\n"
            f"momentum_coeff: {self.momentum_coeff},\n"
            "================================================== \n"
        )


####################################################################
# Global instance of the Configuration class
####################################################################
CFG = Config()
