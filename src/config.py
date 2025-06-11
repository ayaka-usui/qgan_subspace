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

        ########################################################################
        # ---------------------
        # TESTING CONFIGURATION
        # ---------------------
        #   - testing: bool: Enables testing mode for running several different quick configurations.
        #
        ########################################################################
        self.testing: bool = False

        ########################################################################
        # ---------------------
        # RUNS CONFIGURATION
        # ---------------------
        #   - N_initial_exp: Number of initial experiments to run (without change), (default: 20).
        #   - N_reps_each_init_exp: Number of repetitions for each initial experiment afterwards (with changes), (default: 5).
        #
        ########################################################################
        self.N_initial_exp: int = 20  # TODO: For loop twice, first for initial experiments, second for repetitions.
        self.N_reps_each_init_exp: int = 5  # TODO: Change the ancilla mode and topology after the initial experiments.
        self.reps_new_config: dict[str, Any] = (
            {
                "extra_ancilla": True,
                "ancilla_mode": "pass",
                "ancilla_topology": "trivial",
                "type_of_warm_star": "all",
                "warm_start_strength": 0.1,
            },
        )
        # TODO: Also add so that automatically makes the analysis graphs of the improvements, and plots them in folder.

        #######################################################################
        # ---------------------
        # LOADING CONFIGURATION
        # ---------------------
        #   - load_timestamp: Timestamp to load a previous run (ex. None, 2025-06-06__02-05-10").
        #
        #   - type_of_warm_start: Warm start type for loading models (only if loading).
        #       + "none": No warm start.
        #       + "all": Warm start all parameters, by a bit (strength).
        #       + "some": Warm start some parameters (strength), to completely random.
        #
        #   - warm_start_strength: Strength of warm start for generator (only if loading).
        #
        ########################################################################
        self.load_timestamp: Optional[str] = None  # "2025-06-06__02-05-10"
        self.type_of_warm_start: Literal["none", "all", "some"] = "all"
        self.warm_start_strength: Optional[float] = 0.1

        #######################################################################
        # ----------------------
        # TRAINING CONFIGURATION
        # ----------------------
        #   - epochs: Number of training epochs (default: ~10)
        #
        #   - iterations_epoch: Number of iterations per epoch (default: ~100)
        #
        #   - log_every_x_iter: Logging every x iterations (default: ~10)
        #
        #   - max_fidelity: Stopping criterion for fidelity (default: ~0.99)
        #
        #   - ratio_step_dis_to_gen: Discriminator to generator training steps, ratio (dis > gen), (default: 1-5).
        #
        #######################################################################
        self.epochs: int = 2
        self.iterations_epoch: int = 20
        self.log_every_x_iter: int = 10
        self.max_fidelity: float = 0.99
        self.ratio_step_dis_to_gen: int = 1

        #######################################################################
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
        #   - ancilla_topology: Topology for the ancilla connections:
        #       |-----------------|-----------------|-----------------|-----------------------|------------------------|
        #       |    "trivial"    |  "disconnected" |     "ansatz"    |        "bridge"       |         "total"        |
        # |-----|-----------------|-----------------|-----------------|-----------------------|------------------------|
        # | Q0: |  ───|     |───  |  ───|     |───  |  ───|     |───  |  ───|     |───■─────  |  ───|     |───■─────── |
        # | Q1: |  ───|  G  |───  |  ───|  G  |───  |  ───|     |───  |  ───|  G  |───│─────  |  ───|  G  |───│─■───── |
        # | Q2: |  ───|     |───  |  ───|     |───  |  ───|  G  |───  |  ───|     |───│─■───  |  ───|     |───│─│─■─── |
        # |     |                 |                 |     |     |     |               │ │     |               │ │ │    |
        # | A:  |  ─────────────  |  ────X...X────  |  ───|     |───  |  ────X...X────■─■───  |  ────X...X────■─■─■─── |
        # |     |                 |                 |                 |                       |                        |
        # |     |        or       |       or        |         or      |          or           |          or            |
        # |     |                 |                 |                 |                       |                        |
        # |  M  |                 |                 |                 |      Q0──Q1──Q2       |      Q0──Q1──Q2        |
        # |  A  |  Q0──Q1──Q2  A  |  Q0──Q1──Q2  A  |  Q0──Q1──Q2──A  |      │       │        |      │   │   │         |
        # |  P  |                 |                 |                 |      A────────        |      A────────         |
        # |-----|-----------------|-----------------|-----------------|-----------------------|------------------------|
        #
        #######################################################################
        self.system_size: int = 2
        self.extra_ancilla: bool = True
        self.ancilla_mode: Optional[Literal["pass", "project", "trace"]] = "project"
        self.ancilla_project_norm: Optional[Literal["remove", "penalize", "pass"]] = "remove"  # TODO: Document this.
        self.ancilla_norm_penalization: float = 0.01  # TODO: Document this, and add it to where needed in config.
        # TODO: [FUTURE] Decide what to do with trace (work with rho, instead than sampling? Keep both?).
        self.ancilla_topology: Optional[Literal["trivial", "disconnected", "ansatz", "bridge", "total"]] = "total"
        # TODO: [URGENT] Try "project" and "total" and see if you observe any difference in the results, to Ayaka.

        #######################################################################
        # -----------------------
        # GENERATOR CONFIGURATION
        # -----------------------
        #   - gen_layers: Number of layers in the generator ansatz (default: ~4)
        #
        #   - gen_ansatz: Ansatz type for generator:
        #       + "XX_YY_ZZ_Z": 2 body X, 2 body Y, 2 body Z and 1 body Z terms.
        #       + "ZZ_X_Z": 2 body Z, 1 body X and 1 body Z terms.
        #
        #######################################################################
        self.gen_layers: int = 1  # 20 #15 #10 #4 #3 #2 ...
        self.gen_ansatz: Literal["XX_YY_ZZ_Z", "ZZ_X_Z"] = "XX_YY_ZZ_Z"

        #######################################################################
        # ---------------------
        # TARGET CONFIGURATION
        # ---------------------
        #   - target_hamiltonian: Target Hamiltonian type:
        #       + "cluster_h": Cluster Hamiltonian (default).
        #       + "rotated_surface_h": Rotated surface code (only for squared sizes: 4, 9, 16...).
        #       + "custom_h": Custom Hamiltonian terms.
        #
        #   - custom_hamiltonian_terms: Custom Hamiltonian terms (only apply if target_hamiltonian is "custom_h").
        #       + "ZZZ": Adds a 3 body Z term.
        #       + "ZZ": Adds a 2 body Z term.
        #       + "Z": Adds a 1 body Z term.
        #       + "I": Adds a 1 body identity term.
        #
        #######################################################################
        self.target_hamiltonian: Literal["cluster_h", "rotated_surface_h", "custom_h"] = "cluster_h"
        self.custom_hamiltonian_terms: Optional[list[str]] = ["ZZZ"]  # Custom Terms: ["ZZZ", "ZZ", "Z", "I"]

        #######################################################################
        # -----------------------------------
        # MOMENTUM OPTIMIZATION CONFIGURATION
        # -----------------------------------
        #   - l_rate: Learning rate for optimizers (default: 0.01)
        #   - momentum_coeff: Momentum coefficient for optimizers (default: 0.9)
        #
        #######################################################################
        self.l_rate: float = 0.01
        self.momentum_coeff: float = 0.9

        #####################################################################
        # ------------------------------------
        # HYPERPARAMETERS for Wasserstein Cost
        # ------------------------------------
        #  - lamb, s, cst1, cst2, cst3: Constants for Wasserstein cost and gradient.
        #
        #####################################################################
        self.lamb = float(10)
        self.s = np.exp(-1 / (2 * self.lamb)) - 1
        self.cst1 = (self.s / 2 + 1) ** 2
        self.cst2 = (self.s / 2) * (self.s / 2 + 1)
        self.cst3 = (self.s / 2) ** 2

        #####################################################################
        # ----------------------------------
        # SAVING AND LOGGING CONFIGURATION
        # ---------------------------------
        #   - several paths for saving outputs.
        #
        #####################################################################
        # Datetime for current run - initialized once
        self.run_timestamp: str = datetime.now().strftime("%Y-%m-%d__%H-%M-%S")
        self.base_data_path: str = f"./generated_data/{'TESTING/' if self.testing else ''}{self.run_timestamp}"

        # File path settings (dynamic based on run_timestamp and system_size)
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
            f"load_timestamp: {self.load_timestamp},\n"
            f"type_of_warm_start: {self.type_of_warm_start},\n"
            f"warm_start_strength: {self.warm_start_strength},\n"
            f"system_size: {self.system_size},\n"
            f"extra_ancilla: {self.extra_ancilla},\n"
            f"ancilla_mode: {self.ancilla_mode},\n"
            f"ancilla_topology: {self.ancilla_topology},\n"
            f"gen_layers: {self.gen_layers},\n"
            f"gen_ansatz: {self.gen_ansatz},\n"
            f"target_hamiltonian: {self.target_hamiltonian},\n"
            f"custom_hamiltonian_terms: {self.custom_hamiltonian_terms},\n"
            f"epochs: {self.epochs},\n"
            f"iterations_epoch: {self.iterations_epoch},\n"
            f"log_every_x_iter: {self.log_every_x_iter},\n"
            f"max_fidelity: {self.max_fidelity},\n"
            f"l_rate: {self.l_rate},\n"
            f"ratio_step_dis_to_gen: {self.ratio_step_dis_to_gen},\n"
            "================================================== \n"
        )


####################################################################
# Global instance of the Configuration class
####################################################################
CFG = Config()


#####################################################################
# Test Configurations
#####################################################################
test_configurations = [
    # Testing Custom Hamiltonian ZZZ
    {
        "system_size": 2,
        "extra_ancilla": False,
        "epochs": 1,
        "iterations_epoch": 3,
        "gen_layers": 1,
        "gen_ansatz": "XX_YY_ZZ_Z",
        "target_hamiltonian": "custom_h",
        "custom_hamiltonian_terms": ["ZZZ"],
        "label_suffix": "c1_2q_1l_noanc_XXYYZZZ_CustomH_ZZZ",
    },
    # Testing extra ancilla with different modes and topologies
    {
        "system_size": 2,
        "extra_ancilla": True,
        "ancilla_mode": "pass",
        "ancilla_topology": "trivial",
        "epochs": 1,
        "iterations_epoch": 3,
        "gen_layers": 1,
        "gen_ansatz": "ZZ_X_Z",
        "target_hamiltonian": "cluster_h",
        "label_suffix": "c2_2q_1l_anc_pass_ZZXZ_ClusterH",
    },
    {
        "system_size": 2,
        "extra_ancilla": True,
        "ancilla_mode": "project",
        "ancilla_topology": "disconnected",
        "epochs": 2,
        "iterations_epoch": 2,
        "gen_layers": 2,
        "gen_ansatz": "XX_YY_ZZ_Z",
        "target_hamiltonian": "cluster_h",
        "label_suffix": "c3_2q_2l_anc_project_XXYYZZZ_ClusterH",
    },
    {
        "system_size": 3,
        "extra_ancilla": True,
        "ancilla_mode": "trace",
        "ancilla_topology": "ansatz",
        "epochs": 1,
        "iterations_epoch": 3,
        "gen_layers": 1,
        "gen_ansatz": "ZZ_X_Z",
        "target_hamiltonian": "cluster_h",
        "label_suffix": "c4_3q_1l_anc_trace_ZZXZ_ClusterH",
    },
    {
        "system_size": 2,
        "extra_ancilla": True,
        "ancilla_mode": "pass",
        "ancilla_topology": "bridge",
        "epochs": 1,
        "iterations_epoch": 3,
        "gen_layers": 1,
        "gen_ansatz": "ZZ_X_Z",
        "target_hamiltonian": "cluster_h",
        "label_suffix": "c5_2q_1l_anc_pass_total_ZZXZ_ClusterH",
    },
    # Testing rotated surface code Hamiltonian
    {
        "system_size": 4,
        "extra_ancilla": True,
        "ancilla_mode": "project",
        "ancilla_topology": "total",
        "epochs": 1,
        "iterations_epoch": 3,
        "gen_layers": 1,
        "gen_ansatz": "ZZ_X_Z",
        "target_hamiltonian": "rotated_surface_h",
        "label_suffix": "c6_2q_1l_anc_project_total_ZZXZ_RotatedSurfaceH",
    },
    # Testing model loading:
    {
        "system_size": 2,
        "extra_ancilla": False,
        "load_timestamp": "2025-06-10__00-53-18",
        "type_of_warm_start": "none",
        "warm_start_strength": 0.1,
        "epochs": 1,
        "iterations_epoch": 3,
        "gen_layers": 1,
        "gen_ansatz": "XX_YY_ZZ_Z",
        "target_hamiltonian": "cluster_h",
        "label_suffix": "c7_2q_noanc_load_warmstart_XXYYZZZ_ClusterH",
    },
    # Testing model loading with warm start:
    {
        "system_size": 2,
        "extra_ancilla": False,
        "load_timestamp": "2025-06-10__00-53-18",
        "type_of_warm_start": "all",
        "warm_start_strength": 0.1,
        "epochs": 1,
        "iterations_epoch": 3,
        "gen_layers": 1,
        "gen_ansatz": "XX_YY_ZZ_Z",
        "target_hamiltonian": "cluster_h",
        "label_suffix": "c8_2q_noanc_load_warmstart_all_XXYYZZZ_ClusterH",
    },
    {
        "system_size": 2,
        "extra_ancilla": True,
        "ancilla_mode": "pass",
        "ancilla_topology": "ansatz",
        "load_timestamp": "2025-06-10__00-53-18",
        "type_of_warm_start": "some",
        "warm_start_strength": 0.1,
        "epochs": 1,
        "iterations_epoch": 3,
        "gen_layers": 1,
        "gen_ansatz": "XX_YY_ZZ_Z",
        "target_hamiltonian": "cluster_h",
        "label_suffix": "c9_2q_anc_pass_load_warmstart_some_XXYYZZZ_ClusterH",
    },
    # Add more configurations as needed
]
