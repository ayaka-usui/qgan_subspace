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
from typing import Literal, Optional

import numpy as np


################################################################
# CONFIGURATION CLASS
################################################################
class Config:
    def __init__(self):
        ########################################################################
        # CODE CONFIGURATION
        ########################################################################
        self.testing: bool = False  # True for testing mode, or False for single run
        self.load_timestamp: Optional[str] = None #"2025-06-06__02-05-10"

        #######################################################################
        # TRAINING CONFIGURATION
        #######################################################################
        self.epochs: int = 3 # Number of epochs for training (default: 10)
        self.iterations_epoch: int = 100  # Number of iterations per epoch (default: 100)
        self.log_every_x_iter: int = 1  # Log every x iterations (default: 10)
        self.max_fidelity: float = 0.99  # Maximum fidelity to reach, stopping criterion (default: 0.99)
        self.ratio_step_dis_to_gen: int = 1  # Ratio of Steps to train for discriminator to generator (Dis > Gen)

        #######################################################################
        # QUBIT SYSTEM CONFIGURATION
        #######################################################################
        self.system_size: int = 2  # Number of qubits to study (without choi or ancilla)
        self.extra_ancilla: bool = True  # If to include an extra ancilla: #True # False
        self.ancilla_mode: Optional[Literal["pass", "project", "trace_out"]] = "project"  # Ancilla mode from gen to dis
        self.ancilla_topology: Optional[Literal["ansatz", "bridge", "total"]] = "ansatz"  # Connectivity for the ancilla
        # TODO: Make handling of ancilla_mode more efficient, by never creating ancilla in Target.

        #######################################################################
        # GENERATOR CONFIGURATION
        #######################################################################
        self.gen_layers: int = 3  # Number of layers in the Generator ansatz: #20 #15 #10 #4 #3 #2 ...
        self.gen_ansatz: Literal["XX_YY_ZZ_Z", "ZZ_X_Z"] = "XX_YY_ZZ_Z"  # Ansatz for the Generator

        #######################################################################
        # TARGET CONFIGURATION
        #######################################################################
        self.target_hamiltonian: Literal["cluster_h", "rotated_surface_h", "custom_h"] = "cluster_h"
        self.custom_hamiltonian_terms: Optional[list[str]] = ["ZZZ"]  # Custom Terms: ["ZZZ", "ZZ", "Z", "I"]

        #####################################################################
        # SAVING AND LOGGING CONFIGURATION
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

        #######################################################################
        # OPTIMIZATION CONFIGURATION
        #######################################################################
        self.l_rate: float = 0.01  # Initial learning rate for optimizers (default: 0.01)
        self.momentum_coeff: float = 0.9  # Momentum coefficient for optimizers (default: 0.9)

        #####################################################################
        # HYPERPARAMETERS for Wasserstein Cost Function and Gradient
        #####################################################################
        self.lamb = float(10)
        self.s = np.exp(-1 / (2 * self.lamb)) - 1
        self.cst1 = (self.s / 2 + 1) ** 2
        self.cst2 = (self.s / 2) * (self.s / 2 + 1)
        self.cst3 = (self.s / 2) ** 2

    def show_data(self) -> str:
        """Return a dictionary with the current configuration data."""
        return (
            "================================================== \n"
            f"run_timestamp: {self.run_timestamp},\n"
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
    {
        "system_size": 2,
        "extra_ancilla": True,
        "ancilla_mode": "pass",
        "ancilla_topology": "ansatz",
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
        "ancilla_topology": "ansatz",
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
        "ancilla_mode": "trace_out",
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
    # Add more configurations as needed
]
