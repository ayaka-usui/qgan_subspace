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
"""Generator module implemented in PyTorch."""

import itertools
import os

import torch
import torch.nn as nn

from config import CFG
from qgan.ansatz import XX_YY_ZZ_Z_circuit, ZZ_X_Z_circuit
from tools.qobjects.qcircuit import QuantumCircuit
from tools.qobjects.qgates import Identity, QuantumGate


class Generator(nn.Module):
    """Generator class for the Quantum GAN, implemented as a PyTorch Module."""

    def __init__(self, config=CFG):
        super().__init__()
        self.config = config
        self.target_size: int = self.config.system_size

        # Store config for loading/saving compatibility checks
        self.ancilla: bool = self.config.extra_ancilla
        self.ancilla_topology: str = self.config.ancilla_topology
        self.ansatz_type: str = self.config.gen_ansatz
        self.layers: int = self.config.gen_layers
        self.target_hamiltonian: str = self.config.target_hamiltonian

        # The circuit is now a submodule of the Generator
        if config.gen_ansatz == "ZZ_X_Z":
            self.ansatz = ZZ_X_Z_circuit(config=self.config)
        elif config.gen_ansatz == "XX_YY_ZZ_Z":
            self.ansatz = XX_YY_ZZ_Z_circuit(config=self.config)

    def forward(self, total_input_state: torch.Tensor) -> torch.Tensor:
        """
        The forward pass of the generator. It applies the quantum circuit
        to the input state.
        """
        return self.ansatz(total_input_state.flatten()).to(total_input_state.device)

    def load_model_params(self, file_path: str):
        """Loads generator parameters from a saved state_dict."""
        try:
            # Load the entire saved dictionary, which includes config
            saved_data = torch.load(file_path)
            saved_config = saved_data.get("config", {})

            # --- Perform compatibility checks ---
            if saved_config.get("target_size") != self.target_size:
                raise ValueError("Incompatible target size.")
            if saved_config.get("target_hamiltonian") != self.target_hamiltonian:
                raise ValueError("Incompatible target Hamiltonian.")
            if saved_config.get("ansatz_type") != self.ansatz_type:
                raise ValueError("Incompatible ansatz type.")
            if saved_config.get("layers") != self.layers:
                raise ValueError("Incompatible number of layers.")
            # TODO: Missing mismatch for adding ancilla

            # Load the state dictionary
            self.load_state_dict(saved_data["model_state_dict"])
            print(f"Generator parameters loaded successfully from {file_path}")

        except Exception as e:
            print(f"ERROR: Could not load generator model: {e}")

    def save_model_params(self, file_path: str):
        """Saves generator parameters and config to a file."""
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # Save both the model's state_dict and its configuration
        save_data = {
            "model_state_dict": self.state_dict(),
            "config": {
                "target_size": self.target_size,
                "target_hamiltonian": self.target_hamiltonian,
                "ansatz_type": self.ansatz_type,
                "layers": self.layers,
            },
        }
        torch.save(save_data, file_path)
        print(f"Generator model saved to {file_path}")
