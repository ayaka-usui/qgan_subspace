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
"""Discriminator module implemented in PyTorch."""

import torch
import torch.nn as nn

from config import CFG
from tools.qobjects.qgates import COMPLEX_TYPE, I, X, Y, Z, device


class Discriminator(nn.Module):
    """
    Discriminator class for the Quantum GAN, implemented as a PyTorch Module.
    The parameters alpha and beta are learned automatically via AutoGrad.
    """

    def __init__(self, config=CFG):
        super().__init__()
        self.config = config
        # Determine the size of the Hilbert space the discriminator acts on
        self.size: int = self.config.system_size * 2 + (1 if self.config.extra_ancilla and self.config.ancilla_mode == "pass" else 0)

        # Store a list of Hermitian operators (Pauli matrices)
        self.herm: list[torch.Tensor] = [I, X, Y, Z]

        # Define alpha and beta as trainable parameters
        # They are initialized with random values between -1 and 1
        alpha_init = torch.empty(self.size, len(self.herm)).uniform_(-1, 1)
        beta_init = torch.empty(self.size, len(self.herm)).uniform_(-1, 1)
        self.alpha = nn.Parameter(alpha_init)
        self.beta = nn.Parameter(beta_init)

        # Store config for compatibility checks during model loading
        self.target_size: int = self.config.system_size
        self.target_hamiltonian: str = self.config.target_hamiltonian
        self.ancilla_mode: str = self.config.ancilla_mode

    def forward(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Constructs the psi and phi matrices from the alpha and beta parameters.
        This defines the forward pass of the discriminator model.
        """
        device = self.alpha.device
        psi = torch.tensor([[1.0 + 0.0j]], dtype=COMPLEX_TYPE, device=device)
        phi = torch.tensor([[1.0 + 0.0j]], dtype=COMPLEX_TYPE, device=device)

        for i in range(self.size):
            # Sum of Pauli matrices scaled by alpha/beta coefficients for the i-th qubit
            psi_i = sum(self.alpha[i, j] * self.herm[j] for j in range(len(self.herm)))
            phi_i = sum(self.beta[i, j] * self.herm[j] for j in range(len(self.herm)))

            # Build the full psi and phi matrices via Kronecker product
            psi = torch.kron(psi, psi_i)
            phi = torch.kron(phi, phi_i)

        return psi, phi

    def get_dis_matrices_rep(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes the matrices A and B from psi and phi. These are used in the cost function.
        """
        psi, phi = self.forward()

        # lamb is a hyperparameter from the config
        lamb = self.config.lamb

        # A = expm(-1/lamb * phi)
        A = torch.matrix_exp((-1.0 / lamb) * phi)
        # B = expm(1/lamb * psi)
        B = torch.matrix_exp((1.0 / lamb) * psi)

        return A, B, psi, phi

    def load_model_params(self, file_path: str):
        """Loads discriminator parameters from a saved state_dict."""

        try:
            # Load the entire saved dictionary, which includes config
            saved_data = torch.load(file_path, map_location=device)
            saved_config = saved_data.get("config", {})

            # --- Perform compatibility checks ---
            if saved_config.get("target_size") != self.target_size:
                raise ValueError("Incompatible target size.")
            if saved_config.get("target_hamiltonian") != self.target_hamiltonian:
                raise ValueError("Incompatible target Hamiltonian.")
            if saved_config.get("ancilla_mode") != self.ancilla_mode:
                raise ValueError("Incompatible ancilla mode.")
            # TODO: Missing mismatch for adding ancilla

            self.load_state_dict(saved_data["model_state_dict"])
            print(f"Discriminator parameters loaded successfully from {file_path}")

        except Exception as e:
            print(f"ERROR: Could not load discriminator model: {e}")

    def save_model_params(self, file_path: str):
        """Saves discriminator parameters and config to a file."""
        save_data = {
            "model_state_dict": self.state_dict(),
            "config": {
                "target_size": self.target_size,
                "target_hamiltonian": self.target_hamiltonian,
                "ancilla_mode": self.ancilla_mode,
            },
        }
        torch.save(save_data, file_path)
        print(f"Discriminator model saved to {file_path}")
