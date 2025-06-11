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
"""Discriminator module"""

import os
import pickle

import numpy as np
from scipy.linalg import expm

from config import CFG
from qgan.cost_functions import get_final_comp_states_for_dis
from tools.data.data_managers import print_and_train_log
from tools.optimizer import MomentumOptimizer
from tools.qobjects.qgates import I, X, Y, Z

cst1, cst2, cst3, lamb = CFG.cst1, CFG.cst2, CFG.cst3, CFG.lamb
cs = -1 / lamb


class Discriminator:
    """Discriminator class for the Quantum GAN.

    Relevant parameters/matrices:

        psi: Real part of the discriminator (matrix).
        phi: Imaginary part of the discriminator (matrix).

        A: -1/lamb * phi
        B: 1/lamb * psi

        alpha: Parameters for the real part of the discriminator (coeff for I, X, Y, Z).
        beta: Parameters for the imaginary part of the discriminator (coeff for Ij, Xj, Yj, Zj).

    """

    def __init__(self):
        # Set the general used parameters:
        self.size: int = CFG.system_size * 2 + (1 if CFG.extra_ancilla and CFG.ancilla_mode == "pass" else 0)
        self.herm: list = [I, X, Y, Z]
        self.alpha: np.ndarray = np.zeros((self.size, len(self.herm)))
        self.beta: np.ndarray = np.zeros((self.size, len(self.herm)))
        self._init_params()
        self.optimizer_psi: MomentumOptimizer = MomentumOptimizer()
        self.optimizer_phi: MomentumOptimizer = MomentumOptimizer()

        # Set params, for comparison when loading:
        self.ancilla: bool = CFG.extra_ancilla
        self.ancilla_mode: str = CFG.ancilla_mode  # Topology doesn't matter, its not a circuit = fully connect matrix.
        self.target_size: str = CFG.system_size

    def _init_params(self):
        # Random Discriminator Parameters, each param is: (size x 4)
        for i in range(self.size):
            self.alpha[i] = -1 + 2 * np.random.random(len(self.herm))
            self.beta[i] = -1 + 2 * np.random.random(len(self.herm))

    def getPsiPhi(self) -> np.ndarray:
        """Get matrix representation of real (psi) & imaginary (phi) part of the discriminator

        Size of psi/phi (ndarray) = [num_qubit, 4]
            0: I,   1: X,   2: Y,   3: Z

        Returns:
            tuple[np.ndarray]: Tuple of Psi and Phi, matrix representations of real and imaginary part of discriminator.
        """
        psi, phi = 1, 1
        for i in range(self.size):
            psi_i, phi_i = np.zeros_like(self.herm[0], dtype=complex), np.zeros_like(self.herm[0], dtype=complex)
            for j in range(len(self.herm)):
                psi_i += self.alpha[i][j] * self.herm[j]
                phi_i += self.beta[i][j] * self.herm[j]
            psi, phi = np.kron(psi, psi_i), np.kron(phi, phi_i)
        return psi, phi

    def update_dis(self, gen, total_real_state: np.ndarray, total_input_state: np.ndarray):
        """Update the discriminator parameters (alpha, beta) using the gradients.

        Args:
            gen (Generator): The generator object.
            total_real_state (np.ndarray): The total real state of the system.
            total_input_state (np.ndarray): The total input state of the system.
        """
        ####################################################
        # Update alpha
        ####################################################
        grad_alpha = self._compute_grad(gen, total_real_state, total_input_state, "alpha")
        new_alpha = self.optimizer_psi.move_in_grad(self.alpha, grad_alpha, "max")

        ####################################################
        # Update beta
        ####################################################
        grad_beta = self._compute_grad(gen, total_real_state, total_input_state, "beta")
        new_beta = self.optimizer_phi.move_in_grad(self.beta, grad_beta, "max")

        # Update the parameters later, to avoid affecting gradient computations:
        self.alpha = new_alpha
        self.beta = new_beta

    def _compute_grad(self, gen, total_real_state: np.ndarray, total_input_state: np.ndarray, param: str) -> np.ndarray:
        """Calculate the gradient of the discriminator with respect to the param (alpha or beta).

        Args:
            gen (Generator): The generator object.
            total_real_state (np.ndarray): The total real state of the system.
            total_input_state (np.ndarray): The total input state of the system.
            param (str): The parameter to compute the gradient for ("alpha" or "beta").

        Returns:
            np.ndarray: The gradient of the discriminator with respect to beta.
        """
        ################################################################
        # Get the current Generator and Discriminator states:
        ################################################################
        final_fake_state, final_real_state, _ = get_final_comp_states_for_dis(gen, total_input_state, total_real_state)
        A, B, _, _ = self.get_dis_matrices_rep()

        ################################################################
        # Initialize gradients on 0:
        #################################################################
        grad_psi_term = np.zeros_like(self.beta, dtype=complex)
        grad_phi_term = np.zeros_like(self.beta, dtype=complex)
        grad_reg_term = np.zeros_like(self.beta, dtype=complex)

        for type in range(len(self.herm)):
            ###########################################################
            # Compute the alpha or beta gradient step:
            ############################################################
            grad_step = DiscriminatorGradientStep(self).grad_step(param)
            gradpsi_list, gradphi_list, gradreg_list = grad_step(final_fake_state, final_real_state, A, B, type)

            # Add grad of psi, phi and reg terms
            grad_psi_term[:, type] = np.asarray(gradpsi_list)
            grad_phi_term[:, type] = np.asarray(gradphi_list)
            grad_reg_term[:, type] = np.asarray(gradreg_list)

        grad = np.real(grad_psi_term - grad_phi_term - grad_reg_term)

        return grad

    def get_dis_matrices_rep(self) -> tuple:
        """Computes the matrices A and B from the psi and phi matrices, scaled by the inverse of lambda.

        Raises:
            ValueError: If lambda is zero or if psi or phi are not square matrices.

        Returns:
            tuple: A tuple containing the matrices A, B, psi, phi.
        """
        psi, phi = self.getPsiPhi()
        A, B = None, None

        #########################################################
        # Compute the matrix A, with expm:
        ##########################################################
        try:
            A = expm((-1 / lamb) * phi)
        except ValueError:
            print_and_train_log(f"Can't exp(phi/lamb), 1/lamb: {(1 / lamb)}, size of phi:{phi.shape}\n", CFG.log_path)

        #########################################################
        # Compute the matrix B, with expm:
        ##########################################################
        try:
            B = expm((1 / lamb) * psi)
        except ValueError:
            print_and_train_log(f"Can't exp(psi/lamb), 1/lamb: {(1 / lamb)}, size of psi: {psi.shape}\n", CFG.log_path)

        if A is None or B is None:
            raise ValueError("Invalid lambda, phi, or psi parameters for computing gradients.")

        return A, B, psi, phi

    def load_model_params(self, file_path: str) -> bool:
        """
        Load discriminator parameters (alpha, beta) from a saved model, if compatible.

        If the saved model has one less qubit (no ancilla), load only the matching parameters.

        WARNING: Only load trusted pickle files! Untrusted files may be insecure.

        Args:
            file_path (str): Path to the saved discriminator model file.

        Returns:
            bool: True if the model was loaded successfully and is compatible, False otherwise.
        """
        ######################################################################
        # Check if the file exists and is a valid pickle file
        ########################################################################
        if not os.path.exists(file_path):
            print_and_train_log("Discriminator model file not found\n", CFG.log_path)
            return False
        try:
            with open(file_path, "rb") as f:
                saved_dis: Discriminator = pickle.load(f)
        except (OSError, pickle.UnpicklingError) as e:
            print_and_train_log(f"Could not load discriminator model: {e}\n", CFG.log_path)
            return False

        ##################################################################
        # Check for the cases you should't load -> Stop
        ##################################################################
        # For this corner case, in reality the load will still work, since we always have matrices NxN or (N+1)x(N+1)
        # but you would load a Discriminator for distinguishing a T(3) to a T(4), or vice-versa, which shouldn't happen..
        if saved_dis.target_size != self.target_size:
            print_and_train_log("Saved discriminator model is incompatible (target size mismatch).\n", CFG.log_path)
            return False

        ########################################################################
        # Check for exact match (same size)
        ########################################################################
        if saved_dis.size == self.size:
            self.alpha = saved_dis.alpha.copy()
            self.beta = saved_dis.beta.copy()
            print_and_train_log("Discriminator parameters loaded\n", CFG.log_path)
            return True

        ##################################################################
        # When one qubit difference (adding or removing an ancilla with pass)
        ###################################################################
        if abs(saved_dis.size - self.size) == 1:
            # Determine the minimum number of qubits (the overlap)
            min_size = min(saved_dis.size, self.size)
            # Load only the matching parameters
            self.alpha[:min_size] = saved_dis.alpha[:min_size].copy()
            self.beta[:min_size] = saved_dis.beta[:min_size].copy()
            print_and_train_log("Discriminator parameters loaded partially (one qubit difference).\n", CFG.log_path)
            return True

        print_and_train_log("Saved discriminator model is incompatible (size or shape mismatch).\n", CFG.log_path)
        return False


class DiscriminatorGradientStep:
    """Class for computing the gradient steps of the discriminator."""

    def __init__(self, dis: Discriminator):
        """Initialize the DiscriminatorGradientStep with the size of the system."""
        self.dis = dis

    def grad_step(self, param: str) -> callable:
        """Get the gradient step function for the specified parameter.

        Args:
            param (str): The parameter to compute the gradient for ("alpha" or "beta").

        Returns:
            callable: The gradient step function for the specified parameter (alpha or beta).
        """
        return self._grad_alpha if param == "alpha" else self._grad_beta if param == "beta" else None

    def _grad_alpha(
        self,
        final_fake_state: np.ndarray,
        final_real_state: np.ndarray,
        A: np.ndarray,
        B: np.ndarray,
        type: str,
    ) -> np.ndarray:
        """Calculate a step of the gradient of the discriminator with respect to alpha.

        Args:
            final_fake_state (np.ndarray): The final fake state of the system.
            final_real_state (np.ndarray): The final real state of the system.
            A (np.ndarray): The matrix representation of the real part of the discriminator.
            B (np.ndarray): The matrix representation of the fake part of the discriminator.
            type (str): The type of hermitian operator (0: I, 1: X, 2: Y, 3: Z).

        Returns:
            tuple: A tuple containing the gradients of psi, phi, and regularization terms.
        """
        gradpsi: list = self._grad_psi_phi(type, respect_to="psi")
        gradpsi_list, gradphi_list, gradreg_list = [], [], []

        # fmt: off
        for grad_psi in gradpsi:
            ##################################################################
            # Compute the gradient of psi with respect to alpha
            ##################################################################
            gradpsi_list.append(np.ndarray.item(np.matmul(final_real_state.getH(), np.matmul(grad_psi, final_real_state))))
            # gradpsi_list.append(np.asscalar(np.matmul(real_state.getH(), np.matmul(grad_psi, real_state))))

            ##################################################################
            # No gradient of phi with respect to alpha, so append 0
            ##################################################################
            gradphi_list.append(0)

            ##################################################################
            # Compute the regularization terms:
            ##################################################################
            term1 = cs * np.matmul(final_fake_state.getH(), np.matmul(A, final_fake_state)) * np.matmul(final_real_state.getH(), np.matmul(grad_psi, np.matmul(B, final_real_state)))
            term2 = cs * np.matmul(final_fake_state.getH(), np.matmul(grad_psi, np.matmul(B, final_real_state))) * np.matmul(final_real_state.getH(), np.matmul(A, final_fake_state))
            term3 = cs * np.matmul(final_fake_state.getH(), np.matmul(A, final_real_state)) * np.matmul(final_real_state.getH(), np.matmul(grad_psi, np.matmul(B, final_fake_state)))
            term4 = cs * np.matmul(final_fake_state.getH(), np.matmul(grad_psi, np.matmul(B, final_fake_state))) * np.matmul(final_real_state.getH(), np.matmul(A, final_real_state))
            gradreg_list.append(np.ndarray.item(lamb / np.e * (cst1 * term1 - cst2 * term2 - cst2 * term3 + cst3 * term4)))
            # gradreg_list.append(np.asscalar(lamb / np.e * (cst1 * term1 - cst2 * term2 - cst2 * term3 + cst3 * term4)))
        # fmt: on

        return gradpsi_list, gradphi_list, gradreg_list

    def _grad_beta(self, final_fake_state, final_real_state, A: np.ndarray, B: np.ndarray, type: str):
        """Calculate a step of the gradient of the discriminator with respect to beta.

        Args:
            final_fake_state (np.ndarray): The final fake state of the system.
            final_real_state (np.ndarray): The final real state of the system.
            A (np.ndarray): The matrix representation of the real part of the discriminator.
            B (np.ndarray): The matrix representation of the fake part of the discriminator.
            type (str): The type of hermitian operator (0: I, 1: X, 2: Y, 3: Z).

        Returns:
            tuple: A tuple containing the gradients of psi, phi, and regularization terms.
        """
        gradphi: list = self._grad_psi_phi(type, respect_to="phi")
        gradpsi_list, gradphi_list, gradreg_list = [], [], []

        # fmt: off
        for grad_phi in gradphi:
            ##################################################################
            # No gradient of psi with respect to beta, so append 0
            ##################################################################
            gradpsi_list.append(0)

            ##################################################################
            # Compute the gradient of phi with respect to beta
            ##################################################################
            gradphi_list.append(np.ndarray.item(np.matmul(final_fake_state.getH(), np.matmul(grad_phi, final_fake_state))))
            # gradphi_list.append(np.asscalar(np.matmul(total_fake_state.getH(), np.matmul(grad_phi, total_fake_state))))

            ##################################################################
            # Compute the regularization terms:
            ##################################################################
            term1 = cs * np.matmul(final_fake_state.getH(), np.matmul(grad_phi, np.matmul(A, final_fake_state))) * np.matmul(final_real_state.getH(), np.matmul(B, final_real_state))
            term2 = cs * np.matmul(final_fake_state.getH(), np.matmul(B, final_real_state)) * np.matmul(final_real_state.getH(), np.matmul(grad_phi, np.matmul(A, final_fake_state)))
            term3 = cs * np.matmul(final_fake_state.getH(), np.matmul(grad_phi, np.matmul(A, final_real_state))) * np.matmul(final_real_state.getH(), np.matmul(B, final_fake_state))
            term4 = cs * np.matmul(final_fake_state.getH(), np.matmul(B, final_fake_state)) * np.matmul(final_real_state.getH(), np.matmul(grad_phi, np.matmul(A, final_real_state)))
            gradreg_list.append(np.ndarray.item(lamb / np.e * (cst1 * term1 - cst2 * term2 - cst2 * term3 + cst3 * term4)))
            # gradreg_list.append(np.asscalar(lamb / np.e * (cst1 * term1 - cst2 * term2 - cst2 * term3 + cst3 * term4)))
        # fmt: on

        return gradpsi_list, gradphi_list, gradreg_list

    # Psi/Phi gradients
    def _grad_psi_phi(self, type: int, respect_to: str) -> list:
        """Calculate the gradient of the discriminator with respect to psi/phi.

        Args:
            type (int): The type of hermitian operator (0: I, 1: X, 2: Y, 3: Z).
            psi_or_phi (str): Specify whether to compute the gradient for psi or phi.

        Returns:
            list: A list of gradients for each qubit in the discriminator, respect psi/phi.
        """
        ########################################################################
        # Chose respect which variable to compute the gradient (alpha or beta)
        ########################################################################
        alpha_or_beta = self.dis.alpha if respect_to == "psi" else self.dis.beta if respect_to == "phi" else None
        if alpha_or_beta is None:
            raise ValueError("`respect_to` must be either 'psi' or 'phi'.")

        ########################################################################
        # Compute gradients for each qubit:
        ########################################################################
        grad_psi_phi = []
        for i in range(self.dis.size):
            grad_psi_phi_I = 1
            for j in range(self.dis.size):
                if i == j:
                    grad_psi_phi_i = self.dis.herm[type]
                else:
                    grad_psi_phi_i = np.zeros_like(self.dis.herm[0], dtype=complex)
                    for k in range(len(self.dis.herm)):
                        grad_psi_phi_i += alpha_or_beta[j][k] * self.dis.herm[k]
                grad_psi_phi_I = np.kron(grad_psi_phi_I, grad_psi_phi_i)
            grad_psi_phi.append(grad_psi_phi_I)
        return grad_psi_phi
