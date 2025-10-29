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
from copy import deepcopy

import numpy as np
from scipy.linalg import expm

from config import CFG
from qgan.cost_functions import braket
from tools.data.data_managers import print_and_log
from tools.optimizer import MomentumOptimizer
from tools.qobjects.qgates import I, X, Y, Z

cst1, cst2, cst3, lamb = CFG.cst1, CFG.cst2, CFG.cst3, CFG.lamb
cs = -1 / lamb


class Discriminator:
    """Discriminator class for the Quantum GAN.

    Representation with coeff. in front of each herm operator (N x 4):
    ------------------------------------------------------------------
        - alpha: Parameters for the real part of the discriminator (coeff for I, X, Y, Z).
        - beta: Parameters for the imaginary part of the discriminator (coeff for I, X, Y, Z).

    Representation with matrices, spanning the full space  (2^N x 2^N):
    -------------------------------------------------------------------
        - psi: Real part of the discriminator (matrix).
        - phi: Imaginary part of the discriminator (matrix).

    For computing the gradients, we use the following matrices:
    -----------------------------------------------------------
        - A: expm(-1/lamb * phi)
        - B: expm(1/lamb * psi)
    """

    def __init__(self):
        # Set the general used parameters:
        self.size: int = CFG.system_size * 2 + (1 if CFG.extra_ancilla and CFG.ancilla_mode == "pass" else 0)
        self.herm: list = [I, X, Y, Z]
        self._init_params_alpha_beta()
        self.optimizer_psi: MomentumOptimizer = MomentumOptimizer()
        self.optimizer_phi: MomentumOptimizer = MomentumOptimizer()

        # Set params, for comparison when loading:
        self.ancilla: bool = CFG.extra_ancilla
        self.ancilla_mode: str = CFG.ancilla_mode  # Topology doesn't matter, its not a circuit = fully connect matrix.
        self.target_size: int = CFG.system_size
        self.target_hamiltonian: str = CFG.target_hamiltonian

    def _init_params_alpha_beta(self):
        # Each param is: (size x 4)
        self.alpha: np.ndarray = np.zeros((self.size, len(self.herm)))
        self.beta: np.ndarray = np.zeros((self.size, len(self.herm)))

        # Random Discriminator Parameters
        for i in range(self.size):
            self.alpha[i] = -1 + 2 * np.random.random(len(self.herm))
            self.beta[i] = -1 + 2 * np.random.random(len(self.herm))

    def get_psi_and_phi(self) -> np.ndarray:
        """Get matrix representation of real (psi) & imaginary (phi) part of the discriminator

        Size of alpha/beta params (coefficients for each herm) = (num_qubit, 4)
            0: I,   1: X,   2: Y,   3: Z

        The Psi and Phi matrices are constructed by taking the Kronecker product of the individual
        hermitian operators for each qubit, scaled by the corresponding alpha and beta parameters.

        Therefore, the Psi and Phi size is = (2^N, 2^N), where N is the num_qubits in the discriminator.

        Returns:
            tuple[np.ndarray]: Tuple of Psi and Phi, matrix representations of real and imaginary part of discriminator.
        """
        psi, phi = 1, 1
        for i in range(self.size):
            psi_i, phi_i = np.zeros_like(self.herm[0], dtype=complex), np.zeros_like(self.herm[0], dtype=complex)
            for j, herm_j in enumerate(self.herm):
                psi_i += self.alpha[i][j] * herm_j
                phi_i += self.beta[i][j] * herm_j
            psi, phi = np.kron(psi, psi_i), np.kron(phi, phi_i)
        return psi, phi

    def get_dis_matrices_rep(self) -> tuple:
        """Computes the matrices A and B from the psi and phi matrices, scaled by the inverse of lambda.

        Returns:
            tuple: A tuple containing the matrices A, B, psi, phi.
        """
        psi, phi = self.get_psi_and_phi()

        #########################################################
        # Compute the matrix A & B, with expm:
        ##########################################################
        A = expm(float(-1 / lamb) * phi)
        B = expm(float(1 / lamb) * psi)

        return A, B, psi, phi

    def update_dis(self, final_target_state: np.ndarray, final_gen_state: np.ndarray):
        """Update the discriminator parameters (alpha, beta) using the gradients.

        Args:
            final_target_state (np.ndarray): The final target state of the system.
            final_gen_state (np.ndarray): The final gen state of the system.
        """
        ################################################################
        # Get the current Discriminator state:
        ################################################################
        A, B, _, _ = self.get_dis_matrices_rep()

        ####################################################
        # Update alpha
        ####################################################
        grad_alpha = self._compute_grad(final_target_state, final_gen_state, A, B, "alpha")
        new_alpha = self.optimizer_psi.move_in_grad(self.alpha, grad_alpha, "max")

        ####################################################
        # Update beta
        ####################################################
        grad_beta = self._compute_grad(final_target_state, final_gen_state, A, B, "beta")
        new_beta = self.optimizer_phi.move_in_grad(self.beta, grad_beta, "max")

        # Update the parameters later, to avoid affecting gradient computations:
        self.alpha = new_alpha
        self.beta = new_beta

    def _compute_grad(self, final_target_state, final_gen_state, A, B, param: str) -> np.ndarray:
        """Calculate the gradient of the discriminator with respect to the param (alpha or beta).

        Args:
            final_target_state (np.ndarray): The final target state of the system.
            final_gen_state (np.ndarray): The final gen state of the system.
            A (np.ndarray): expm(-1/lamb * phi), with phi being the matrix repr. of the imag part of the dis.
            B (np.ndarray): expm(1/lamb * psi), with psi being the matrix repr. of the real part of the dis.
            param (str): The parameter to compute the gradient for ("alpha" or "beta").

        Returns:
            np.ndarray: The gradient of the discriminator with respect to beta.
        """
        ################################################################
        # Initialize gradients on 0:
        #################################################################
        zero_param = self.alpha if param == "alpha" else self.beta
        grad_psi_term = np.zeros_like(zero_param, dtype=complex)
        grad_phi_term = np.zeros_like(zero_param, dtype=complex)
        grad_reg_term = np.zeros_like(zero_param, dtype=complex)

        for type in range(len(self.herm)):
            ###########################################################
            # Compute the alpha or beta gradient step:
            ############################################################
            grad_step = self._grad_alpha if param == "alpha" else self._grad_beta if param == "beta" else None
            gradpsi_list, gradphi_list, gradreg_list = grad_step(final_target_state, final_gen_state, A, B, type)

            # Add grad of psi, phi and reg terms
            grad_psi_term[:, type] = np.asarray(gradpsi_list)
            grad_phi_term[:, type] = np.asarray(gradphi_list)
            grad_reg_term[:, type] = np.asarray(gradreg_list)

        grad = np.real(grad_psi_term - grad_phi_term - grad_reg_term)

        return np.asarray(grad)

    def _grad_alpha(self, final_target_state, final_gen_state, A: np.ndarray, B: np.ndarray, type: str) -> np.ndarray:
        """Calculate a step of the gradient of the discriminator with respect to alpha.

        Args:
            final_target_state (np.ndarray): The final target state of the system.
            final_gen_state (np.ndarray): The final gen state of the system.
            A (np.ndarray): expm(-1/lamb * phi), with phi being the matrix repr. of the imag part of the dis.
            B (np.ndarray): expm(1/lamb * psi), with psi being the matrix repr. of the real part of the dis.
            type (str): The type of hermitian operator (0: I, 1: X, 2: Y, 3: Z).

        Returns:
            tuple: A tuple containing the gradients of psi, phi, and regularization terms.
        """
        cs = 1 / lamb

        gradpsi: list = self._grad_psi_or_phi(type, respect_to="psi")
        gradpsi_list, gradphi_list, gradreg_list = [], [], []

        # fmt: off
        for grad_psi in gradpsi:
            ##################################################################
            # Compute the gradient of psi with respect to alpha
            ##################################################################
            gradpsi_list.append(np.ndarray.item(braket(final_target_state, grad_psi, final_target_state)))

            ##################################################################
            # No gradient of phi with respect to alpha, so append 0
            ##################################################################
            gradphi_list.append(0)

            ##################################################################
            # Compute the regularization terms:
            ##################################################################
            term1 = cs * braket(final_gen_state, A, final_gen_state) * braket(final_target_state, grad_psi, B, final_target_state)
            term2 = cs * braket(final_gen_state, grad_psi, B, final_target_state) * braket(final_target_state, A, final_gen_state)
            term3 = cs * braket(final_gen_state, A, final_target_state) * braket(final_target_state, grad_psi, B, final_gen_state)
            term4 = cs * braket(final_gen_state, grad_psi, B, final_gen_state) * braket(final_target_state, A, final_target_state)
            gradreg_list.append(np.ndarray.item(lamb / np.e * (cst1 * term1 - cst2 * (term2 + term3) + cst3 * term4)))
        # fmt: on

        return gradpsi_list, gradphi_list, gradreg_list

    def _grad_beta(self, final_target_state, final_gen_state, A: np.ndarray, B: np.ndarray, type: str):
        """Calculate a step of the gradient of the discriminator with respect to beta.

        Args:
            final_target_state (np.ndarray): The final target state of the system.
            final_gen_state (np.ndarray): The final gen state of the system.
            A (np.ndarray): expm(-1/lamb * phi), with phi being the matrix repr. of the imag part of the dis.
            B (np.ndarray): expm(1/lamb * psi), with psi being the matrix repr. of the real part of the dis.
            type (str): The type of hermitian operator (0: I, 1: X, 2: Y, 3: Z).

        Returns:
            tuple: A tuple containing the gradients of psi, phi, and regularization terms.
        """
        cs = -1 / lamb

        gradphi: list = self._grad_psi_or_phi(type, respect_to="phi")
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
            gradphi_list.append(np.ndarray.item(braket(final_gen_state, grad_phi, final_gen_state)))

            ##################################################################
            # Compute the regularization terms:
            ##################################################################
            term1 = cs * braket(final_gen_state, grad_phi, A, final_gen_state) * braket(final_target_state, B, final_target_state)
            term2 = cs * braket(final_gen_state, B, final_target_state) * braket(final_target_state, grad_phi, A, final_gen_state)
            term3 = cs * braket(final_gen_state, grad_phi, A, final_target_state) * braket(final_target_state, B, final_gen_state)
            term4 = cs * braket(final_gen_state, B, final_gen_state) * braket(final_target_state, grad_phi, A, final_target_state)
            gradreg_list.append(np.ndarray.item(lamb / np.e * (cst1 * term1 - cst2 * (term2 + term3) + cst3 * term4)))
        # fmt: on

        return gradpsi_list, gradphi_list, gradreg_list

    # Psi/Phi gradients
    def _grad_psi_or_phi(self, type: int, respect_to: str) -> list:
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
        coefficients = self.alpha if respect_to == "psi" else self.beta

        ########################################################################
        # Compute gradients for each qubit:
        ########################################################################
        grad_matrix = []
        for i in range(self.size):
            grad_matrix_I = 1
            for j in range(self.size):
                if i == j:
                    grad_matrix_j = self.herm[type]
                else:
                    grad_matrix_j = np.zeros_like(self.herm[0], dtype=complex)
                    for k, herm_k in enumerate(self.herm):
                        grad_matrix_j += coefficients[j][k] * herm_k
                grad_matrix_I = np.kron(grad_matrix_I, grad_matrix_j)
            grad_matrix.append(grad_matrix_I)
        return grad_matrix

    def load_model_params(self, file_path: str) -> bool:
        """
        Load discriminator parameters (alpha, beta) from a saved model, if compatible.

        Doesn't load the respective optimizer momentum. They are reset upon loading.

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
            print_and_log("Discriminator model file not found\n", CFG.log_path)
            return False
        try:
            with open(file_path, "rb") as f:
                saved_dis: Discriminator = pickle.load(f)
        except (OSError, pickle.UnpicklingError) as e:
            print_and_log(f"ERROR: Could not load discriminator model: {e}\n", CFG.log_path)
            return False

        ##################################################################
        # Check for the cases you should't load -> Stop
        ##################################################################s
        cant_load = False

        # For this corner case, in reality the load will still work, since we always have matrices NxN or (N+1)x(N+1)
        # but you would load a Discriminator for distinguishing a T(3) to a T(4), or vice-versa, which shouldn't happen..
        if saved_dis.target_size != self.target_size:
            print_and_log("ERROR: Saved discriminator model is incompatible (target size mismatch).\n", CFG.log_path)
            cant_load = True

        # This one could work, but it wouldn't make sense, since the discriminator would be useless, better to stop:
        if saved_dis.target_hamiltonian != self.target_hamiltonian:
            print_and_log("ERROR: Saved discriminator model is incompatible (target hamiltonian mismatch).\n", CFG.log_path)
            cant_load = True

        if cant_load:
            return False

        ########################################################################
        # Check for exact match (same size)
        ########################################################################
        if saved_dis.size == self.size:  # This size check, already takes care into ancilla match!
            self.alpha = deepcopy(saved_dis.alpha)
            self.beta = deepcopy(saved_dis.beta)

            print_and_log("Discriminator parameters loaded\n", CFG.log_path)
            return True

        ##################################################################
        # When one qubit difference (adding or removing an ancilla with pass)
        ###################################################################
        if abs(saved_dis.size - self.size) == 1:  # This size check, already takes care into ancilla match!
            # Determine the minimum number of qubits (the overlap)
            min_size = min(saved_dis.size, self.size)
            # Load only the matching parameters
            self.alpha[:min_size] = saved_dis.alpha[:min_size].copy()
            self.beta[:min_size] = saved_dis.beta[:min_size].copy()

            print_and_log("Discriminator parameters loaded partially (one qubit difference).\n", CFG.log_path)
            return True

        print_and_log("Saved discriminator model is incompatible (size or shape mismatch).\n", CFG.log_path)
        return False
