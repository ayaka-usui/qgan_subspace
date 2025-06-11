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
"""Generator module"""

import os
import pickle

import numpy as np

from config import CFG
from qgan.ancilla import get_final_fake_state_for_discriminator
from qgan.cost_functions import get_final_comp_states_for_dis
from qgan.discriminator import Discriminator
from qgan.generator.ansatz import get_ansatz_func
from tools.data.data_managers import print_and_train_log
from tools.optimizer import MomentumOptimizer
from tools.qobjects.qcircuit import Identity, QuantumCircuit


class Generator:
    """Generator class for Quantum GAN."""

    def __init__(self):
        # Set general used params:
        self.size: int = CFG.system_size + (1 if CFG.extra_ancilla else 0)
        self.qc: QuantumCircuit = self.init_qcircuit()
        self.optimizer: MomentumOptimizer = MomentumOptimizer()

        # Set the params, for comparison while loading:
        self.ancilla: bool = CFG.extra_ancilla
        self.ancilla_topology: str = CFG.ancilla_topology  # Type doesn't matter, ancilla always passes through gen
        self.ansatz: str = CFG.gen_ansatz
        self.layers: int = CFG.gen_layers

        # Set the ansatz circuit:
        self.set_qcircuit(get_ansatz_func(self.ansatz)(self.qc, self.size, self.layers))

    def set_qcircuit(self, qc: QuantumCircuit):
        self.qc = qc

    def init_qcircuit(self):
        """Initialize the quantum circuit for the generator."""
        qcircuit = QuantumCircuit(self.size, "generator")
        return qcircuit

    def get_Untouched_qubits_and_Gen(self) -> np.ndarray:
        """Get the matrix representation of the circuit at the Generator step, including the untouched qubits in front (choi).

        Returns:
            np.ndarray: The Kronecker product of the untouched qubits (choi) and the quantum circuit's matrix representation.
        """
        return np.kron(Identity(CFG.system_size), self.qc.get_mat_rep())

    def update_gen(self, dis: Discriminator, total_real_state: np.ndarray, total_input_state: np.ndarray):
        """Update the generator parameters (angles) using the optimizer.

        Args:
            dis (Discriminator): The discriminator to compute gradients.
            total_real_state (np.ndarray): The real state vector.
            total_input_state (np.ndarray): The input state vector.
        """
        ###############################################################
        # Compute the gradient
        ###############################################################
        # Store old angles, needed later for the optimizer:
        theta = []
        for gate in self.qc.gates:
            theta.append(gate.angle)

        grad: np.ndarray = np.asarray(self._grad_theta(dis, total_real_state, total_input_state))
        new_angle = self.optimizer.move_in_grad(np.asarray(theta), grad, "min")

        ###############################################################
        # Update the angles in the quantum circuit
        ###############################################################
        for i in range(self.qc.depth):
            self.qc.gates[i].angle = new_angle[i]

    def _grad_theta(self, dis: Discriminator, total_real_state: np.ndarray, total_input_state: np.ndarray):
        """Compute the gradient of the generator parameters (angles) with respect to the discriminator's output.

        Args:
            dis (Discriminator): The discriminator to compute gradients.
            total_real_state (np.ndarray): The real state vector.
            total_input_state (np.ndarray): The input state vector.

        Returns:
            np.ndarray: The gradient of the generator parameters.
        """
        #######################################################################
        # Get the current Generator and Discriminator states:
        #######################################################################
        final_fake_state, final_real_state, _ = get_final_comp_states_for_dis(self, total_input_state, total_real_state)
        A, B, _, phi = dis.get_dis_matrices_rep()

        grad_g_psi, grad_g_phi, grad_g_reg = [], [], []

        for i in range(self.qc.depth):
            # fmt: off
            grad_i = np.kron(Identity(CFG.system_size), self.qc.get_grad_mat_rep(i))

            # For psi term
            grad_g_psi.append(0)

            # For phi term
            fake_grad = np.matmul(grad_i, total_input_state)
            final_fake_grad = np.matrix(get_final_fake_state_for_discriminator(fake_grad))
            tmp_grad = np.matmul(final_fake_grad.getH(), np.matmul(phi, final_fake_state)) + np.matmul(final_fake_state.getH(), np.matmul(phi, final_fake_grad))

            grad_g_phi.append(np.ndarray.item(tmp_grad))
            # grad_g_phi.append(np.asscalar(tmp_grad))

            # For reg term
            term1 = np.matmul(final_fake_grad.getH(), np.matmul(A, final_fake_state)) * np.matmul(final_real_state.getH(), np.matmul(B, final_real_state))
            term2 = np.matmul(final_fake_state.getH(), np.matmul(A, final_fake_grad)) * np.matmul(final_real_state.getH(), np.matmul(B, final_real_state))
            term3 = np.matmul(final_fake_grad.getH(), np.matmul(B, final_real_state)) * np.matmul(final_real_state.getH(), np.matmul(A, final_fake_state))
            term4 = np.matmul(final_fake_state.getH(), np.matmul(B, final_real_state)) * np.matmul(final_real_state.getH(), np.matmul(A, final_fake_grad))
            term5 = np.matmul(final_fake_grad.getH(), np.matmul(A, final_real_state)) * np.matmul(final_real_state.getH(), np.matmul(B, final_fake_state))
            term6 = np.matmul(final_fake_state.getH(), np.matmul(A, final_real_state)) * np.matmul(final_real_state.getH(), np.matmul(B, final_fake_grad))
            term7 = np.matmul(final_fake_grad.getH(), np.matmul(B, final_fake_state)) * np.matmul(final_real_state.getH(), np.matmul(A, final_real_state))
            term8 = np.matmul(final_fake_state.getH(), np.matmul(B, final_fake_grad)) * np.matmul(final_real_state.getH(), np.matmul(A, final_real_state))
            tmp_reg_grad = CFG.lamb / np.e * (CFG.cst1 * (term1 + term2) - CFG.cst2 * (term3 + term4) - CFG.cst2 * (term5 + term6) + CFG.cst3 * (term7 + term8))

            grad_g_reg.append(np.ndarray.item(tmp_reg_grad))
            # grad_g_reg.append(np.asscalar(tmp_reg_grad))
            # fmt: on

        g_psi = np.asarray(grad_g_psi)
        g_phi = np.asarray(grad_g_phi)
        g_reg = np.asarray(grad_g_reg)

        grad = np.real(g_psi - g_phi - g_reg)

        return grad

    def load_model_params(self, file_path: str) -> bool:
        """
        Load generator parameters (angles) from a saved model, if compatible.

        Supports loading when adding or removing an ancilla (one qubit difference).

        WARNING: Only load trusted pickle files! Untrusted files may be insecure.

        Args:
            file_path (str): Path to the saved generator model file.

        Returns:
            bool: True if the model was loaded successfully, False otherwise.
        """
        ##################################################################
        # Check if the file exists and is a valid pickle file
        ##################################################################
        if not os.path.exists(file_path):
            print_and_train_log("Generator model file not found\n", CFG.log_path)
            return False
        try:
            with open(file_path, "rb") as f:
                saved_gen: Generator = pickle.load(f)
        except (OSError, pickle.UnpicklingError) as e:
            print_and_train_log(f"Could not load generator model: {e}\n", CFG.log_path)
            return False

        ##################################################################
        # Check for the cases you can't load -> Stop
        ##################################################################
        if saved_gen.ansatz != self.ansatz or saved_gen.layers != self.layers:
            print_and_train_log("ERROR: Can't load due to different ansatz or num layers.\n", CFG.log_path)
            return False

        if saved_gen.ancilla and self.ancilla and saved_gen.ancilla_topology != self.ancilla_topology:
            print_and_train_log(
                "NOT_IMPLEMENTED_ERROR: Can't load model with ancilla, into another one with ancilla in a different topology (number of gates, wouldn't match, and a gate reduction isn't trivial).\n",
                CFG.log_path,
            )
            return False

        ##################################################################
        # Check for exact match
        ##################################################################
        if saved_gen.size == self.size and saved_gen.ancilla == self.ancilla:  # If both true, they have same topology
            print_and_train_log("Match in size and ancilla.\n", CFG.log_path)
            # Corner case, when ancilla number of gates has changed:
            if len(saved_gen.qc.gates) != len(self.qc.gates):
                print_and_train_log("Number of gates don't match, check if older code implementations.\n", CFG.log_path)
                return False
            for i, gate in enumerate(self.qc.gates):
                gate.angle = saved_gen.qc.gates[i].angle
            print_and_train_log("Generator parameters loaded\n", CFG.log_path)
            return True

        # TODO: Check that the -1 dim, load works properly
        ##################################################################
        # When adding or removing an ancilla (one qubit difference)
        ###################################################################
        if saved_gen.ancilla != self.ancilla and abs(saved_gen.size - self.size) == 1:
            # Determine the minimum number of qubits (the overlap)
            min_size = min(saved_gen.size, self.qc.size)
            for gate in self.qc.gates:
                q1 = getattr(gate, "qubit1", None)
                q2 = getattr(gate, "qubit2", None)
                # Only consider gates that act on the overlapping qubits
                if (q1 is not None and q1 >= min_size) or (q2 is not None and q2 >= min_size):
                    continue
                for saved_gate in saved_gen.qc.gates:
                    if (
                        isinstance(gate, type(saved_gate))
                        and getattr(saved_gate, "qubit1", None) == q1
                        and getattr(saved_gate, "qubit2", None) == q2
                    ):
                        gate.angle = saved_gate.angle
                        break
            print_and_train_log("Generator parameters partially loaded (excluding ancilla difference)\n", CFG.log_path)
            return True

        ##################################################################
        # For other cases, error the loading
        ###################################################################
        print_and_train_log("ERROR: Saved generator model is incompatible (size or depth mismatch).\n", CFG.log_path)
        return False
