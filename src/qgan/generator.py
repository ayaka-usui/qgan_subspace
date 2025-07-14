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

import itertools
import os
import pickle
from copy import deepcopy

import torch as np

from config import CFG
from qgan.ancilla import get_final_gen_state_for_discriminator
from qgan.cost_functions import braket
from qgan.discriminator import Discriminator
from tools.data.data_managers import print_and_log
from tools.optimizer import MomentumOptimizer
from tools.qobjects import Identity, QuantumCircuit, QuantumGate

np.set_default_device(CFG.device)


class Generator:
    """Generator class for Quantum GAN."""

    def __init__(self, total_input_state):
        # Set general used params:
        self.size: int = CFG.system_size + (1 if CFG.extra_ancilla else 0)
        self.qc: QuantumCircuit = QuantumCircuit(self.size, "generator")
        self.optimizer: MomentumOptimizer = MomentumOptimizer()

        # Set the params, for comparison while loading:
        self.ancilla: bool = CFG.extra_ancilla
        self.ancilla_topology: str = CFG.ancilla_topology  # Type doesn't matter, ancilla always passes through gen
        self.ansatz: str = CFG.gen_ansatz
        self.layers: int = CFG.gen_layers
        self.target_size: int = CFG.system_size
        self.target_hamiltonian: str = CFG.target_hamiltonian

        # Set the ansatz circuit:
        self.qc = Ansatz.get_ansatz_type_circuit(self.ansatz)(self.qc, self.size, self.layers)
        self.total_input_state = total_input_state
        self.total_gen_state = self.get_total_gen_state()

    def get_total_gen_state(self):
        """Get the total generator state, including the untouched qubits in front (choi).

        Args:
            total_input_state (np.Tensor): The input state vector.

        Returns:
            np.Tensor: The total generator state vector.
        """
        Untouched_x_G = np.kron(
            np.eye(2**CFG.system_size, dtype=np.complex64),
            self.qc.get_mat_rep(),
        )

        return np.matmul(Untouched_x_G, self.total_input_state)

    def get_total_gen_grad(self, index):
        """Get the total generator gradient for a specific gate index.

        Args:
            index (int): The index of the gate for which to compute the gradient.

        Returns:
            np.Tensor: The total generator gradient vector for the specified gate.
        """
        Untouched_x_G_grad_i = np.kron(
            np.eye(2**CFG.system_size, dtype=np.complex64),
            self.qc.get_grad_mat_rep(index),
        )
        return np.matmul(Untouched_x_G_grad_i, self.total_input_state)

    def update_gen(self, dis: Discriminator, final_target_state):
        """Update the generator parameters (angles) using the optimizer.

        Args:
            dis (Discriminator): The discriminator to compute gradients.
            final_target_state (np.Tensor): The target state vector.
        """
        ###############################################################
        # Compute the gradient
        ###############################################################
        grad = self._grad_theta(dis, final_target_state, self.total_gen_state)

        # Get the new thetas from the gradient
        theta = np.tensor([gate.angle for gate in self.qc.gates], dtype=np.float32)
        new_theta = self.optimizer.move_in_grad(theta, grad, "min")

        ###############################################################
        # Update the angles in the quantum circuit
        ###############################################################
        for i in range(self.qc.depth):
            self.qc.gates[i].angle = new_theta[i].item()

        ###############################################################
        # Update the total generator state with the new angles
        ###############################################################
        self.total_gen_state = self.get_total_gen_state()

    def _grad_theta(self, dis: Discriminator, final_target_state, total_gen_state):
        """Compute the gradient of the generator parameters (angles) with respect to the discriminator's output.

        Args:
            dis (Discriminator): The discriminator to compute gradients.
            final_target_state (np.Tensor): The target state vector.
            total_gen_state (np.Tensor): The current generator state vector.

        Returns:
            np.Tensor: The gradient of the generator parameters.
        """
        #######################################################################
        # Get the current Generator, Target and Discriminator states:
        #######################################################################
        final_gen_state = get_final_gen_state_for_discriminator(total_gen_state)
        A, B, _, phi = dis.get_dis_matrices_rep()

        grad_g_psi, grad_g_phi, grad_g_reg = [], [], []

        for i in range(self.qc.depth):
            # fmt: off
            # For psi term
            grad_g_psi.append(np.tensor(0.0, dtype=np.complex64))

            # For phi term
            total_gen_grad = self.get_total_gen_grad(i)
            final_gen_grad = get_final_gen_state_for_discriminator(total_gen_grad)
            tmp_grad = braket(final_gen_grad, phi, final_gen_state) + braket(final_gen_state, phi, final_gen_grad)
            grad_g_phi.append(tmp_grad)

            # For reg term
            term1 = braket(final_gen_grad, A, final_gen_state) * braket(final_target_state, B, final_target_state)
            term2 = braket(final_gen_state, A, final_gen_grad) * braket(final_target_state, B, final_target_state)
            term3 = braket(final_gen_grad, B, final_target_state) * braket(final_target_state, A, final_gen_state)
            term4 = braket(final_gen_state, B, final_target_state) * braket(final_target_state, A, final_gen_grad)
            term5 = braket(final_gen_grad, A, final_target_state) * braket(final_target_state, B, final_gen_state)
            term6 = braket(final_gen_state, A, final_target_state) * braket(final_target_state, B, final_gen_grad)
            term7 = braket(final_gen_grad, B, final_gen_state) * braket(final_target_state, A, final_target_state)
            term8 = braket(final_gen_state, B, final_gen_grad) * braket(final_target_state, A, final_target_state)
            tmp_reg_grad = CFG.lamb / np.e * (CFG.cst1 * (term1 + term2) - CFG.cst2 * (term3 + term4 + term5 + term6) + CFG.cst3 * (term7 + term8))

            grad_g_reg.append(tmp_reg_grad)
            # fmt: on

        g_psi = np.stack(grad_g_psi)
        g_phi = np.stack(grad_g_phi)
        g_reg = np.stack(grad_g_reg)

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
            print_and_log("ERROR: Generator model file not found\n", CFG.log_path)
            return False
        try:
            with open(file_path, "rb") as f:
                saved_gen: Generator = pickle.load(f)
        except (OSError, pickle.UnpicklingError) as e:
            print_and_log(f"ERROR: Could not load generator model: {e}\n", CFG.log_path)
            return False

        ##################################################################
        # Check for the cases you can't load -> Stop
        ##################################################################
        cant_load = False

        if saved_gen.target_size != self.target_size:
            print_and_log("ERROR: Saved generator model is incompatible (target size mismatch).\n", CFG.log_path)
            cant_load = True

        # This one could work, but it wouldn't make sense, since the generator would be useless, better to stop:
        if saved_gen.target_hamiltonian != self.target_hamiltonian:
            print_and_log("ERROR: Saved generator model is incompatible (target hamiltonian mismatch).\n", CFG.log_path)
            cant_load = True

        if saved_gen.ansatz != self.ansatz:
            print_and_log("ERROR: Can't load due to different ansatz in gen.\n", CFG.log_path)
            cant_load = True

        if saved_gen.layers != self.layers:
            print_and_log("ERROR: Can't load due to different number of layers in gen.\n", CFG.log_path)
            cant_load = True

        if saved_gen.ancilla and self.ancilla and saved_gen.ancilla_topology != self.ancilla_topology:
            print_and_log(
                "ERROR: Can't load gen with ancilla into another one with ancilla too, but in a different topology.\n",
                CFG.log_path,
            )
            cant_load = True

        # Stop loading, logging all the errors at the same time in one execution:
        if cant_load:
            return False

        ##################################################################
        # Case of exact match
        ##################################################################
        if saved_gen.size == self.size and saved_gen.ancilla == self.ancilla:  # Redundant size check, kept for clarity
            print_and_log("Gen match in size and ancilla.\n", CFG.log_path)

            # Corner case, when ancilla number of gates has changed:
            if len(saved_gen.qc.gates) != len(self.qc.gates):
                print_and_log("Gen number of gates don't match (change in code implementation?).\n", CFG.log_path)
                return False

            # Normal case, when all gates match:
            self.qc = deepcopy(saved_gen.qc)
            self.total_gen_state = deepcopy(saved_gen.total_gen_state)

            # Load the optimizer parameters if they exist in the saved generator
            self.optimizer = deepcopy(saved_gen.optimizer)

            print_and_log("Generator parameters loaded\n", CFG.log_path)
            return True

        ##################################################################
        # Case of adding or removing an ancilla (one qubit difference)
        ###################################################################
        if saved_gen.ancilla != self.ancilla and abs(saved_gen.size - self.size) == 1:  # Rdundt size check, but clarity
            print_and_log("Gen match in size, but with diff in ancilla.\n", CFG.log_path)

            # Partially load the generator parameters:
            self.set_common_gate_params_from_loaded_gen(saved_gen)

            # Since we can't copy the gen state, we regenerate it:
            self.total_gen_state = self.get_total_gen_state()

            # Load the optimizer parameters if they exist in the saved generator
            # self.optimizer.v = saved_gen.optimizer.v
            # TODO: Check how to load momentum, if not exact match

            print_and_log("Generator parameters partially loaded (excluding ancilla difference)\n", CFG.log_path)
            return True

        ##################################################################
        # For other cases, error the loading
        ###################################################################
        print_and_log("ERROR: Saved generator model is incompatible (size or depth mismatch).\n", CFG.log_path)
        return False

    def set_common_gate_params_from_loaded_gen(self, saved_gen: "Generator") -> None:
        """Set the common gate parameters (angles) from the loaded generator.

        Args:
            saved_gen (Generator): The generator instance.
        """
        # Determine the minimum number of qubits (the overlap):
        min_size = min(saved_gen.qc.size, self.qc.size)
        # Map: for each gate in self.qc.gates, find a matching gate in saved_gen.qc.gates
        # A matching gate: same type, same qubits (within min_size), same number of qubits (1q/2q)
        # To handle multiple gates with same type/qubits, use an index to track which have been matched
        used_indices = set()
        for self_gate in self.qc.gates:
            # Only consider gates that act only on the overlapping qubits (no ancilla)
            q1, q2 = self_gate.qubit1, self_gate.qubit2
            if (q1 is not None and q1 >= min_size) or (q2 is not None and q2 >= min_size):
                continue
            # Try to find the next matching gate in saved_gen.qc.gates that hasn't been used
            for idx, saved_gate in enumerate(saved_gen.qc.gates):
                if idx in used_indices:
                    continue
                sq1, sq2 = saved_gate.qubit1, saved_gate.qubit2
                if self_gate.name == saved_gate.name and ((q1 == sq1 and q2 == sq2) or (q1 == sq2 and q2 == sq1)):
                    self_gate.angle = saved_gate.angle
                    used_indices.add(idx)
                    break


##################################################################
# GENERATOR ANSATZ DEFINITIONS
##################################################################
class Ansatz:
    """Ansatz class for constructing quantum circuits with specific gates"""

    @staticmethod
    def get_ansatz_type_circuit(type_of_ansatz: str) -> callable:
        """Construct the ansatz based on the type specified.

        Args:
            type_of_ansatz (str): Type of ansatz to construct, either 'XX_YY_ZZ_Z' or 'ZZ_X_Z'.

        Returns:
            callable: Function to construct the quantum circuit with the specified ansatz.
        """
        if type_of_ansatz == "XX_YY_ZZ_Z":
            return Ansatz.construct_qcircuit_XX_YY_ZZ_Z

        if type_of_ansatz == "ZZ_X_Z":
            return Ansatz.construct_qcircuit_ZZ_X_Z

        raise ValueError("Invalid type of ansatz specified.")

    @staticmethod
    def construct_qcircuit_XX_YY_ZZ_Z(qc: QuantumCircuit, size: int, layer: int) -> QuantumCircuit:
        """Construct a quantum circuit with the ansatz of XX YY ZZ and FieldZ

        Args:
            qc (QuantumCircuit): Quantum Circuit
            size (int): Size of the Quantum Circuit
            layer (int): Number of layers

        Returns:
            QuantumCircuit: Quantum Circuit with the ansatz of XYZ and FieldZ
        """
        # If extra ancilla is used, different than ansatz, we reduce the size by 1,
        # to implement the ancilla logic separately.
        if CFG.extra_ancilla:
            size -= 1

        entg_list = ["XX", "YY", "ZZ"]
        for _ in range(layer):
            # First 1 qubit gates
            for i in range(size):
                qc.add_gate(QuantumGate("Z", i, angle=0))
            # Ancilla 1q gates for: total, bridge and disconnected:
            if CFG.extra_ancilla and CFG.do_ancilla_1q_gates:
                qc.add_gate(QuantumGate("Z", size, angle=0))

            # Then 2 qubit gates:
            for i, gate in itertools.product(range(size - 1), entg_list):
                qc.add_gate(QuantumGate(gate, i, i + 1, angle=0))
            # Ancilla ancilla coupling (2q) logic for: total and bridge
            if CFG.extra_ancilla:
                if CFG.ancilla_topology == "total":
                    for i, gate in itertools.product(range(size), entg_list):
                        qc.add_gate(QuantumGate(gate, i, size, angle=0))
                if CFG.ancilla_topology == "bridge":
                    for gate in entg_list:
                        qc.add_gate(QuantumGate(gate, 0, size, angle=0))
                if CFG.ancilla_topology in ["bridge", "ansatz"]:
                    qubit_to_connect_to = CFG.ancilla_connect_to if CFG.ancilla_connect_to is not None else size - 1
                    for gate in entg_list:
                        qc.add_gate(QuantumGate(gate, qubit_to_connect_to, size, angle=0))

        # Make uniform random angles for the gates (0 to 2*pi)
        theta = np.empty(len(qc.gates)).uniform_(0, 2 * np.pi)
        for i, gate_i in enumerate(qc.gates):
            gate_i.angle = theta[i]

        return qc

    @staticmethod
    def construct_qcircuit_ZZ_X_Z(qc: QuantumCircuit, size: int, layer: int) -> QuantumCircuit:
        """Construct a quantum circuit with the ansatz of ZZ and XZ

        Args:
            qc (QuantumCircuit): Quantum Circuit
            size (int): Size of the Quantum Circuit
            layer (int): Number of layers

        Returns:
            QuantumCircuit: Quantum Circuit with the ansatz of ZZ and XZ
        """
        # If extra ancilla is used, different than ansatz, we reduce the size by 1,
        # to implement the ancilla logic separately.
        if CFG.extra_ancilla:
            size -= 1

        for _ in range(layer):
            # First 1 qubit gates
            for i in range(size):
                qc.add_gate(QuantumGate("X", i, angle=0))
                qc.add_gate(QuantumGate("Z", i, angle=0))
            # Ancilla 1q gates for: total, bridge and disconnected:
            if CFG.extra_ancilla and CFG.do_ancilla_1q_gates:
                qc.add_gate(QuantumGate("X", size, angle=0))
                qc.add_gate(QuantumGate("Z", size, angle=0))
            # Then 2 qubit gates
            for i in range(size - 1):
                qc.add_gate(QuantumGate("ZZ", i, i + 1, angle=0))
            # Ancilla ancilla coupling (2q) logic for: total and bridge
            if CFG.extra_ancilla:
                if CFG.ancilla_topology == "total":
                    for i in range(size):
                        qc.add_gate(QuantumGate("ZZ", i, size, angle=0))
                if CFG.ancilla_topology == "bridge":
                    qc.add_gate(QuantumGate("ZZ", 0, size, angle=0))
                if CFG.ancilla_topology in ["bridge", "ansatz"]:
                    qubit_to_connect_to = CFG.ancilla_connect_to if CFG.ancilla_connect_to is not None else size - 1
                    qc.add_gate(QuantumGate("ZZ", qubit_to_connect_to, size, angle=0))

        # Make uniform random angles for the gates (0 to 2*pi)
        theta = np.empty(len(qc.gates)).uniform_(0, 2 * np.pi)
        for i, gate_i in enumerate(qc.gates):
            gate_i.angle = theta[i]

        return qc
