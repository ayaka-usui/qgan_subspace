import numpy as np
import itertools
from config import CFG
from test_qc.qcircuit import QuantumCircuit
from test_qc.qgates import QuantumGate

class Ansatz_ZZ_X_Z:
    """Ansatz class for constructing quantum circuits with specific gates"""
    def __init__(self, config=CFG):
        self.config = config
        self.layer = self.config.gen_layers
        self.size = self.config.system_size
        self.n_params = self.count_total_params()

    def randomize_gates_in_qc(self, qc: QuantumCircuit, size: int, theta=None) -> QuantumCircuit:
        # Make uniform random angles for the gates (0 to 2*pi)
        if theta is None:
            theta = np.random.uniform(0, 2 * np.pi, len(qc.gates))
        for i, gate_i in enumerate(qc.gates):
            # Depending on the config, randomize the ancilla gates or not:
            if self.config.start_ancilla_gates_randomly or size not in [gate_i.qubit1, gate_i.qubit2]:
                gate_i.angle = theta[i]
        return qc
    
    def count_total_params(self):
        n_params = 0
        size = self.size
        if self.config.extra_ancilla:
            size -= 1
        for _ in range(self.layer):
            # First 1 qubit gates
            for i in range(size):
                n_params += 2
            # Ancilla 1q gates for: total, bridge and disconnected:
            if self.config.extra_ancilla and self.config.do_ancilla_1q_gates:
                n_params += 2
            # Then 2 qubit gates
            for i in range(size - 1):
                n_params += 1
            # Ancilla ancilla coupling (2q) logic for: total and bridge
            if self.config.extra_ancilla:
                if self.config.ancilla_topology == "total":
                    for i in range(size):
                        n_params += 1
                if self.config.ancilla_topology == "bridge":
                    n_params += 1
                if self.config.ancilla_topology in ["bridge", "ansatz"]:
                    n_params += 1
        return n_params
            

    def construct_qcircuit_ZZ_X_Z(self, theta=None) -> QuantumCircuit:
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
        qc = QuantumCircuit(self.size, name='ZZ_X_Z')
        size = self.size
        if self.config.extra_ancilla:
            size -= 1
        for _ in range(self.layer):
            # First 1 qubit gates
            for i in range(size):
                qc.add_gate(QuantumGate("X", i, angle=0))
                qc.add_gate(QuantumGate("Z", i, angle=0))
            # Ancilla 1q gates for: total, bridge and disconnected:
            if self.config.extra_ancilla and self.config.do_ancilla_1q_gates:
                qc.add_gate(QuantumGate("X", size, angle=0))
                qc.add_gate(QuantumGate("Z", size, angle=0))
            # Then 2 qubit gates
            for i in range(size - 1):
                qc.add_gate(QuantumGate("ZZ", i, i + 1, angle=0))
            # Ancilla ancilla coupling (2q) logic for: total and bridge
            if self.config.extra_ancilla:
                if self.config.ancilla_topology == "total":
                    for i in range(size):
                        qc.add_gate(QuantumGate("ZZ", i, size, angle=0))
                if self.config.ancilla_topology == "bridge":
                    qc.add_gate(QuantumGate("ZZ", 0, size, angle=0))
                if self.config.ancilla_topology in ["bridge", "ansatz"]:
                    qubit_to_connect_to = self.config.ancilla_connect_to if self.config.ancilla_connect_to is not None else size - 1
                    qc.add_gate(QuantumGate("ZZ", qubit_to_connect_to, size, angle=0))
        return self.randomize_gates_in_qc(qc, size, theta)
    
class Ansatz_XX_YY_ZZ_Z:
    """Ansatz class for constructing quantum circuits with specific gates"""
    def __init__(self, config=CFG):
        self.config = config
        self.layer = self.config.gen_layers
        self.size = self.config.system_size
        self.n_params = self.count_total_params()

    def randomize_gates_in_qc(self, qc: QuantumCircuit, size: int, theta=None) -> QuantumCircuit:
        # Make uniform random angles for the gates (0 to 2*pi)
        if theta is None:
            theta = np.random.uniform(0, 2 * np.pi, len(qc.gates))
        for i, gate_i in enumerate(qc.gates):
            # Depending on the config, randomize the ancilla gates or not:
            if self.config.start_ancilla_gates_randomly or size not in [gate_i.qubit1, gate_i.qubit2]:
                gate_i.angle = theta[i]
        return qc
    
    def count_total_params(self):
        n_params = 0
        size = self.size
        if self.config.extra_ancilla:
            size -= 1

        entg_list = ["XX", "YY", "ZZ"]
        for _ in range(self.layer):
            # First 1 qubit gates
            for i in range(size):
                n_params += 1
            # Ancilla 1q gates for: total, bridge and disconnected:
            if self.config.extra_ancilla and self.config.do_ancilla_1q_gates:
                n_params += 1

            # Then 2 qubit gates:
            for i, gate in itertools.product(range(size - 1), entg_list):
                n_params += 1
            # Ancilla ancilla coupling (2q) logic for: total and bridge
            if self.config.extra_ancilla:
                if self.config.ancilla_topology == "total":
                    for i, gate in itertools.product(range(size), entg_list):
                        n_params += 1
                if self.config.ancilla_topology == "bridge":
                    for gate in entg_list:
                        n_params += 1
                if self.config.ancilla_topology in ["bridge", "ansatz"]:
                    qubit_to_connect_to = self.config.ancilla_connect_to if self.config.ancilla_connect_to is not None else size - 1
                    for gate in entg_list:
                        n_params += 1

        return n_params     

    def construct_qcircuit_XX_YY_ZZ_Z(self, theta=None) -> QuantumCircuit:
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
        qc = QuantumCircuit(self.size, name='XX_YY_ZZ_Z')
        size = self.size
        if self.config.extra_ancilla:
            size -= 1

        entg_list = ["XX", "YY", "ZZ"]
        for _ in range(self.layer):
            # First 1 qubit gates
            for i in range(size):
                qc.add_gate(QuantumGate("Z", i, angle=0))
            # Ancilla 1q gates for: total, bridge and disconnected:
            if self.config.extra_ancilla and self.config.do_ancilla_1q_gates:
                qc.add_gate(QuantumGate("Z", size, angle=0))

            # Then 2 qubit gates:
            for i, gate in itertools.product(range(size - 1), entg_list):
                qc.add_gate(QuantumGate(gate, i, i + 1, angle=0))
            # Ancilla ancilla coupling (2q) logic for: total and bridge
            if self.config.extra_ancilla:
                if self.config.ancilla_topology == "total":
                    for i, gate in itertools.product(range(size), entg_list):
                        qc.add_gate(QuantumGate(gate, i, size, angle=0))
                if self.config.ancilla_topology == "bridge":
                    for gate in entg_list:
                        qc.add_gate(QuantumGate(gate, 0, size, angle=0))
                if self.config.ancilla_topology in ["bridge", "ansatz"]:
                    qubit_to_connect_to = self.config.ancilla_connect_to if self.config.ancilla_connect_to is not None else size - 1
                    for gate in entg_list:
                        qc.add_gate(QuantumGate(gate, qubit_to_connect_to, size, angle=0))

        return self.randomize_gates_in_qc(qc, size, theta)