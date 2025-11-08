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
"""Definition of a quantum circuit as a PyTorch Module."""

import torch
import torch.nn as nn
from tools.qobjects.qgates import Identity, QuantumGate

class QuantumCircuit(nn.Module):
    """
    Represents a quantum circuit as a sequence of gates.
    Inherits from torch.nn.Module to handle trainable parameters and device placement.
    """
    def __init__(self, size: int):
        super().__init__()
        self.size = size
        # Use ModuleList to ensure parameters of gates are registered
        self.gates = nn.ModuleList()

    def add_gate(self, quantum_gate: QuantumGate):
        """Adds a gate to the circuit."""
        self.gates.append(quantum_gate)

    def forward(self) -> torch.Tensor:
        """
        Calculates the matrix representation of the entire circuit.
        This is the forward pass of the model.
        """
        # Determine the device from the model's parameters. This ensures that
        # new tensors are created on the same device the model is on.
        # We check if there are any gates first to avoid errors on empty circuits.
        if self.gates:
            device = next(self.parameters()).device
        else:
            # Fallback for an empty circuit, can be cpu or a globally defined device
            from .qgates import device as default_device
            device = default_device

        # Start with the identity matrix for the total number of qubits, on the correct device.
        matrix = Identity(self.size).to(device)

        # Apply each gate in sequence
        for gate in self.gates:
            # Get the matrix for the current gate
            g = gate(self.size)
            # Left-multiply the gate to the total circuit matrix
            matrix = torch.matmul(g, matrix)

        return matrix

    def check_circuit(self):
        """Validates that gate qubits are within the circuit size."""
        for i, gate in enumerate(self.gates):
            if gate.qubit1 is not None and gate.qubit1 >= self.size:
                raise IndexError(f"Error: Gate #{i} ({gate.name}) has qubit1 out of range.")
            if gate.qubit2 is not None and gate.qubit2 >= self.size:
                raise IndexError(f"Error: Gate #{i} ({gate.name}) has qubit2 out of range.")