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
"""Ancilla post-processing tools — PennyLane rewrite.

Unchanged:
    get_max_entangled_state_with_ancilla_if_needed(size) -> (gen_state, target_state)
    project_ancilla_zero(state, renormalize)              -> (projected_state, prob)
    trace_out_ancilla(state)                              -> sampled_state
    get_final_gen_state_for_discriminator(state)           -> final_state
"""

import numpy as np
import pennylane as qml
import torch 
import torch.nn as nn
from config import CFG


# -- MAXIMALLY ENTANGLED STATE PREPARATION -------------------
def get_max_entangled_state_torch(size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Prepare |\Phi^+> on 2*size qubits, optionally tensored with |0> for ancilla.

    Returns torch tensors (no gradient needed, since they are fixed input states).

    Args:
        size: Number of qubits per register (system_size).

    Returns:
        (initial_state_for_gen, initial_state_for_target):
            Both as column vectors, shape (d, 1), complex128.
    """
    n_choi = 2 * size
    dev = qml.device("default.qubit", wires=n_choi)

    @qml.qnode(dev, interface="numpy")
    def bell_state_circuit():
        for i in range(size):
            qml.Hadamard(wires=i)
            qml.CNOT(wires=[i, i + size])
        return qml.state()

    state_np = np.array(bell_state_circuit(), dtype=complex)
    state = torch.tensor(state_np, dtype=torch.complex128)

    # Add ancilla |0> if needed
    ancilla_zero = torch.tensor([1.0, 0.0], dtype=torch.complex128)
    state_with_ancilla = torch.kron(state, ancilla_zero)

    initial_for_gen = state_with_ancilla if CFG.extra_ancilla else state
    initial_for_target = (
        state_with_ancilla if CFG.extra_ancilla and CFG.ancilla_mode == "pass" else state
    )

    return initial_for_gen.reshape(-1, 1), initial_for_target.reshape(-1, 1)

# -- ANCILLA POST-PROCESSING -----------------------------
def _project_ancilla_zero_torch(state: torch.Tensor,
                                 renormalize: bool = True
                                 ) -> tuple[torch.Tensor, torch.Tensor]:
    """Project the last qubit onto |0> (torch version, differentiable).

    Keeps only the amplitudes where the ancilla is in |0> (even indices).
    This is differentiable through autograd since it's just slicing + division.

    Args:
        state: Statevector as a 1D torch tensor, shape (2^N,).
        renormalize: Whether to renormalize the projected state.

    Returns:
        (projected_state, prob_zero):
            projected_state: shape (2^(N-1), 1) column vector.
            prob_zero: scalar, probability of ancilla being |0>.
    """
    # Even indices = ancilla in |0>
    projected = state[::2]

    # Norm^2 = probability of |0>
    prob = torch.sum(projected.conj() * projected).real

    if prob.item() < 1e-15:
        n_system_qubits = CFG.system_size * 2  # choi + system register
        return (
            torch.zeros(2**n_system_qubits, 1, dtype=torch.complex128),
            torch.tensor(0.0, dtype=torch.float64),
        )

    if renormalize:
        if CFG.ancilla_project_norm == "re-norm":
            projected = projected / torch.sqrt(prob)
        elif CFG.ancilla_project_norm != "pass":
            raise ValueError(f"Unknown ancilla_project_norm: {CFG.ancilla_project_norm}")

    return projected.reshape(-1, 1), prob


def _trace_out_ancilla_torch(state: torch.Tensor) -> torch.Tensor:
    """Trace out the last qubit and return a sampled pure state (torch version).

    This operation involves eigendecomposition + stochastic sampling,
    which breaks differentiability. Same limitation as the original numpy version.
    (Ask Guille if is legal or we can do it better)

    Args:
        state: Statevector as a 1D torch tensor, shape (2^N,).

    Returns:
        Sampled pure state, shape (2^(N-1), 1) column vector.
    """
    n_total = state.shape[0]
    n_qubits = int(np.log2(n_total))
    n_system = n_qubits - 1

    # Full density matrix |psi><psi|
    rho_full = torch.outer(state, state.conj())

    # Reshape to (2^n_system, 2, 2^n_system, 2) and trace over ancilla (last qubit)
    dim_sys = 2 ** n_system
    rho_reshaped = rho_full.reshape(dim_sys, 2, dim_sys, 2)
    # Trace over ancilla: sum over ancilla indices (axes 1 and 3)
    rho_reduced = torch.einsum("iaja->ij", rho_reshaped)

    # Sample a pure state from the reduced density matrix
    rho_np = rho_reduced.detach().numpy()
    eigvals, eigvecs = np.linalg.eigh(rho_np)
    eigvals = np.maximum(eigvals, 0)
    eigvals = eigvals / np.sum(eigvals)

    idx = np.random.choice(len(eigvals), p=eigvals)
    sampled = torch.tensor(eigvecs[:, idx], dtype=torch.complex128)

    return sampled.reshape(-1, 1)


def get_final_gen_state_torch(total_output_state: torch.Tensor) -> torch.Tensor:
    """Process the generator output state according to ancilla_mode.

    Routes to the appropriate post-processing:
        - "pass":    state goes directly to discriminator
        - "project": project ancilla onto |0>, remove it 
        - "trace":   trace out ancilla, sample pure state 

    Args:
        total_output_state: Generator output, 1D torch tensor of shape (2^N,).

    Returns:
        Column vector, shape (d, 1), as torch tensor.
    """
    if not CFG.extra_ancilla:
        return total_output_state.reshape(-1, 1)

    if CFG.ancilla_mode == "pass":
        return total_output_state.reshape(-1, 1)
    if CFG.ancilla_mode == "project":
        projected, _ = _project_ancilla_zero_torch(total_output_state)
        return projected
    if CFG.ancilla_mode == "trace":
        return _trace_out_ancilla_torch(total_output_state)

    raise ValueError(f"Unknown ancilla_mode: {CFG.ancilla_mode}")