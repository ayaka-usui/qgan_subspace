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
"""Ancilla post-processing tools using PyTorch."""

import torch
from config import CFG
from tools.qobjects.qgates import device, COMPLEX_TYPE

def get_max_entangled_state_with_ancilla_if_needed(config = CFG) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Get the maximally entangled state for the system size (with Ancilla if needed).
    Returns tensors on the configured device.
    """
    dim_register = 2**config.system_size
    state = torch.zeros(dim_register * dim_register, dtype=COMPLEX_TYPE, device=device)
    # Create the Bell state |Φ+⟩ = 1/√d * Σ |ii⟩
    for i in range(dim_register):
        state[i * dim_register + i] = 1.0
    state /= (dim_register**0.5)

    # Add ancilla qubit |0⟩ at the end, if needed
    ancilla_state = torch.tensor([1, 0], dtype=COMPLEX_TYPE, device=device)
    initial_state_with_ancilla = torch.kron(state, ancilla_state)

    # Determine the correct state for the generator and target based on config
    initial_state_for_gen = initial_state_with_ancilla if config.extra_ancilla else state
    initial_state_for_target = initial_state_with_ancilla if config.extra_ancilla and config.ancilla_mode == "pass" else state

    # Return as column vectors
    return initial_state_for_gen.view(-1, 1), initial_state_for_target.view(-1, 1)


def project_ancilla_zero(state: torch.Tensor, renormalize: bool = True, config = CFG) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Project the last qubit of a state vector onto |0> and renormalize.
    Assumes state is a column vector.
    """
    state = state.flatten()

    # Keep only the elements corresponding to the ancilla being in the |0> state
    projected = state[::2]

    # Compute the norm of the projected state (probability of measuring |0>)
    norm = torch.linalg.norm(projected)
    prob = norm**2

    if norm < 1e-9:  # Check for near-zero norm
        return torch.zeros((2**(config.system_size * 2), 1), dtype=COMPLEX_TYPE, device=device), prob

    # Renormalize if specified
    if renormalize:
        if config.ancilla_project_norm == "re-norm":
            projected = projected / norm
        elif config.ancilla_project_norm != "pass":
            raise ValueError(f"Unknown ancilla_project_norm: {config.ancilla_project_norm}")

    return projected.view(-1, 1), prob


def trace_out_ancilla(state: torch.Tensor) -> torch.Tensor:
    """
    Trace out the last qubit and return a sampled pure state from the
    reduced density matrix.
    """
    state = state.flatten()
    # Reshape to (2**(n-1), 2) to separate the last qubit
    state = state.view(-1, 2)

    # Compute the reduced density matrix rho = Tr_anc( |psi⟩⟨psi| )
    # This is equivalent to Σ_i ( psi_i * psi_i^† ) where psi_i are the columns
    rho_reduced = torch.matmul(state, state.conj().T)

    # Sample a pure state from the reduced density matrix's spectral decomposition
    eigvals, eigvecs = torch.linalg.eigh(rho_reduced)
    
    # Ensure probabilities are non-negative and normalized
    probabilities = torch.maximum(eigvals.real, torch.tensor(0.0, device=device))
    probabilities /= torch.sum(probabilities)

    # Sample an index based on the eigenvalue probabilities
    idx = torch.multinomial(probabilities, 1).item()
    sampled_state = eigvecs[:, idx]

    return sampled_state.view(-1, 1)


def get_final_gen_state_for_discriminator(total_output_state: torch.Tensor, config = CFG) -> torch.Tensor:
    """
    Modifies the generator's output state to be passed to the discriminator,
    according to the configured ancilla_mode.
    """
    if not config.extra_ancilla:
        return total_output_state

    if config.ancilla_mode == "pass":
        return total_output_state
    elif config.ancilla_mode == "project":
        projected, _ = project_ancilla_zero(total_output_state)
        return projected
    elif config.ancilla_mode == "trace":
        return trace_out_ancilla(total_output_state)
    else:
        raise ValueError(f"Unknown ancilla_mode: {config.ancilla_mode}")