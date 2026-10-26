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

get_max_entangled_state_with_ancilla_if_needed(size) -> (gen_state, target_state)
gen_branches(state)                                   -> what the discriminator sees (see below)
as_gen_state(x)                                       -> canonical 1D state or (n_branches, d) branches
ancilla_block_diagnostics(K00, K10, V)                -> post-selection probability, fidelities, purity, ...

What the discriminator sees, by CFG.ancilla_mode (ancilla = last qubit, prepared in |0>):
  "pass"    : the full output state (the critic acts on the ancilla too; target ⊗ |0>).
  "project" : the ancilla-|0> branch g_0 = (I ⊗ <0|)|g>, post-selection probability p_0 = ||g_0||^2;
              ancilla_project_norm = "re-norm": g_0 / sqrt(p_0) (post-selected state), "pass": g_0 as is.
  "trace"   : the reduced state rho = sum_j |g_j><g_j|, g_j = (I ⊗ <j|)|g>, passed as the two unnormalised
              branches (2, d); every term of the Wasserstein cost is linear in rho, so it is evaluated
              exactly and differentiably as a sum over branches (cost_functions._calc_wasserstein).
"""

import numpy as np
import pennylane as qml
import torch

from config import CFG
from qgan.precision import cdtype, rdtype


# -- MAXIMALLY ENTANGLED STATE PREPARATION -------------------
def get_max_entangled_state_with_ancilla_if_needed(size: int) -> np.ndarray:
    """Get the maximally entangled state for the system size (With Ancilla if needed).
    Args:
        size (int): the size of the system.

    Returns:
        tuple[np.ndarray]: the maximally entangled states, plus ancilla
        if needed for generation and target.
    """
    # Generate the maximally entangled state for the system size
    state = np.zeros(2 ** (2 * size), dtype=complex)
    dim_register = 2**size
    for i in range(dim_register):
        state[i * dim_register + i] = 1.0
    state /= np.sqrt(dim_register)

    # Add ancilla qubit at the end, if needed
    initial_state_with_ancilla = np.kron(state, np.array([1, 0], dtype=complex))

    # Different conditions for gen and target:
    initial_state_for_gen = initial_state_with_ancilla if CFG.extra_ancilla else state
    initial_state_for_target = initial_state_with_ancilla if CFG.extra_ancilla and CFG.ancilla_mode == "pass" else state

    return np.asmatrix(initial_state_for_gen).T, np.asmatrix(initial_state_for_target).T


# -- HAAR RANDOM STATE PREPARATION -------------------
def haar_random_batch(dim: int, batch_size: int) -> list[torch.Tensor]:
    """Generate batch of Haar-random pure states, with ancilla if needed.
    In this case, we reuse the vector for the target (we multiply the target Unitary to the vector)
    and for the generator (we transform vector to quantum state using qml.StatePrep)

    Args:
        dim: Dimension of the system Hilbert space (2^system_size).
        batch_size: Number of states to generate.

    Returns:
        List of torch tensors, each of shape (dim,) or (2*dim,) with ancilla.
    """
    ancilla_zero = torch.tensor([1.0, 0.0], dtype=cdtype())
    batch_raw = []
    batch_inputs = []
    for _ in range(batch_size):
        real = torch.randn(dim, dtype=rdtype())
        imag = torch.randn(dim, dtype=rdtype())
        v = torch.complex(real, imag)
        v = v / v.norm()
        batch_raw.append(v)
        batch_inputs.append(torch.kron(v, ancilla_zero) if CFG.extra_ancilla else v)
    return batch_raw, batch_inputs

# -- DETERMINISTIC ALL-PAIRS STATE SET ---------------------
# Cached module-level: built once on first use, reused across all iterations.
# A new build is triggered only when dim changes (e.g. between experiments).
_DET_STATES: list[torch.Tensor] | None = None
_DET_DIM: int | None = None
 
 
def _build_deterministic_state_set(dim: int) -> list[torch.Tensor]:
    """Build the all-pairs deterministic state set for unitary identification.

    Args:
        dim: Hilbert space dimension d = 2^n.
 
    Returns:
        List of d(d+1)/2 torch tensors of shape (dim,), dtype complex64.
        Order: |0>, |1>, ..., |d-1>, (|0>+|1>)/sqrt(2), (|0>+|2>)/sqrt(2), ...
    """
    states: list[torch.Tensor] = []
 
    # Basis states |j>
    for j in range(dim):
        v = torch.zeros(dim, dtype=cdtype())
        v[j] = 1.0
        states.append(v)
 
    # Pair superpositions 
    inv_sqrt2 = 1.0 / np.sqrt(2.0)
    for i in range(dim):
        for j in range(i + 1, dim):
            v = torch.zeros(dim, dtype=cdtype())
            v[i] = inv_sqrt2
            v[j] = inv_sqrt2
            states.append(v)
 
    return states
 
def deterministic_batch_size(dim: int) -> int:
    """Return M = d(d+1)/2, the size of the deterministic all-pairs set."""
    return dim * (dim + 1) // 2
 
def comp_random_batch(dim: int, batch_size: int) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Return a random subset of the deterministic all-pairs state set.

    The full set has size M = d(d+1)/2. If `batch_size >= M` (or <= 0) the
    full set is returned; otherwise `batch_size` states are drawn uniformly
    at random without replacement from the cached set.

    Args:
        dim: Hilbert space dimension d = 2^n.
        batch_size: Number of states to draw, in [1, M]. Values >= M return
                    the full set.

    Returns:
        (batch_raw, batch_inputs):
            batch_raw    : list of dim-vectors (no ancilla), length = batch_size.
            batch_inputs : list of (dim,) or (2*dim,) vectors depending on
                           CFG.extra_ancilla; ancilla is kron'd with |0⟩.
    """
    global _DET_STATES, _DET_DIM
    # Build (or rebuild if dim changed) the cached state set.
    if _DET_STATES is None or _DET_DIM != dim:
        _DET_STATES = _build_deterministic_state_set(dim)
        _DET_DIM = dim

    max_bs = deterministic_batch_size(dim)
    if batch_size <= 0 or batch_size >= max_bs:
        selected = list(_DET_STATES)
    else:
        idx = torch.randperm(max_bs)[:batch_size].tolist()
        selected = [_DET_STATES[i] for i in idx]

    ancilla_zero = torch.tensor([1.0, 0.0], dtype=cdtype())
    batch_raw: list[torch.Tensor] = list(selected)
    batch_inputs: list[torch.Tensor] = [
        torch.kron(v, ancilla_zero) if CFG.extra_ancilla else v for v in selected
    ]
    return batch_raw, batch_inputs

def get_random_batch(dim: int, batch_size: int) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Dispatch to the input-state sampler selected by CFG.batch_mode."""
    if CFG.batch_mode == "haar":
        return haar_random_batch(dim, batch_size)
    if CFG.batch_mode == "comp":
        return comp_random_batch(dim, batch_size)
    raise ValueError(f"Unknown batch_mode: {CFG.batch_mode}")

def prepare_batch_targets(
    batch_raw: list[torch.Tensor],
    batch_inputs: list[torch.Tensor],
    target_op: torch.Tensor,
) -> list[torch.Tensor]:
    r"""Give us the target state
    If ancilla_mode == "pass":
        target_i = (U_target \otimes I_2) |\psi_i> \otimes |0>
    Otherwise:
        target_i = U_target |\psi_i>

    Args:
        batch_raw               : States without ancilla (dim 2^n).
        batch_inputs            : States with ancilla
        target_op (torch tensor): Pre-computed target operator.

    Returns:
        List of target state vectors, each 1D torch tensor.
    """
    if CFG.extra_ancilla and CFG.ancilla_mode == "pass":
        return [(target_op @ psi).reshape(-1) for psi in batch_inputs]
    else:
        return [(target_op @ psi).reshape(-1) for psi in batch_raw]

# -- ANCILLA POST-PROCESSING -----------------------------
def as_gen_state(x: torch.Tensor) -> torch.Tensor:
    """Canonical generator-state representation: 1D pure state, or (n_branches, d) branch matrix.

    Column vectors (d, 1) are flattened; (n_branches, d) branch matrices are kept.
    """
    if x.dim() == 2 and x.shape[1] == 1:
        return x.reshape(-1)
    return x


def gen_branches(total_output_state: torch.Tensor) -> torch.Tensor:
    """Generator output as seen by the discriminator (see the module docstring)."""
    state = total_output_state.reshape(-1).to(cdtype())
    if not CFG.extra_ancilla or CFG.ancilla_mode == "pass":
        return state
    if CFG.ancilla_mode == "project":
        g0 = state[::2]  # ancilla = last qubit: even indices = ancilla |0>
        if CFG.ancilla_project_norm == "re-norm":
            p0 = (g0.conj() * g0).real.sum()
            g0 = g0 / torch.sqrt(torch.clamp(p0, min=1e-30))
        elif CFG.ancilla_project_norm != "pass":
            raise ValueError(f"Unknown ancilla_project_norm: {CFG.ancilla_project_norm}")
        return g0
    if CFG.ancilla_mode == "trace":
        return torch.stack([state[::2], state[1::2]])
    raise ValueError(f"Unknown ancilla_mode: {CFG.ancilla_mode}")


def gen_branches_batched(total_output_states: torch.Tensor) -> torch.Tensor:
    """gen_branches row by row, for (R, D) generator outputs: (R, d) states, or (R, 2, d) branches for "trace"."""
    state = total_output_states.reshape(total_output_states.shape[0], -1).to(cdtype())
    if not CFG.extra_ancilla or CFG.ancilla_mode == "pass":
        return state
    if CFG.ancilla_mode == "project":
        g0 = state[:, ::2]
        if CFG.ancilla_project_norm == "re-norm":
            p0 = (g0.conj() * g0).real.sum(1, keepdim=True)
            g0 = g0 / torch.sqrt(torch.clamp(p0, min=1e-30))
        elif CFG.ancilla_project_norm != "pass":
            raise ValueError(f"Unknown ancilla_project_norm: {CFG.ancilla_project_norm}")
        return g0
    if CFG.ancilla_mode == "trace":
        return torch.stack([state[:, ::2], state[:, 1::2]], dim=1)
    raise ValueError(f"Unknown ancilla_mode: {CFG.ancilla_mode}")


def ancilla_block_diagnostics(K00: torch.Tensor, K10: torch.Tensor, V: torch.Tensor) -> dict:
    """Blocks K_j0 = <j|U_gen|0> (ancilla in |0>, out |j>) vs the target unitary V (d x d).

    Returns (averaged over the maximally mixed / Choi input):
        p0       : post-selection probability  Tr(K00^† K00)/d
        F_joint  : |Tr(V^† K00)|^2 / d^2          (ancilla must come back to |0>; = "pass" fidelity)
        F_post   : F_joint / p0                   (fidelity after post-selecting the ancilla on |0>)
        F_trace  : sum_j |Tr(V^† K_j0)|^2 / d^2   (entanglement fidelity of the channel, ancilla discarded)
        purity_anc : Tr(rho_a^2) of the ancilla's reduced state (1 = ancilla not entangled with the rest)
        k00_sv_ratio : smallest / largest singular value of K00 (1 = K00 proportional to a unitary)
    """
    d = V.shape[0]
    if not (torch.isfinite(K00).all() and torch.isfinite(K10).all()):  # a diverged run (NaN state): no crash
        return dict.fromkeys(("p0", "F_joint", "F_post", "F_trace", "purity_anc", "k00_sv_ratio"), float("nan"))
    with torch.no_grad():
        n00 = (K00.conj() * K00).real.sum() / d
        n11 = (K10.conj() * K10).real.sum() / d
        c01 = (K10.conj() * K00).sum() / d  # Tr(K10^† K00)/d
        a0 = torch.trace(V.conj().T @ K00) / d
        a1 = torch.trace(V.conj().T @ K10) / d
        sv = torch.linalg.svdvals(K00)
        p0 = float(n00)
        F_joint = float(a0.abs() ** 2)
        return {
            "p0": p0,
            "F_joint": F_joint,
            "F_post": F_joint / max(p0, 1e-30),
            "F_trace": float(a0.abs() ** 2 + a1.abs() ** 2),
            "purity_anc": float(n00**2 + n11**2 + 2 * c01.abs() ** 2),
            "k00_sv_ratio": float(sv.min() / sv.max()) if float(sv.max()) > 0 else 0.0,
        }


def choi_blocks(total_output_state: torch.Tensor, system_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """K00, K10 from the Choi output (I ⊗ U)|Φ+>|0> (wires: reference | system | ancilla)."""
    d = 2**system_size
    state = total_output_state.reshape(-1)
    g0, g1 = state[::2].reshape(d, d), state[1::2].reshape(d, d)
    # <i|_ref <j|_sys (I ⊗ K)|Φ+> = K_{ji} / sqrt(d)
    return np.sqrt(d) * g0.T, np.sqrt(d) * g1.T



def get_final_gen_state_torch(total_output_state: torch.Tensor) -> torch.Tensor:
    """Process the generator output state according to ancilla_mode (see gen_branches).

    Returns a column vector (d, 1) for "pass"/"project"/no ancilla, and the (2, d) branch matrix for
    "trace". Use as_gen_state() to get the canonical form (1D state or branch matrix).
    """
    out = gen_branches(total_output_state)
    return out.reshape(-1, 1) if out.dim() == 1 else out

def extract_gen_unitary(gen) -> torch.Tensor:
    """Extract U_gen as a (D, D) torch tensor using qml.matrix.

    D = 2^(system_size + 1) if ancilla is present and ancilla_mode == "pass",
    otherwise D = 2^system_size.

    Only meaningful when the post-ancilla map is linear (i.e. ancilla_mode == "pass"
    or no ancilla). For "project"/"trace" the generator is not a unitary.
    """
    from qgan.generator import Ansatz, _wire_layout

    _, gen_wires, ancilla_wire, total_wires = _wire_layout()

    def ansatz_only(params):
        Ansatz.apply(params, gen_wires, ancilla_wire)

    with torch.no_grad():
        U = qml.matrix(ansatz_only, wire_order=list(range(total_wires)))(
            gen._effective_params()
        )
    return U.to(cdtype())


def process_fidelity(U_gen: torch.Tensor, U_target: torch.Tensor) -> float:
    """F_proc = |Tr(U_target^† U_gen)|^2 / d^2.

    ⟨U_target | U_gen⟩_HS normalised. Equals the Choi-state fidelity.
    """
    d = U_target.shape[0]
    tr = torch.trace(U_target.conj().T @ U_gen)
    return (tr.abs() ** 2 / d ** 2).item()

# -- ENTANGLEMENT DIAGNOSTICS (ported from upstream `entanglement_entropy` + `negativity` branches) --
# All functions below are pure numpy and act on plain state vectors / unitaries, so they are
# identical to the upstream implementation; only the way U_gen is obtained changed (qml.matrix).
def get_ancilla_reduced_density_matrix(total_output_state) -> np.ndarray:
    """Return the 2x2 reduced density matrix of the last qubit (ancilla) of a pure state vector."""
    if isinstance(total_output_state, torch.Tensor):
        total_output_state = total_output_state.detach().cpu().numpy()
    state = np.asarray(total_output_state).flatten().astype(np.complex128)
    if state.size % 2 != 0:
        raise ValueError("The total output state dimension must be even to trace out the ancilla qubit.")

    reshaped = state.reshape(-1, 2)
    rho = reshaped.conj().T @ reshaped  # = rho_anc^T, same spectrum
    tr = np.real_if_close(np.trace(rho), tol=1000)
    if not np.isclose(tr, 1.0, rtol=1e-2, atol=1e-2):
        raise ValueError(f"Reduced density matrix trace must be close to 1, but got {tr!r}.")
    rho = rho / tr
    return (rho + rho.conj().T) / 2


def compute_ancilla_entanglement_entropy(total_output_state) -> float:
    """Von Neumann entropy S(rho_anc) = -Tr(rho log2 rho) of the ancilla (last qubit)."""
    if not np.isfinite(np.asarray(total_output_state)).all():  # a diverged run (NaN state): no value, no crash
        return float("nan")
    eigvals = np.real(np.linalg.eigvalsh(get_ancilla_reduced_density_matrix(total_output_state)))
    if np.any(eigvals < -1e-6) or np.any(eigvals > 1.0 + 1e-6):  # looser than upstream: float32 states
        raise ValueError(f"Eigenvalues of reduced density matrix must be in [0, 1], but got: {eigvals}")
    eigvals = np.clip(eigvals, 0.0, 1.0)
    non_zero = eigvals > 1e-12
    if not np.any(non_zero):
        return 0.0
    return float(-np.sum(eigvals[non_zero] * np.log2(eigvals[non_zero])))


def compute_bipartite_negativity(total_output_state, global_i: int, global_j: int) -> float:
    """Negativity N = (||rho_ij^{T_i}||_1 - 1)/2 between two qubits of a pure state vector."""
    if not np.isfinite(np.asarray(total_output_state)).all():  # a diverged run (NaN state): no value, no crash
        return float("nan")
    state = np.asarray(total_output_state).flatten()
    num_qubits = int(np.log2(state.size))
    if 2**num_qubits != state.size:
        raise ValueError("State dimension is not a power of 2.")

    keep = sorted([global_i, global_j])
    trace_idx = [q for q in range(num_qubits) if q not in keep]
    psi = state.reshape([2] * num_qubits)
    rho = np.tensordot(psi, psi.conj(), axes=(trace_idx, trace_idx))  # axes (i, j, i', j')
    rho_pt = np.transpose(rho, (2, 1, 0, 3)).reshape(4, 4)  # partial transpose on the first qubit
    eigvals = np.linalg.eigvalsh(rho_pt)
    neg = eigvals[eigvals < -1e-12]
    return 0.0 if len(neg) == 0 else float(np.sum(np.abs(neg)))


def get_random_product_states(size: int, num_states: int = 50, fixed_ancilla: bool = False) -> list[np.ndarray]:
    """Haar-random single-qubit product states (last qubit fixed to |0> if fixed_ancilla)."""
    states = []
    for _ in range(num_states):
        state = None
        for i in range(size):
            if fixed_ancilla and i == size - 1:
                v = np.array([1, 0], dtype=complex).reshape(2, 1)
            else:
                v = np.random.randn(2) + 1j * np.random.randn(2)
                v = (v / np.linalg.norm(v)).reshape(2, 1)
            state = v if state is None else np.kron(state, v)
        states.append(state)
    return states


def negativity_pairs(system_size: int, extra_ancilla: bool) -> list[tuple[str, int, int]]:
    """Qubit pairs of the generator register with upstream labels ('1-2', ..., '1-a', ...).

    Generalises the upstream labels, which were hard-coded for system_size == 3.
    """
    pairs = [(f"{i + 1}-{j + 1}", i, j) for i in range(system_size) for j in range(i + 1, system_size)]
    if extra_ancilla:
        pairs += [(f"{i + 1}-a", i, system_size) for i in range(system_size)]
    return pairs


def compute_negativities(gen, neg_dict: dict[str, list]) -> dict[str, list]:
    """Append the mean pairwise negativity of U_gen over random product inputs to neg_dict.

    Same quantity as upstream `compute_negativities` (ancilla input fixed to |0>), with U_gen
    taken from the PennyLane circuit (`gen.get_gen_unitary()`, wires = system [+ ancilla]).
    """
    if gen.size < 2:
        return neg_dict
    pairs = negativity_pairs(CFG.system_size, CFG.extra_ancilla)
    U_gen = np.asarray(gen.get_gen_unitary(), dtype=np.complex128)
    states = get_random_product_states(gen.size, num_states=CFG.negativity_n_states, fixed_ancilla=CFG.extra_ancilla)
    sums = {label: 0.0 for label, _, _ in pairs}
    for state in states:
        out = U_gen @ state
        for label, i, j in pairs:
            sums[label] += compute_bipartite_negativity(out, i, j)
    for label in sums:
        neg_dict.setdefault(label, []).append(sums[label] / len(states))
    return neg_dict
