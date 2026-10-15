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
"""Cost and Fidelity Functions — PyTorch version.

Reduced from the original cost_functions.py:
    - braket()                    -> REMOVED (replaced by torch.vdot _calc_wasserstein)
    - compute_fidelity()          -> kept, torch version
    - compute_fidelity_and_cost() -> kept, calls dis.compute_loss
"""

import numpy as np
import torch

from config import CFG


def _calc_wasserstein(g, t, dis_matrices):
    """Computation of the Wasserstein loss.
    Instead of repeating the code, we create this function to reuse
    in Choi and Haar method.

    g: pure generator state (d,) / (d, 1), or (n_branches, d) unnormalised branches g_j of the mixed
       state rho = sum_j |g_j><g_j| (ancilla traced out, see qgan.ancilla.gen_branches).
    """
    A, B, psi, phi = dis_matrices
    if g.dim() == 2 and g.shape[1] != 1:
        return _calc_wasserstein_mixed(g, t.reshape(-1), dis_matrices)
    g = g.reshape(-1)
    t = t.reshape(-1)

    Ag = A @ g
    Bg = B @ g
    At = A @ t
    Bt = B @ t

    # <g|A|g> · <t|B|t>
    term1 = torch.vdot(g, Ag) * torch.vdot(t, Bt)
    # cross terms: <t|A|g><g|B|t> + <g|A|t><t|B|g>
    term2 = torch.vdot(t, Ag) * torch.vdot(g, Bt)
    term3 = torch.vdot(g, At) * torch.vdot(t, Bg)
    # <t|A|t> · <g|B|g>
    term4 = torch.vdot(t, At) * torch.vdot(g, Bg)

    psiterm = torch.vdot(t, psi @ t)
    phiterm = torch.vdot(g, phi @ g)

    regterm = (CFG.lamb / np.e) * (CFG.cst1 * term1 - CFG.cst2 * (term2 + term3) + CFG.cst3 * term4)
    return (psiterm - phiterm - regterm).real


def _calc_wasserstein_mixed(Gb, t, dis_matrices):
    """W(rho, |t><t|) for rho = sum_j |g_j><g_j| (rows of Gb). Every term is linear in rho:
        Tr(rho phi) = sum_j <g_j|phi|g_j>,  Tr(rho A) Tr(sigma B),  Tr(A rho B sigma) = sum_j <t|A|g_j><g_j|B|t>, ...
    so this equals the pure-state formula when Gb has a single row.
    """
    A, B, psi, phi = dis_matrices
    Gc = Gb.conj()
    AG, BG, PG = Gb @ A.T, Gb @ B.T, Gb @ phi.T  # row j = A g_j, B g_j, phi g_j
    At, Bt = A @ t, B @ t
    gAg = (Gc * AG).sum()
    gBg = (Gc * BG).sum()
    phiterm = (Gc * PG).sum()
    tAg = AG @ t.conj()  # <t|A|g_j>
    gBt = Gc @ Bt  # <g_j|B|t>
    term1 = gAg * torch.vdot(t, Bt)
    term23 = 2 * (tAg * gBt).real.sum()  # sum_j <t|A|g_j><g_j|B|t> + c.c.
    term4 = torch.vdot(t, At) * gBg
    psiterm = torch.vdot(t, psi @ t)
    regterm = (CFG.lamb / np.e) * (CFG.cst1 * term1 - CFG.cst2 * term23 + CFG.cst3 * term4)
    return (psiterm - phiterm - regterm).real


def batched_wasserstein(G: torch.Tensor, t: torch.Tensor, dis_matrices) -> torch.Tensor:
    """Vectorised `_calc_wasserstein` for a batch of generator states sharing one discriminator.

    Args:
        G: (N, d) generator states (row n = |g_n>).
        t: (d,) common target state, or (N, d) one target per row.
        dis_matrices: (A, B, psi, phi) of a single discriminator.

    Returns:
        (N,) real tensor, entry n = _calc_wasserstein(G[n], t[n], dis_matrices).

    G may also be (N, n_branches, d): mixed states rho_n = sum_j |g_nj><g_nj| (ancilla_mode "trace"), entry n =
    _calc_wasserstein_mixed(G[n], t[n], dis_matrices).
    """
    if G.dim() == 3:
        return _batched_wasserstein_mixed(G, t, dis_matrices)
    A, B, psi, phi = dis_matrices
    T = t.unsqueeze(0).expand_as(G) if t.dim() == 1 else t
    Gc, Tc = G.conj(), T.conj()
    AG, BG, PG = G @ A.T, G @ B.T, G @ phi.T  # row n = A g_n, B g_n, phi g_n
    AT, BT, ST = T @ A.T, T @ B.T, T @ psi.T  # row n = A t_n, B t_n, psi t_n
    term1 = (Gc * AG).sum(1) * (Tc * BT).sum(1)  # <g|A|g><t|B|t>
    term2 = (Tc * AG).sum(1) * (Gc * BT).sum(1)  # <t|A|g><g|B|t>
    term3 = (Gc * AT).sum(1) * (Tc * BG).sum(1)  # <g|A|t><t|B|g>
    term4 = (Tc * AT).sum(1) * (Gc * BG).sum(1)  # <t|A|t><g|B|g>
    psiterm = (Tc * ST).sum(1)
    phiterm = (Gc * PG).sum(1)
    regterm = (CFG.lamb / np.e) * (CFG.cst1 * term1 - CFG.cst2 * (term2 + term3) + CFG.cst3 * term4)
    return (psiterm - phiterm - regterm).real


def _batched_wasserstein_mixed(Gb: torch.Tensor, t: torch.Tensor, dis_matrices) -> torch.Tensor:
    """Vectorised _calc_wasserstein_mixed for (N, n_branches, d) branch states sharing one discriminator."""
    A, B, psi, phi = dis_matrices
    N, nb, d = Gb.shape
    T = t.unsqueeze(0).expand(N, -1) if t.dim() == 1 else t
    Gc, Tc = Gb.conj(), T.conj()
    AG, BG, PG = Gb @ A.T, Gb @ B.T, Gb @ phi.T  # (N, nb, d)
    AT, BT, ST = T @ A.T, T @ B.T, T @ psi.T  # (N, d)
    gAg = (Gc * AG).sum((1, 2))
    gBg = (Gc * BG).sum((1, 2))
    phiterm = (Gc * PG).sum((1, 2))
    tAg = (AG * Tc.unsqueeze(1)).sum(-1)  # <t|A|g_j>, (N, nb)
    gBt = (Gc * BT.unsqueeze(1)).sum(-1)  # <g_j|B|t>
    term1 = gAg * (Tc * BT).sum(1)
    term23 = 2 * (tAg * gBt).real.sum(1)
    term4 = (Tc * AT).sum(1) * gBg
    psiterm = (Tc * ST).sum(1)
    regterm = (CFG.lamb / np.e) * (CFG.cst1 * term1 - CFG.cst2 * term23 + CFG.cst3 * term4)
    return (psiterm - phiterm - regterm).real


def compute_fidelity(final_target_state: torch.Tensor, final_gen_state: torch.Tensor) -> float:
    """Calculate the fidelity between target and generated states.

    Fidelity = |<target|gen>|^2 (for pure states)

    Args:
        final_target_state: Target state, shape (d,) or (d, 1).
        final_gen_state: Generator state, shape (d,) or (d, 1).

    Returns:
        float: fidelity ∈ [0, 1].
    """
    t = final_target_state.reshape(-1)
    if final_gen_state.dim() == 2 and final_gen_state.shape[1] != 1:  # branches: <t|rho|t> = sum_j |<t|g_j>|^2
        return float((final_gen_state @ t.conj()).abs().pow(2).sum())
    g = final_gen_state.reshape(-1)
    overlap = torch.vdot(t, g)  # better to use .vdot than .dot(t.conj(), g)
    return float(torch.abs(overlap) ** 2)


def compute_fidelity_and_cost(
    dis, final_target_state: torch.Tensor, final_gen_state: torch.Tensor
) -> tuple[float, float]:
    """Calculate both fidelity and cost for logging.

    Args:
        dis: Discriminator (torch version).
        final_target_state: Target state.
        final_gen_state: Generator state.

    Returns:
        (fidelity, cost): both as floats.
    """
    fidelity = compute_fidelity(final_target_state, final_gen_state)
    dis_matrices = dis.get_dis_matrices_rep()
    cost = _calc_wasserstein(final_gen_state, final_target_state, dis_matrices)
    return fidelity, cost.item()
