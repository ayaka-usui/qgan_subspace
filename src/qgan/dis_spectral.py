"""Exact tensor-product (eigenbasis) evaluation of the Wasserstein cost.

The discriminator observables are tensor products of single-qubit Hermitians,

    psi = H(alpha_0) ⊗ ... ⊗ H(alpha_{n-1}),   phi = H(beta_0) ⊗ ... ⊗ H(beta_{n-1}),
    H(c) = c_0 I + c_1 X + c_2 Y + c_3 Z.

Each H(c_i) = V_i diag(m_i) V_i^† (2x2), hence psi = V diag(m) V^† with V = ⊗ V_i and m = ⊗ m_i,
and A = exp(-phi/lambda), B = exp(+psi/lambda) are diagonal in the same product bases.  Every term
of the cost is then a weighted sum over the d = 2^n eigenbasis amplitudes of |g>, |t>:

    <x|A|y> = sum_k exp(-m_phi[k]/lambda) conj(x~_k) y~_k,   x~ = V_phi^† x   (same for B, psi, phi).

V^† x is applied as n successive 2x2 mode products, O(n 2^n) per state, so no d x d matrix and no
matrix exponential is ever formed. Mixed generator states (ancilla_mode "trace": rho = sum_j |g_j><g_j| over the
unnormalised branches g_j) are handled exactly by linearity: every term of W is linear in rho.  This is the same objective as cost_functions._calc_wasserstein
with the dense (A, B, psi, phi) of Discriminator.get_dis_matrices_rep() (equal to machine
precision); it makes evaluating many discriminators cheap, e.g. one per sample.  The training uses it with
CFG.dis_eval = "spectral" (Discriminator.compute_loss through spectral_critic_cost, Generator.compute_loss and the
logged cost through Discriminator.detached_spectrum), and for the runs trained together (qgan.parallel_training).
"""

from __future__ import annotations

import numpy as np
import torch

from config import CFG


def single_qubit_spectra(coeffs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Eigen-decomposition of H(c) = c0 I + c1 X + c2 Y + c3 Z for every row.

    Args:
        coeffs: real array (..., n, 4).

    Returns:
        (w, V): eigenvalues (..., n, 2) and eigenvectors as columns (..., n, 2, 2), H = V diag(w) V^†.
    """
    c = np.asarray(coeffs, dtype=np.float64)
    if c.shape[-1] != 4:
        raise ValueError(f"Expected (..., n, 4) Pauli coefficients, got shape {c.shape}")
    H = np.empty(c.shape[:-1] + (2, 2), dtype=np.complex128)
    H[..., 0, 0] = c[..., 0] + c[..., 3]
    H[..., 1, 1] = c[..., 0] - c[..., 3]
    H[..., 0, 1] = c[..., 1] - 1j * c[..., 2]
    H[..., 1, 0] = c[..., 1] + 1j * c[..., 2]
    w, V = np.linalg.eigh(H)
    return w, V


def kron_eigenvalues(w: np.ndarray) -> np.ndarray:
    """(..., n, 2) single-qubit eigenvalues -> (..., 2^n) eigenvalues of the tensor product.

    Row 0 is the most significant (left-most) Kronecker factor, as in Discriminator.get_psi_and_phi.
    """
    m = w[..., 0, :]
    for i in range(1, w.shape[-2]):
        m = (m[..., :, None] * w[..., i, None, :]).reshape(m.shape[:-1] + (-1,))
    return m


class DisSpectrum:
    """Product eigenbases and eigenvalues of one discriminator, or of N per-sample discriminators.

    Args:
        alpha, beta: (n, 4) for one shared discriminator, or (N, n, 4) for one discriminator per sample.
        rdt, cdt: torch real / complex dtypes of the states the spectrum will be contracted with.
    """

    def __init__(self, alpha: np.ndarray, beta: np.ndarray, rdt: torch.dtype, cdt: torch.dtype):
        alpha, beta = np.asarray(alpha, dtype=np.float64), np.asarray(beta, dtype=np.float64)
        if alpha.shape != beta.shape or alpha.ndim not in (2, 3):
            raise ValueError(f"alpha/beta must both be (n, 4) or (N, n, 4); got {alpha.shape}, {beta.shape}")
        self.batched = alpha.ndim == 3
        self.n = alpha.shape[-2]
        w_psi, V_psi = single_qubit_spectra(alpha)
        w_phi, V_phi = single_qubit_spectra(beta)
        m_psi, m_phi = kron_eigenvalues(w_psi), kron_eigenvalues(w_phi)
        lamb = float(CFG.lamb)
        # Single-qubit V^† (applied to kets); float64 eigensolve, then cast to the working precision.
        self.Vh_psi = torch.tensor(np.conj(np.swapaxes(V_psi, -1, -2)), dtype=cdt)
        self.Vh_phi = torch.tensor(np.conj(np.swapaxes(V_phi, -1, -2)), dtype=cdt)
        self.m_psi = torch.tensor(m_psi, dtype=rdt)
        self.m_phi = torch.tensor(m_phi, dtype=rdt)
        self.b = torch.tensor(np.exp(m_psi / lamb), dtype=rdt)  # spectrum of B = exp(+psi/lambda)
        self.a = torch.tensor(np.exp(-m_phi / lamb), dtype=rdt)  # spectrum of A = exp(-phi/lambda)

    @property
    def dim(self) -> int:
        return 2**self.n

    def to_eigenbasis(self, X: torch.Tensor, which: str) -> torch.Tensor:
        """Rows of X (N, d) mapped to (⊗_i V_i^†) x_n for the 'psi' or 'phi' product basis."""
        Vh = self.Vh_psi if which == "psi" else self.Vh_phi
        N, d = X.shape
        if d != self.dim:
            raise ValueError(f"State dimension {d} does not match discriminator dimension {self.dim}")
        if self.batched and Vh.shape[0] != N:
            raise ValueError(f"{Vh.shape[0]} per-sample discriminators for {N} states")
        Y = X
        for i in range(self.n):
            Y = Y.reshape(N, 2**i, 2, 2 ** (self.n - i - 1))
            if self.batched:
                Y = torch.einsum("nab,nlbr->nlar", Vh[:, i], Y)
            else:
                Y = torch.einsum("ab,nlbr->nlar", Vh[i], Y)
        return Y.reshape(N, d)


def spectral_wasserstein(G: torch.Tensor, t: torch.Tensor, spec: DisSpectrum) -> torch.Tensor:
    """Row-wise Wasserstein cost W(g_n, t_n), identical to cost_functions._calc_wasserstein.

    Args:
        G: (N, d) generator states (may require grad).
        t: (d,) common target state or (N, d) one target per row.
        spec: DisSpectrum of one shared discriminator or of N per-sample discriminators.

    Returns:
        (N,) real tensor.

    G may also be (N, n_branches, d), the branches of a mixed state rho = sum_j |g_j><g_j| ("trace" ancilla mode):
    then the cost is that of rho (cost_functions._calc_wasserstein_mixed), see _spectral_wasserstein_mixed.
    """
    if G.dim() == 3:
        return _spectral_wasserstein_mixed(G, t, spec)
    T = t.unsqueeze(0).expand_as(G) if t.dim() == 1 else t
    gp, tp = spec.to_eigenbasis(G, "phi"), spec.to_eigenbasis(T, "phi")
    gs, ts = spec.to_eigenbasis(G, "psi"), spec.to_eigenbasis(T, "psi")
    a, b = spec.a, spec.b
    if not spec.batched:
        a, b = a.unsqueeze(0), b.unsqueeze(0)
        m_psi, m_phi = spec.m_psi.unsqueeze(0), spec.m_phi.unsqueeze(0)
    else:
        m_psi, m_phi = spec.m_psi, spec.m_phi
    gp2, tp2, gs2, ts2 = gp.abs() ** 2, tp.abs() ** 2, gs.abs() ** 2, ts.abs() ** 2
    gAg, tAt = (a * gp2).sum(1), (a * tp2).sum(1)
    gBg, tBt = (b * gs2).sum(1), (b * ts2).sum(1)
    tAg = (a * tp.conj() * gp).sum(1)  # <t|A|g>
    gBt = (b * gs.conj() * ts).sum(1)  # <g|B|t>
    term1 = gAg * tBt  # <g|A|g><t|B|t>
    cross = 2 * (tAg * gBt).real  # <t|A|g><g|B|t> + <g|A|t><t|B|g>  (A, B Hermitian)
    term4 = tAt * gBg  # <t|A|t><g|B|g>
    psiterm = (m_psi * ts2).sum(1)  # <t|psi|t>
    phiterm = (m_phi * gp2).sum(1)  # <g|phi|g>
    regterm = (CFG.lamb / np.e) * (CFG.cst1 * term1 - CFG.cst2 * cross + CFG.cst3 * term4)
    return psiterm - phiterm - regterm


def _spectral_wasserstein_mixed(Gb: torch.Tensor, t: torch.Tensor, spec: DisSpectrum) -> torch.Tensor:
    """Row-wise W(rho_n, t_n) for rho_n = sum_j |g_nj><g_nj| (Gb: (N, n_branches, d)), as _calc_wasserstein_mixed:
        Tr(rho phi) = sum_j <g_j|phi|g_j>, Tr(rho A) = sum_j <g_j|A|g_j>, Tr(A rho B sigma) = sum_j <t|A|g_j><g_j|B|t>.
    A single branch gives the pure-state cost (the pure path of spectral_wasserstein is kept separate and unchanged).
    """
    N, nb, _ = Gb.shape
    T = t.unsqueeze(0).expand(N, -1) if t.dim() == 1 else t
    tp, ts = spec.to_eigenbasis(T, "phi"), spec.to_eigenbasis(T, "psi")
    a, b = spec.a, spec.b
    if not spec.batched:
        a, b = a.unsqueeze(0), b.unsqueeze(0)
        m_psi, m_phi = spec.m_psi.unsqueeze(0), spec.m_phi.unsqueeze(0)
    else:
        m_psi, m_phi = spec.m_psi, spec.m_phi
    tp2, ts2 = tp.abs() ** 2, ts.abs() ** 2
    tAt, tBt = (a * tp2).sum(1), (b * ts2).sum(1)
    gAg = gBg = phiterm = cross = 0.0
    for j in range(nb):
        gp, gs = spec.to_eigenbasis(Gb[:, j], "phi"), spec.to_eigenbasis(Gb[:, j], "psi")
        gp2, gs2 = gp.abs() ** 2, gs.abs() ** 2
        gAg = gAg + (a * gp2).sum(1)
        gBg = gBg + (b * gs2).sum(1)
        phiterm = phiterm + (m_phi * gp2).sum(1)
        tAg = (a * tp.conj() * gp).sum(1)  # <t|A|g_j>
        gBt = (b * gs.conj() * ts).sum(1)  # <g_j|B|t>
        cross = cross + 2 * (tAg * gBt).real
    psiterm = (m_psi * ts2).sum(1)
    regterm = (CFG.lamb / np.e) * (CFG.cst1 * gAg * tBt - CFG.cst2 * cross + CFG.cst3 * tAt * gBg)
    return psiterm - phiterm - regterm


# ---------------------------------------------------------------------------------------------------
# Exact critic gradient dW/dalpha, dW/dbeta in the product eigenbasis (numpy, float64).
#
# d psi / d alpha_ik = D_ik = h_0 ⊗ .. ⊗ sigma_k (site i) ⊗ .. ⊗ h_{n-1}. In the eigenbasis of psi,
#   <a|D_ik|b> = delta(a_{-i}, b_{-i}) * mu_{-i}(a_{-i}) * M^{ik}_{a_i b_i},   M^{ik} = V_i^† sigma_k V_i,
# with mu_{-i} the product of the other rows' eigenvalues. The derivative of E = exp(s X), X = psi / lambda
# (s = +1 for B; for A = exp(-phi/lambda) use phi and s = -1) is the Daleckii-Krein formula
#   <a| dE |b> = (s/lambda) <a|D_ik|b> * Gamma(x_a, x_b),   Gamma(x, y) = (e^x - e^y)/(x - y),  Gamma(x, x) = e^x,
# x = s m / lambda the exponent's eigenvalues. Exact everywhere, including degenerate (identity) rows.
# ---------------------------------------------------------------------------------------------------
_SIGMA = np.array([[[1, 0], [0, 1]], [[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]], dtype=np.complex128)


def _gamma(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """(e^x - e^y)/(x - y) = e^{(x+y)/2} sinh(u)/u, u = (x - y)/2; stable for x ≈ y."""
    u = 0.5 * (x - y)
    small = np.abs(u) < 1e-6
    shc = np.where(small, 1.0 + u * u / 6.0, np.sinh(u) / np.where(small, 1.0, u))
    return np.exp(0.5 * (x + y)) * shc


def _basis_np(X: np.ndarray, Vh: np.ndarray) -> np.ndarray:
    """(N, 2^n) states -> (⊗_i Vh_i) x for per-sample Vh (N, n, 2, 2)."""
    N, d = X.shape
    n = Vh.shape[1]
    Y = X
    for i in range(n):
        Y = np.einsum("nab,nlbr->nlar", Vh[:, i], Y.reshape(N, 2**i, 2, 2 ** (n - i - 1)))
    return Y.reshape(N, d)


def _kron_rows(w: np.ndarray) -> np.ndarray:
    """(N, k, 2) -> (N, 2^k) Kronecker product of the rows (k = 0 -> ones)."""
    out = np.ones((w.shape[0], 1))
    for j in range(w.shape[1]):
        out = (out[:, :, None] * w[:, j, None, :]).reshape(w.shape[0], -1)
    return out


def _row_terms(pairs, m: np.ndarray, i: int, sign: float, lamb: float):
    """Row-i sums for several (u, v) pairs sharing the same basis, with Gamma computed once.

    For each pair returns (S_lin, S_exp), (N, 2, 2):
        S_lin[pq] = sum_{l,r} mu conj(u_p) v_q,   S_exp[pq] = sum_{l,r} mu Gamma(x_p, x_q) conj(u_p) v_q,
    with mu = mu_{-i}(l, r) and x_p = sign * mu * m_i(p) / lambda. u, v: (N, 2^n) in the eigenbasis.
    """
    N, n = m.shape[0], m.shape[1]
    L, R = 2**i, 2 ** (n - i - 1)
    mu = _kron_rows(m[:, :i])[:, :, None] * _kron_rows(m[:, i + 1:])[:, None, :]  # (N, L, R)
    x0 = sign * mu * m[:, i, 0][:, None, None] / lamb
    x1 = sign * mu * m[:, i, 1][:, None, None] / lamb
    g01 = _gamma(x0, x1)
    gam = ((np.exp(x0), g01), (g01, np.exp(x1)))
    out = []
    for u, v in pairs:
        U, V = u.reshape(N, L, 2, R), v.reshape(N, L, 2, R)
        S_lin = np.empty((N, 2, 2), dtype=np.complex128)
        S_exp = np.empty((N, 2, 2), dtype=np.complex128)
        for p in range(2):
            Up = np.conj(U[:, :, p, :]) * mu
            for q in range(2):
                prod = Up * V[:, :, q, :]
                S_lin[:, p, q] = prod.sum(axis=(1, 2))
                S_exp[:, p, q] = (prod * gam[p][q]).sum(axis=(1, 2))
        out.append((S_lin, S_exp))
    return out


def critic_gradients(G: np.ndarray, T: np.ndarray, alpha: np.ndarray, beta: np.ndarray):
    """Exact dW/dalpha and dW/dbeta (W as in cost_functions._calc_wasserstein), one critic per sample.

    Args:
        G: (N, d) generator states; T: (d,) or (N, d) target states (complex).
        alpha, beta: (N, n, 4) critic rows (one critic per sample) or (n, 4) shared.

    Returns:
        (dW/dalpha, dW/dbeta), each (N, n, 4) float64.

    G may also be (N, n_branches, d), a mixed state rho = sum_j |g_j><g_j| ("trace"): see _critic_gradients_mixed.
    """
    G = np.asarray(G, dtype=np.complex128)
    if G.ndim == 3:
        return _critic_gradients_mixed(G, T, alpha, beta)
    N, d = G.shape
    T = np.broadcast_to(np.asarray(T, dtype=np.complex128), (N, d))
    alpha = np.broadcast_to(np.asarray(alpha, dtype=np.float64), (N,) + np.shape(alpha)[-2:])
    beta = np.broadcast_to(np.asarray(beta, dtype=np.float64), (N,) + np.shape(beta)[-2:])
    n = alpha.shape[1]
    if 2**n != d:
        raise ValueError(f"State dimension {d} does not match {n} critic rows")
    lamb, e = float(CFG.lamb), np.e
    c1, c2, c3 = CFG.cst1, CFG.cst2, CFG.cst3
    w_psi, V_psi = single_qubit_spectra(alpha)
    w_phi, V_phi = single_qubit_spectra(beta)
    Vh_psi, Vh_phi = np.conj(np.swapaxes(V_psi, -1, -2)), np.conj(np.swapaxes(V_phi, -1, -2))
    gs, ts = _basis_np(G, Vh_psi), _basis_np(T, Vh_psi)
    gp, tp = _basis_np(G, Vh_phi), _basis_np(T, Vh_phi)
    m_psi, m_phi = kron_eigenvalues(w_psi), kron_eigenvalues(w_phi)
    b, a = np.exp(m_psi / lamb), np.exp(-m_phi / lamb)
    gAg, tAt = (a * np.abs(gp) ** 2).sum(1), (a * np.abs(tp) ** 2).sum(1)
    gBg, tBt = (b * np.abs(gs) ** 2).sum(1), (b * np.abs(ts) ** 2).sum(1)
    tAg = (a * np.conj(tp) * gp).sum(1)
    gBt = (b * np.conj(gs) * ts).sum(1)
    # M^{ik} = V_i^† sigma_k V_i for every sample, row, k: (N, n, 4, 2, 2)
    M_psi = np.einsum("niab,kbc,nicd->nikad", Vh_psi, _SIGMA, V_psi)
    M_phi = np.einsum("niab,kbc,nicd->nikad", Vh_phi, _SIGMA, V_phi)
    dalpha = np.empty((N, n, 4))
    dbeta = np.empty((N, n, 4))
    pref = lamb / e

    def contract(Mi, S):
        return np.einsum("nkpq,npq->nk", Mi, S)

    for i in range(n):
        # alpha: <t|D|t> - (lambda/e)[c1 gAg <t|dB|t> - 2 c2 Re(tAg <g|dB|t>) + c3 tAt <g|dB|g>],  dB = (1/lambda) D Gamma
        (Lt, Et), (_, Egt), (_, Eg) = _row_terms(((ts, ts), (gs, ts), (gs, gs)), w_psi, i, +1.0, lamb)
        Mi = M_psi[:, i]
        lin = contract(Mi, Lt)
        dB_tt, dB_gt, dB_gg = contract(Mi, Et) / lamb, contract(Mi, Egt) / lamb, contract(Mi, Eg) / lamb
        dalpha[:, i] = (lin - pref * (c1 * gAg[:, None] * dB_tt - 2 * c2 * (tAg[:, None] * dB_gt).real
                                      + c3 * tAt[:, None] * dB_gg)).real
        # beta: -<g|D|g> - (lambda/e)[c1 <g|dA|g> tBt - 2 c2 Re(<t|dA|g> gBt) + c3 <t|dA|t> gBg],  dA = -(1/lambda) D Gamma
        (Lg, Eg2), (_, Etg), (_, Et2) = _row_terms(((gp, gp), (tp, gp), (tp, tp)), w_phi, i, -1.0, lamb)
        Mi = M_phi[:, i]
        lin = contract(Mi, Lg)
        dA_gg, dA_tg, dA_tt = -contract(Mi, Eg2) / lamb, -contract(Mi, Etg) / lamb, -contract(Mi, Et2) / lamb
        dbeta[:, i] = (-lin - pref * (c1 * dA_gg * tBt[:, None] - 2 * c2 * (dA_tg * gBt[:, None]).real
                                      + c3 * dA_tt * gBg[:, None])).real
    return dalpha, dbeta


def _critic_gradients_mixed(Gb: np.ndarray, T: np.ndarray, alpha: np.ndarray, beta: np.ndarray):
    """critic_gradients for mixed states rho = sum_j |g_j><g_j| (Gb: (N, n_branches, d), unnormalised branches).

    Every term of W is linear in rho, so each g-dependent factor of the pure-state derivative becomes a sum over the
    branches (gAg, gBg, <g|D|g>, and the cross term sum_j <t|A|g_j><g_j|dB|t>); the target-only terms are not
    repeated. One branch reproduces critic_gradients (whose pure path is kept separate and unchanged).
    """
    N, nb, d = Gb.shape
    T = np.broadcast_to(np.asarray(T, dtype=np.complex128), (N, d))
    alpha = np.broadcast_to(np.asarray(alpha, dtype=np.float64), (N,) + np.shape(alpha)[-2:])
    beta = np.broadcast_to(np.asarray(beta, dtype=np.float64), (N,) + np.shape(beta)[-2:])
    n = alpha.shape[1]
    if 2**n != d:
        raise ValueError(f"State dimension {d} does not match {n} critic rows")
    lamb, e = float(CFG.lamb), np.e
    c1, c2, c3 = CFG.cst1, CFG.cst2, CFG.cst3
    w_psi, V_psi = single_qubit_spectra(alpha)
    w_phi, V_phi = single_qubit_spectra(beta)
    Vh_psi, Vh_phi = np.conj(np.swapaxes(V_psi, -1, -2)), np.conj(np.swapaxes(V_phi, -1, -2))
    ts, tp = _basis_np(T, Vh_psi), _basis_np(T, Vh_phi)
    gs = [_basis_np(np.ascontiguousarray(Gb[:, j]), Vh_psi) for j in range(nb)]
    gp = [_basis_np(np.ascontiguousarray(Gb[:, j]), Vh_phi) for j in range(nb)]
    m_psi, m_phi = kron_eigenvalues(w_psi), kron_eigenvalues(w_phi)
    b, a = np.exp(m_psi / lamb), np.exp(-m_phi / lamb)
    gAg = sum((a * np.abs(x) ** 2).sum(1) for x in gp)
    gBg = sum((b * np.abs(x) ** 2).sum(1) for x in gs)
    tAt, tBt = (a * np.abs(tp) ** 2).sum(1), (b * np.abs(ts) ** 2).sum(1)
    tAg = [(a * np.conj(tp) * x).sum(1) for x in gp]
    gBt = [(b * np.conj(x) * ts).sum(1) for x in gs]
    M_psi = np.einsum("niab,kbc,nicd->nikad", Vh_psi, _SIGMA, V_psi)
    M_phi = np.einsum("niab,kbc,nicd->nikad", Vh_phi, _SIGMA, V_phi)
    dalpha = np.empty((N, n, 4))
    dbeta = np.empty((N, n, 4))
    pref = lamb / e

    def contract(Mi, S):
        return np.einsum("nkpq,npq->nk", Mi, S)

    for i in range(n):
        pairs = [(ts, ts)] + [(x, ts) for x in gs] + [(x, x) for x in gs]
        terms = _row_terms(pairs, w_psi, i, +1.0, lamb)
        Mi = M_psi[:, i]
        lin = contract(Mi, terms[0][0])
        dB_tt = contract(Mi, terms[0][1]) / lamb
        cross = sum((tAg[j][:, None] * contract(Mi, terms[1 + j][1]) / lamb).real for j in range(nb))
        dB_gg = sum(contract(Mi, terms[1 + nb + j][1]) / lamb for j in range(nb))
        dalpha[:, i] = (lin - pref * (c1 * gAg[:, None] * dB_tt - 2 * c2 * cross + c3 * tAt[:, None] * dB_gg)).real
        pairs = [(tp, tp)] + [(x, x) for x in gp] + [(tp, x) for x in gp]
        terms = _row_terms(pairs, w_phi, i, -1.0, lamb)
        Mi = M_phi[:, i]
        lin = sum(contract(Mi, terms[1 + j][0]) for j in range(nb))
        dA_tt = -contract(Mi, terms[0][1]) / lamb
        dA_gg = sum(-contract(Mi, terms[1 + j][1]) / lamb for j in range(nb))
        cross = sum((-contract(Mi, terms[1 + nb + j][1]) / lamb * gBt[j][:, None]).real for j in range(nb))
        dbeta[:, i] = (-lin - pref * (c1 * dA_gg * tBt[:, None] - 2 * c2 * cross + c3 * dA_tt * gBg[:, None])).real
    return dalpha, dbeta


class _SpectralCriticCost(torch.autograd.Function):
    """W of one critic (alpha, beta) at fixed states: the value from spectral_wasserstein, the gradient with respect to
    alpha and beta from critic_gradients (exact, closed form); no gradient with respect to the states."""

    @staticmethod
    def forward(ctx, alpha: torch.Tensor, beta: torch.Tensor, G: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        a, b = alpha.detach().cpu().numpy(), beta.detach().cpu().numpy()
        W = spectral_wasserstein(G.detach(), T.detach(), DisSpectrum(a, b, alpha.dtype, G.dtype))
        ctx.saved = (G.detach().cpu().numpy(), T.detach().cpu().numpy(), a, b, alpha.dtype)
        return W

    @staticmethod
    def backward(ctx, grad_W: torch.Tensor):
        G, T, a, b, dtype = ctx.saved
        da, db = critic_gradients(G, T, a, b)  # (N, n, 4): dW_n / d(alpha, beta)
        w = grad_W.detach().cpu().numpy().astype(np.float64)
        return (torch.tensor(np.einsum("n,nij->ij", w, da), dtype=dtype),
                torch.tensor(np.einsum("n,nij->ij", w, db), dtype=dtype), None, None)


def spectral_critic_cost(alpha: torch.Tensor, beta: torch.Tensor, G: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
    """(N,) costs W(g_n, t_n) of the critic with rows alpha, beta (n, 4) (differentiable in alpha, beta), for fixed
    states G (N, d) or branches (N, n_branches, d) and targets T (d,) or (N, d): Discriminator.compute_loss with
    CFG.dis_eval = "spectral" (the same values and gradients as through the dense matrix exponentials)."""
    return _SpectralCriticCost.apply(alpha, beta, G, T)


def spectral_cost(g: torch.Tensor, t: torch.Tensor, spec: DisSpectrum) -> torch.Tensor:
    """W of one generator state g (d,), (d, 1) or branches (n_branches, d), target t: _calc_wasserstein's argument
    shapes, for spectral_wasserstein with one critic."""
    if g.dim() == 2 and g.shape[1] == 1:
        g = g.reshape(-1)
    return spectral_wasserstein(g.unsqueeze(0), t.reshape(-1), spec)[0]


def critic_norms(alpha: np.ndarray, beta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Spectral norms ||psi||, ||phi|| = prod_i (|c_i0| + |c_i|) for (..., n, 4) rows."""
    def nrm(c):
        c = np.asarray(c, dtype=np.float64)
        return np.prod(np.abs(c[..., 0]) + np.linalg.norm(c[..., 1:], axis=-1), axis=-1)
    return nrm(alpha), nrm(beta)
