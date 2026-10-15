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
"""Trotter-Suzuki baseline for the QGAN target Hamiltonian.

Standalone comparison script: no training, no ancilla/generator machinery.
The QGAN learns a circuit approximating U = e^{-iHt}; here we approximate the
SAME U with first- and second-order Trotter-Suzuki product formulas and report

    fidelity( U_trotter , U_exact )   vs.   number of entangling gates

so it can be overlaid on the QGAN's fidelity-vs-cost curve.

IMPORTANT — platform-consistent counting:
The entangling cost of a weight-k Pauli rotation is a CNOT/RZZ ladder of length
(k-1). How much each ladder rung costs depends on the hardware:

    superconducting (CNOT-based) : 2 entangling gates per rung -> 2*(k-1)
    trapped-ion     (RZZ native) : 1 entangling gate  per rung -> 2*k-3
"""

from __future__ import annotations

import numpy as np
import pennylane as qml

from config import CFG
import qgan.target as th


# -- SWEEP / OUTPUT CONFIG  --------------------------------------------
N_STEPS_SWEEP = [1, 2, 3, 4, 5, 7, 10]
ORDERS = [1, 2]

# Platforms to count for
PLATFORMS = ("superconducting", "ion-trapped")

# Native entangling gate, by platform
ENT_NAME = {
    "superconducting": "CNOT",
    "ion-trapped": "RZZ",
}

CENTRAL_RZ = {
    "superconducting": 1,
    "ion-trapped": 0,
}

SAVE_PLOT = True
PLOT_PATH = "trotter_vs_exact_fidelity.png"


# -- GET THE HAMILTONIAN --------------------------------------
def build_target_hamiltonian() -> qml.Hamiltonian:
    """Rebuild the exact qml.Hamiltonian the project uses for the target."""
    ttype = CFG.target_hamiltonian
    size = CFG.system_size
    if ttype == "cluster_h":
        return th._cluster_hamiltonian(size)
    if ttype == "ising_h":
        return th._ising_hamiltonian(size)
    if ttype == "rotated_surface_h":
        return th._rotated_surface_code_hamiltonian(size)
    if ttype == "custom_h":
        return th._build_hamiltonian(
            size, CFG.custom_hamiltonian_terms, CFG.custom_hamiltonian_strengths
        )
    raise ValueError(f"Unknown target type: {ttype}")


def extract_terms(H: qml.Hamiltonian, size: int):
    """Read (coeff, pauli_string, wires) for every term of a qml.Hamiltonian."""
    wire_map = {w: i for i, w in enumerate(range(size))}
    coeffs, ops = H.terms()
    out = []
    for c, op in zip(coeffs, ops):
        pstr = qml.pauli.pauli_word_to_string(op, wire_map=wire_map)
        wires = [w for w, ch in zip(range(size), pstr) if ch != "I"]
        out.append((float(c), pstr, wires))
    return out


# -- EXACT AND TROTTER UNITARIES ---------------------------------------------
def exact_unitary() -> np.ndarray:
    """U = e^{-iHt}, taken straight from target.py get_target_unitary."""
    return th.get_target_unitary(CFG.target_hamiltonian, CFG.system_size)


def trotter_unitary(order, n, H, size, t) -> np.ndarray:
    """First-/second-order Trotter-Suzuki for e^{-iHt}, via qml.TrotterProduct."""
    op = qml.TrotterProduct(H, time=-t, n=n, order=order)
    return qml.matrix(op, wire_order=range(size))


# -- FIDELITY ----------------------------------------------------------------
def gate_fidelity(U: np.ndarray, V: np.ndarray) -> float:
    """F = |Tr(U^dag V)|^2 / d^2. Global-phase invariant; 1 iff equal."""
    d = U.shape[0]
    tr = np.trace(U.conj().T @ V)
    return float(np.abs(tr) ** 2 / d ** 2)

def ladder_ent_cost(k: int, platform: str) -> int:
    """Entangling gates of a weight-k Pauli rotation ladder."""
    if k < 2:
        return 0
    if platform == "superconducting":
        return 2 * (k - 1)          # CNOT ladder, 2 per rung
    if platform == "ion-trapped":
        return 2 * k - 3            # inner rung -> 1 RZZ(theta);
                                    # outer CNOTs -> 1 RZZ(pi/2) each side
    raise ValueError(platform)

CNOT_1Q_OVERHEAD_ION = 3  # 1q gates per CNOT compiled to RZZ (approx.)

# -- GATE COUNTING -----------------------------------------------------------
def gates_per_pauli_rotation(pauli_string_full: str, platform: str):
    """Native-gate cost of exp(-i theta P) for a weight-k Pauli string.

        single_q   = CENTRAL_RZ[platform]  +  2*nX  +  4*nY
                       central RZ : 1 on superconducting (explicit RZ in the
                                    CNOT-RZ-CNOT ladder), 0 on trapped-ion(the
                                    native RZZ absorbs it, since
                                    CNOT-RZ-CNOT == RZZ).
                       2 per X    : H before + H after  (basis change, both qubits)
                       4 per Y    : (S^dag H) before + (H S) after, counted as 4
    """
    k = sum(1 for c in pauli_string_full if c != "I")
    if k == 0:
        return 0, 0
    ent = ladder_ent_cost(k, platform)
    nX = pauli_string_full.count("X")
    nY = pauli_string_full.count("Y")
    single_q = CENTRAL_RZ[platform] + 2 * nX + 4 * nY
    if platform == "ion-trapped" and k >= 3:
        single_q += CNOT_1Q_OVERHEAD_ION * 2 * (k - 2)
    return ent, single_q


def gate_count(order, n, term_data, platform):
    per_step_ent = per_step_1q = 0
    for idx, (_c, p, _w) in enumerate(term_data):
        e, s = gates_per_pauli_rotation(p, platform)
        # 2nd-order Suzuki doubles every term except the central one
        mult = 1 if (order == 1 or idx == len(term_data) - 1) else 2
        per_step_ent += mult * e
        per_step_1q += mult * s
    total = n * (per_step_ent + per_step_1q)
    return total, n * per_step_ent, n * per_step_1q


def entangling_depth(order, n, term_data, platform):
    wire_depth = {}
    for _step in range(n):
        seq = term_data if order == 1 else list(term_data) + list(term_data[-2::-1])
        for _c, p, w in seq:
            k = len(w)
            if k < 2:
                continue
            add = ladder_ent_cost(k, platform)
            d = max((wire_depth.get(wi, 0) for wi in w), default=0) + add
            for wi in w:
                wire_depth[wi] = d
    return max(wire_depth.values(), default=0)


# -- MAIN SWEEP --------------------------------------------------------------
def main():
    size = CFG.system_size
    t = CFG.time_to_evolve

    H = build_target_hamiltonian()
    term_data = extract_terms(H, size)
    U_exact = exact_unitary()

    print(f"Target: {CFG.target_hamiltonian}  |  size={size}  |  t={t}")
    print(f"H has {len(term_data)} Pauli terms:")
    for c, p, w in term_data:
        print(f"    {c:+.3f} * {p}  on wires {w}")

    # results[platform][order] -> dict of curves
    results = {
        plat: {o: {"n": [], "fid": [], "gates": [], "ent": [], "depth": []}
               for o in ORDERS}
        for plat in PLATFORMS
    }

    # fidelity + Trotter unitary depend only on (order, n), NOT on platform,
    # so compute them once and reuse across platforms.
    for order in ORDERS:
        for n in N_STEPS_SWEEP:
            Ut = trotter_unitary(order, n, H, size, t)
            fid = gate_fidelity(U_exact, Ut)
            for plat in PLATFORMS:
                tot, ent, _one_q = gate_count(order, n, term_data, plat)
                dep = entangling_depth(order, n, term_data, plat)
                r = results[plat][order]
                r["n"].append(n); r["fid"].append(fid)
                r["gates"].append(tot); r["ent"].append(ent)
                r["depth"].append(dep)

    # print one table per platform
    for plat in PLATFORMS:
        ent_name = ENT_NAME[plat]
        print("=" * 78)
        print(f"Platform: {plat}  (entangling gate = {ent_name})")
        print("-" * 78)
        print(f"{'order':>5} {'n':>4} {'fidelity':>12} {'gates':>8} "
              f"{ent_name:>6} {'depth':>6}")
        print("-" * 78)
        for order in ORDERS:
            r = results[plat][order]
            for i in range(len(r["n"])):
                print(f"{order:>5} {r['n'][i]:>4} {r['fid'][i]:>12.8f} "
                      f"{r['gates'][i]:>8} {r['ent'][i]:>6} {r['depth'][i]:>6}")
            print("-" * 78)

    if SAVE_PLOT:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, len(PLATFORMS),
                                 figsize=(7 * len(PLATFORMS), 5),
                                 squeeze=False)
        colors = {1: "#d1495b", 2: "#1f77b4"}
        pastel = {1: "#eba8b2", 2: "#a8cbe8"}

        for col, plat in enumerate(PLATFORMS):
            ent_name = ENT_NAME[plat]
            ax = axes[0][col]
            for order in ORDERS:
                r = results[plat][order]
                ax.plot(r["gates"], r["fid"], "o-", color=colors[order],
                        label=f"order {order} — total gates")
                ax.plot(r["ent"], r["fid"], "s--", color=pastel[order],
                        label=f"order {order} — 2q gates ({ent_name})")
            ax.set_xlabel("Gates")
            ax.set_ylabel("Fidelity")
            ax.set_title(f"{plat} — fidelity vs gate count")
            ax.grid(True, alpha=0.3)
            ax.legend()

        fig.suptitle(f"{CFG.target_hamiltonian}, t={t}", y=1.05)
        fig.tight_layout()
        fig.savefig(PLOT_PATH, dpi=150, bbox_inches="tight")
        print(f"Saved plot -> {PLOT_PATH}")


if __name__ == "__main__":
    main()