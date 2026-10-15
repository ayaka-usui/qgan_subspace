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
"""Training module for the Quantum GAN — PyTorch and PennyLane version.

The training loop logic is unchanged from the original.
Changes:
    - We defined the update_gen in generator.py since the parameters of A and B,
    are detached from the parameters of the circuit.
    - We have now use_choi (bool) to specify if we are going to train using Choi or using Haar
    random states.
"""

from datetime import datetime

import numpy as np
import torch

from config import CFG
from qgan.precision import cdtype, rdtype
import os

from qgan.ancilla import (
    ancilla_block_diagnostics,
    as_gen_state,
    choi_blocks,
    compute_ancilla_entanglement_entropy,
    compute_negativities,
    get_final_gen_state_torch,
    get_max_entangled_state_with_ancilla_if_needed,
    get_random_batch,
    negativity_pairs,
    prepare_batch_targets,
    extract_gen_unitary,
    process_fidelity,
    deterministic_batch_size,
    haar_random_batch
)
from qgan.cost_functions import compute_fidelity_and_cost
from qgan.discriminator import Discriminator
from qgan.generator import (
    Generator,
)
from qgan.target import get_target_operator, get_target_unitary
from tools.data_managers import (
    print_and_log,
    save_entropy,
    save_fidelity_loss,
    save_gen_final_params,
    save_negativity,
)
from tools.loading_helpers import load_models_if_specified
from tools.plot_hub import plt_fidelity_vs_iter

np.random.seed()


class Training:
    def __init__(self):
        """Builds the training configuration.

        Choi mode:
            1. Maximally entangled state |\Phi^+> (+ ancilla if needed).
            2. Target state: (I \otimes U_target [\otimes I_ancilla]) |\Phi^+>.

        Batch modes (use_choi=False):
            - "haar": fresh Haar-random batch of size CFG.batch_size each iter.
            - "comp": full deterministic all-pairs set of size d(d+1)/2 each iter.
            Target states: U_target |\psi_i> (with ancilla if needed).
 
        Evaluation (use_choi=False):
            Always Haar-random states, seeded with CFG.eval_seed, of size
            CFG.eval_batch_size. Reused across runs as the generalisation probe.
        """
        # -- Target operator --------
        self.target_op = torch.tensor(get_target_operator(), dtype=cdtype())
        # system-only target unitary V (d x d), for the ancilla block diagnostics
        self.target_unitary = torch.tensor(
            np.asarray(get_target_unitary(CFG.target_hamiltonian, CFG.system_size)), dtype=cdtype()
        )
        self.anc_diag: dict[str, list] = {}

        if CFG.use_choi:
            # -- Choi mode: apply target_op to |\Phi^+> --
            _, initial_state_final = get_max_entangled_state_with_ancilla_if_needed(CFG.system_size)
            initial_state = torch.tensor(np.asarray(initial_state_final), dtype=cdtype()).reshape(-1)

            self.final_target_state = (self.target_op @ initial_state).reshape(-1)

        if not CFG.use_choi and CFG.batch_mode == "comp":
            dim = 2**CFG.system_size
            M = deterministic_batch_size(dim)
            if CFG.batch_size > M:
                print_and_log(
                    f"batch_size {CFG.batch_size} -> {M} (capped at d(d+1)/2 for d={dim}).",
                    CFG.log_path,
                )
                CFG.batch_size = M

        # Generator: variational quantum circuit
        self.gen: Generator = Generator()

        # Discriminator: classical Hermitian operator
        self.dis: Discriminator = Discriminator()

        # -- Eval batch (batching only) ----------------------
        # Fixed evaluation set: generated once with a fixed seed so that
        # the fidelity curve reflects generalisation, not the luck of the
        # current training batch. Same set is reused across iters and runs.
        if not CFG.use_choi:
            dim = 2**CFG.system_size
            self.eval_batch_size = CFG.eval_batch_size
            eval_seed = getattr(CFG, "eval_seed", 1234)

            # Snapshot + restore RNG state so eval-set generation does not
            # perturb the training RNG stream.
            torch_state = torch.random.get_rng_state()
            np_state = np.random.get_state()
            torch.manual_seed(eval_seed)
            np.random.seed(eval_seed)
            try:
                self.eval_raw, self.eval_inputs = haar_random_batch(
                    dim, self.eval_batch_size
                )
                self.eval_targets = prepare_batch_targets(
                    self.eval_raw, self.eval_inputs, self.target_op
                )
            finally:
                torch.random.set_rng_state(torch_state)
                np.random.set_state(np_state)

    def run(self):
        """Run the training loop.

        For each iteration:
            1. Process ancilla on generator output (torch).
            2. Update discriminator (maximise Wasserstein distance).
            3. Update generator (minimise Wasserstein distance).
            4. Log fidelity and loss periodically.

        Choi mode:
            Each iteration uses the fixed |\Phi^+> state.

        Batch mode (haar or comp):
            Each iteration draws B states via `get_random_batch`:
              - "haar": fresh Haar-random sample of size CFG.batch_size.
              - "comp": full deterministic all-pairs set of size d(d+1)/2
                        (CFG.batch_size was overridden to this value in __init__).
            Loss and gradients are averaged over the batch (mini-batch SGD).

        Stops when max_fidelity is reached or max epochs is reached.
        Saves models, fidelity history, and generator parameters at the end.
        """
        # -- Initialise training -----------------------------------
        print_and_log("\n" + CFG.show_data(), CFG.log_path)

        # Load models if a previous checkpoint is specified
        load_models_if_specified(self)

        fidelities_history, losses_history, entropy_history = [], [], []
        neg_labels = [label for label, _, _ in negativity_pairs(CFG.system_size, CFG.extra_ancilla)]
        neg_history = {label: [] for label in neg_labels}
        track_entropy = CFG.compute_entanglement and CFG.extra_ancilla and CFG.ancilla_mode == "pass"

        # Plateau histories (when continuing from a saved run) to stitch the live plots, as upstream.
        plat_fids, plat_losses, plat_ents, plat_negs = self._load_plateau_histories(neg_labels)

        starttime = datetime.now()
        num_epochs: int = 0

        # -- Main training loop -----------------------------------
        while True:
            fidelities = []
            losses = []
            entropies = []
            neg_dict = {label: [] for label in neg_labels}
            num_epochs += 1

            for epoch_iter in range(CFG.iterations_epoch):
                if CFG.use_choi:
                    self._train_step_choi(epoch_iter, fidelities, losses)
                else:
                    self._train_step_batch(epoch_iter, fidelities, losses)

                # -- Evaluation on fixed set --
                if epoch_iter % CFG.save_fid_and_loss_every_x_iter == 0:
                    if CFG.use_choi:
                        self._evaluate_choi(fidelities, losses)
                    else:
                        self._evaluate_batch(fidelities, losses)
                    if CFG.compute_entanglement:
                        if track_entropy:
                            entropies.append(compute_ancilla_entanglement_entropy(self._choi_output_state()))
                        compute_negativities(self.gen, neg_dict)

                # -- Log
                if epoch_iter % CFG.log_every_x_iter == 0 and fidelities:
                    info = (
                        f"\nepoch:{num_epochs:4d} | "
                        f"iters:{epoch_iter + 1:4d} | "
                        f"fidelity:{round(fidelities[-1], 6):8f} | "
                        f"loss:{round(losses[-1], 6):8f}"
                    )
                    if entropies:
                        info += f" | entropy:{round(entropies[-1], 6):8f}"
                    if self.anc_diag:
                        info += f" | p0:{self.anc_diag['p0'][-1]:.4f} | ancilla purity:{self.anc_diag['purity_anc'][-1]:.4f}"
                    for label in neg_labels[:2]:
                        if neg_dict.get(label):
                            info += f" | neg({label}):{round(neg_dict[label][-1], 6):8f}"
                    print_and_log(info, CFG.log_path)

            # -- End of epoch: store history and plot -----------------------------------
            fidelities_history = np.append(fidelities_history, fidelities)
            losses_history = np.append(losses_history, losses)
            if track_entropy:
                entropy_history = np.append(entropy_history, entropies)
            for label in neg_labels:
                neg_history[label] = np.append(neg_history[label], neg_dict[label])

            plt_fidelity_vs_iter(
                plat_fids + list(fidelities_history),
                plat_losses + list(losses_history),
                CFG,
                num_epochs,
                ([np.nan] * max(0, len(plat_fids) - len(plat_ents)) + plat_ents + list(entropy_history))
                if track_entropy
                else None,
                {k: plat_negs.get(k, []) + list(v) for k, v in neg_history.items()} if CFG.compute_entanglement else None,
                len(plat_fids),
            )

            # -- Stopping conditions -----------------------------------
            if num_epochs >= CFG.epochs:
                print_and_log(
                    "\n==================================================\n",
                    CFG.log_path,
                )
                print_and_log(
                    f"\nThe number of epochs exceeds {CFG.epochs}.",
                    CFG.log_path,
                )
                break

            if fidelities[-1] > CFG.max_fidelity:
                print_and_log(
                    "\n==================================================\n",
                    CFG.log_path,
                )
                print_and_log(
                    f"\nThe fidelity {fidelities[-1]} exceeds the maximum {CFG.max_fidelity}.",
                    CFG.log_path,
                )
                break

        # -- End of training: save everything -----------------------------------
        save_fidelity_loss(fidelities_history, losses_history, CFG.fid_loss_path)
        if track_entropy:
            save_entropy(entropy_history, CFG.entropy_path)
        if CFG.compute_entanglement:
            for label, history in neg_history.items():
                save_negativity(history, CFG.negativity_path(label))
        self.gen.save_model(CFG.model_gen_path)
        self.dis.save_model(CFG.model_dis_path)
        save_gen_final_params(self.gen, CFG.gen_final_params_path)
        # Time calculation
        endtime = datetime.now()
        print_and_log(f"\nRun took: {endtime - starttime} time.", CFG.log_path)

        # -- Save ancilla block diagnostics (one entry per evaluation) --
        if self.anc_diag:
            np.savez(f"{CFG.base_data_path}/ancilla_diagnostics.npz", **{k: np.asarray(v) for k, v in self.anc_diag.items()})

        # -- Save gradient history --
        if self.gen.grad_history:
            grad_array = np.stack(self.gen.grad_history, axis=0)
            np.save(f"{CFG.base_data_path}/grad_history.npy", grad_array)
            if self.gen.ancilla:
                anc_idx = np.array(self.gen._get_ancilla_param_indices(),
                                   dtype=int)
                np.save(f"{CFG.base_data_path}/ancilla_indices.npy", anc_idx)
            print_and_log(
                f"\nSaved gradient history: shape {grad_array.shape}",
                CFG.log_path,
            )
        # -- Save discriminator gradient history --
        if self.dis.grad_history:
            dis_grad_array = np.stack(self.dis.grad_history, axis=0)
            np.save(f"{CFG.base_data_path}/dis_grad_history.npy", dis_grad_array)
            print_and_log(
                f"\nSaved discriminator gradient history: shape {dis_grad_array.shape}",
                CFG.log_path,
            )

    # -- Entanglement helpers --------------------------------------------
    def _choi_output_state(self) -> np.ndarray:
        """(I ⊗ U_gen)|Φ+>|0>: the cached generator state in Choi mode, rebuilt from U_gen in batch mode."""
        if CFG.use_choi:
            return self.gen.total_gen_state.detach().cpu().numpy()
        me_state, _ = get_max_entangled_state_with_ancilla_if_needed(CFG.system_size)
        U = self.gen.get_gen_unitary()
        return np.kron(np.eye(2**CFG.system_size), U) @ np.asarray(me_state).flatten()

    @staticmethod
    def _load_plateau_histories(neg_labels: list[str]):
        """Histories of the run being continued (CFG.load_timestamp), for stitched live plots.

        log_fidelity_loss.txt is ONE column [fidelities..., losses...] (two np.savetxt calls),
        so it is split in halves (the upstream version indexed rows and always fell into its except).
        """
        fids, losses, ents, negs = [], [], [], {}
        if not CFG.load_timestamp:
            return fids, losses, ents, negs
        base = os.path.join("generated_data", CFG.load_timestamp, "fidelities")
        try:
            path = os.path.join(base, "log_fidelity_loss.txt")
            if os.path.exists(path):
                data = np.atleast_1d(np.loadtxt(path))
                half = len(data) // 2
                fids, losses = list(data[:half]), list(data[half:])
            path = os.path.join(base, "log_entropy.txt")
            if os.path.exists(path):
                ents = list(np.atleast_1d(np.loadtxt(path)))
            for label in neg_labels:
                path = os.path.join(base, f"log_negativity_{label.replace('-', '')}.txt")
                if os.path.exists(path):
                    negs[label] = list(np.atleast_1d(np.loadtxt(path)))
        except (OSError, ValueError) as e:
            print_and_log(f"Warning: could not load plateau data for plotting: {e}\n", CFG.log_path)
        return fids, losses, ents, negs

    @staticmethod
    def _fidelity_key() -> str:
        """Which ancilla-block fidelity is the training fidelity of the current ancilla mode."""
        if CFG.ancilla_mode == "trace":
            return "F_trace"
        if CFG.ancilla_mode == "project" and CFG.ancilla_project_norm == "re-norm":
            return "F_post"
        return "F_joint"

    def _record_anc_diag(self, K00: torch.Tensor, K10: torch.Tensor) -> dict:
        """Append the ancilla block diagnostics (p0, F_joint, F_post, F_trace, purity, K00 singular-value ratio)."""
        diag = ancilla_block_diagnostics(K00, K10, self.target_unitary.to(K00.dtype))
        for k, v in diag.items():
            self.anc_diag.setdefault(k, []).append(v)
        return diag

    def _freeze_ancilla_row(self):
        """CFG.dis_ancilla_row_trainable = False (pass mode, ancilla present): zero the gradient of the critic's
        ancilla row (the last one) before the step, and return its value to restore after the step (so a momentum
        buffer restored from a checkpoint cannot move it either). Returns None when nothing is frozen."""
        if CFG.dis_ancilla_row_trainable or not (self.dis.ancilla and self.dis.ancilla_mode == "pass"):
            return None
        for p in (self.dis.alpha, self.dis.beta):
            if p.grad is not None:
                p.grad[-1] = 0.0
        return self.dis.alpha.detach()[-1].clone(), self.dis.beta.detach()[-1].clone()

    def _restore_ancilla_row(self, frozen) -> None:
        if frozen is not None:
            with torch.no_grad():
                self.dis.alpha[-1] = frozen[0]
                self.dis.beta[-1] = frozen[1]

    def _record_dis_grad(self) -> None:
        """Record flat gradient vector over all discriminator parameters."""
        pieces = []
        for p in self.dis.parameters():
            if p.grad is not None:
                pieces.append(p.grad.detach().cpu().numpy().ravel().copy())
        if pieces:
            self.dis.grad_history.append(np.concatenate(pieces))

    # -- Choi mode (max entangled state) ------------------
    def _train_step_choi(self, epoch_iter: int, fidelities: list, losses: list):
        """One training iteration in Choi mode."""
        # -- Detached state for discriminator
        with torch.no_grad():
            total_gen_detached = self.gen.get_total_gen_state()
            final_gen_detached = as_gen_state(get_final_gen_state_torch(total_gen_detached))

        # -- Discriminator steps
        for _ in range(CFG.steps_dis):
            self.dis.optimizer.zero_grad()
            if CFG.dis_gradient == "upstream":
                ga, gb = self.dis.upstream_gradients(self.final_target_state, final_gen_detached)
                self.dis.alpha.grad, self.dis.beta.grad = -ga, -gb  # optimiser minimises -W
            else:
                dis_loss = self.dis.compute_loss(self.final_target_state, final_gen_detached)
                dis_loss.backward()
            frozen = self._freeze_ancilla_row()
            self._record_dis_grad()
            self.dis.optimizer.step()
            self._restore_ancilla_row(frozen)

        # -- Generator steps
        for _ in range(CFG.steps_gen):
            self.gen.update_gen(self.dis, self.final_target_state)

    # -- Batching (random states) ------------------
    def _train_step_batch(self, epoch_iter: int, fidelities: list, losses: list):
        """One training iteration in batch mode (Haar or deterministic).
 
        1. Draw B input states via `get_random_batch`:
            - "haar": fresh Haar-random sample of size CFG.batch_size.
            - "comp": the FULL deterministic all-pairs set of size
                      M = d(d+1)/2 (CFG.batch_size is overridden to M in
                      __init__, and `get_random_batch` returns the cached
                      set on every call).
        2. Compute target states: U_target|ψ_i⟩ for each.
        3. Discriminator (maximise) and Generator (minimise).
        4. Evaluate average fidelity over batch periodically.
        """
        B = CFG.batch_size
        dim = 2**CFG.system_size

        # -- Generate Haar batch for each iter (includes ancilla |0> if needed) --
        batch_raw, batch_inputs = get_random_batch(dim, B)
        batch_targets = prepare_batch_targets(batch_raw, batch_inputs, self.target_op)

        # -- Discriminator: detached generator states, averaged loss --
        with torch.no_grad():
            det_gen_states = [
                as_gen_state(get_final_gen_state_torch(self.gen.get_total_gen_state(inp))) for inp in batch_inputs
            ]

        for _ in range(CFG.steps_dis):
            self.dis.optimizer.zero_grad()
            if CFG.dis_gradient == "upstream":
                grads = [self.dis.upstream_gradients(t, g) for t, g in zip(batch_targets, det_gen_states)]
                self.dis.alpha.grad = -sum(ga for ga, _ in grads) / B
                self.dis.beta.grad = -sum(gb for _, gb in grads) / B
            else:
                dis_loss = torch.tensor(0.0, dtype=rdtype())
                for t, g in zip(batch_targets, det_gen_states):
                    dis_loss = dis_loss + self.dis.compute_loss(t, g)
                dis_loss = dis_loss / B
                dis_loss.backward()
            frozen = self._freeze_ancilla_row()
            self._record_dis_grad()
            self.dis.optimizer.step()
            self._restore_ancilla_row(frozen)

        # -- Generator: averaged loss with gradients --
        for _ in range(CFG.steps_gen):
            self.gen.update_gen(
                self.dis,
                batch_inputs=batch_inputs,
                batch_targets=batch_targets,
            )

    # -- Evaluation (separated from training) ------------------
    def _evaluate_choi(self, fidelities: list, losses: list) -> None:
        """Evaluate fidelity and loss on the fixed |\Phi^+> state."""
        with torch.no_grad():
            final_gen_eval = as_gen_state(get_final_gen_state_torch(self.gen.total_gen_state))
        fid, loss = compute_fidelity_and_cost(
            self.dis, self.final_target_state, final_gen_eval
        )
        fidelities.append(fid)
        losses.append(loss)
        if CFG.extra_ancilla:
            K00, K10 = choi_blocks(self.gen.total_gen_state.detach(), CFG.system_size)
            self._record_anc_diag(K00, K10)

    def _evaluate_batch(self, fidelities: list, losses: list) -> None:
        """Evaluate unitary process fidelity + loss on the fixed eval batch.

        Process fidelity is computed from U_gen (via qml.matrix) vs U_target,
        matching the Choi-mode fidelity exactly:
            F = |Tr(U_t^dagger U_g)|^2 / d^2
        The loss still uses the eval batch + discriminator since it depends on
        the discriminator state.
        """
        with torch.no_grad():
            U_gen = extract_gen_unitary(self.gen)
            if CFG.extra_ancilla:
                # Ancilla enters in |0>: only the blocks K_j0 = <j|U_gen|0> act. The fidelity matches the
                # Choi-mode fidelity of each ancilla mode: pass / project "pass" -> F_joint (ancilla back in |0>),
                # project "re-norm" -> F_post (post-selected), trace -> F_trace (channel, ancilla discarded).
                d = 2**CFG.system_size
                K00 = U_gen.reshape(d, 2, d, 2)[:, 0, :, 0]
                K10 = U_gen.reshape(d, 2, d, 2)[:, 1, :, 0]
                diag = self._record_anc_diag(K00, K10)
                fid = diag[self._fidelity_key()]
            else:
                fid = process_fidelity(U_gen, self.target_op)

            eval_gen = [
                as_gen_state(get_final_gen_state_torch(self.gen.get_total_gen_state(inp)))
                for inp in self.eval_inputs
            ]
        loss_sum = 0.0
        for t, g in zip(self.eval_targets, eval_gen):
            _, l = compute_fidelity_and_cost(self.dis, t, g)
            loss_sum += l
        fidelities.append(fid)
        losses.append(loss_sum / len(self.eval_inputs))
