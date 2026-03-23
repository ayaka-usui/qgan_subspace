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
What changes:
    - Generator and Discriminator are now PyTorch-PennyLane-based (see generator.py,
      discriminator.py).
    - States are torch tensors instead of numpy matrices.
    - Ancilla processing uses the torch functions from generator_torch.py.
    - cost_functions reduced to fidelity + cost evaluation (no braket).
    - save_model uses torch.save for the discriminator, pickle for the generator
      (which handles its own __getstate__/__setstate__).

The training loop logic is unchanged from the original:
    1. Process ancilla on generator output.
    2. Update discriminator (maximise Wasserstein distance).
    3. Update generator (minimise Wasserstein distance).
    4. Log fidelity and loss periodically.
"""

from datetime import datetime

import numpy as np
import torch

from config import CFG
from qgan.generator import (
    Generator,
)
from qgan.ancilla import (
    get_final_gen_state_torch,
    get_max_entangled_state_torch,
)
from qgan.discriminator import Discriminator
from qgan.cost_functions import compute_fidelity_and_cost
from qgan.target import get_final_target_state
from tools.data_managers import (
    print_and_log,
    save_fidelity_loss,
    save_gen_final_params,
)
from tools.loading_helpers import load_models_if_specified
from tools.plot_hub import plt_fidelity_vs_iter

np.random.seed()


class Training:
    def __init__(self):
        """Builds the training configuration.

        Prepares:
            1. Maximally entangled state (Choi) with ancilla if needed (torch tensors).
            2. Target state from the target Hamiltonian (numpy -> torch).
            3. Generator (PennyLane + PyTorch).
            4. Discriminator (PyTorch nn.Module).
        """
        # Prepare maximally entangled state (+ ancilla if needed)
        # Both are torch tensors, shape (d, 1), no gradient needed
        _, initial_state_final = get_max_entangled_state_torch(CFG.system_size)

        # Target state: (I ⊗ U_target) |\Phi^+ >
        # get_final_target_state returns numpy, convert to torch
        target_np = get_final_target_state(initial_state_final.numpy())
        self.final_target_state: torch.Tensor = torch.tensor(
            np.asarray(target_np), dtype=torch.complex128
        ).reshape(-1)

        # Generator: variational quantum circuit (PennyLane + PyTorch)
        self.gen: Generator = Generator()

        # Discriminator: classical Hermitian operator (PyTorch nn.Module)
        self.dis: Discriminator = Discriminator()

    def run(self):
        """Run the training loop.

        For each iteration:
            1. Process ancilla on generator output (torch).
            2. Update discriminator (maximise Wasserstein distance).
            3. Update generator (minimise Wasserstein distance).
            4. Log fidelity and loss periodically.

        Stops when max_fidelity is reached or max epochs is reached.
        Saves models, fidelity history, and generator parameters at the end.
        """
        # -- Initialise training -----------------------------------
        print_and_log("\n" + CFG.show_data(), CFG.log_path)

        # Load models if a previous checkpoint is specified
        load_models_if_specified(self)

        fidelities_history, losses_history = [], []
        starttime = datetime.now()
        num_epochs: int = 0

        # -- Main training loop -----------------------------------
        while True:
            fidelities = []
            losses = []
            num_epochs += 1

            for epoch_iter in range(CFG.iterations_epoch):
                # --- Process ancilla on current generator state
                total_gen_state = self.gen.get_total_gen_state()
                final_gen_state = get_final_gen_state_torch(total_gen_state)

                # Flatten to 1D for the loss functions
                final_gen_1d = final_gen_state.reshape(-1).detach()
                final_target_1d = self.final_target_state.reshape(-1)

                # -- Discriminator steps: MAX Wasserstein distance
                for _ in range(CFG.steps_dis):
                    self.dis.optimizer.zero_grad()
                    dis_loss = self.dis.compute_loss(final_target_1d, final_gen_1d)
                    dis_loss.backward()
                    self.dis.optimizer.step()

                # -- Generator step(s): MIN Wasserstein distance
                for _ in range(CFG.steps_gen):
                    self.gen.update_gen(self.dis, self.final_target_state)

                # --- Periodically compute and save fidelity & loss
                if epoch_iter % CFG.save_fid_and_loss_every_x_iter == 0:
                    with torch.no_grad():
                        total_gen_eval = self.gen.get_total_gen_state()
                        final_gen_eval = get_final_gen_state_torch(total_gen_eval)
                    fid, loss = compute_fidelity_and_cost(
                        self.dis, final_target_1d, final_gen_eval.reshape(-1)
                    )
                    fidelities.append(fid)
                    losses.append(loss)

                # -- Periodically log -----------------------------------
                if epoch_iter % CFG.log_every_x_iter == 0:
                    info = (
                        f"\nepoch:{num_epochs:4d} | "
                        f"iters:{epoch_iter + 1:4d} | "
                        f"fidelity:{round(fid, 6):8f} | "
                        f"loss:{round(loss, 6):8f}"
                    )
                    print_and_log(info, CFG.log_path)

            # -- End of epoch: store history and plot -----------------------------------
            fidelities_history = np.append(fidelities_history, fidelities)
            losses_history = np.append(losses_history, losses)
            plt_fidelity_vs_iter(fidelities_history, losses_history, CFG, num_epochs)

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
        # Fidelity and loss history
        save_fidelity_loss(fidelities_history, losses_history, CFG.fid_loss_path)

        # Generator model (torch dict format)
        self.gen.save_model(CFG.model_gen_path)

        # Discriminator model (torch dict format)
        self.dis.save_model(CFG.model_dis_path)

        # Generator final parameters (plain text)
        save_gen_final_params(self.gen, CFG.gen_final_params_path)

        endtime = datetime.now()
        print_and_log(f"\nRun took: {endtime - starttime} time.", CFG.log_path)