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
"""Training module for the Quantum GAN using PyTorch."""

import torch
import numpy as np
from config import CFG
from qgan.ancilla import get_final_gen_state_for_discriminator, get_max_entangled_state_with_ancilla_if_needed
from qgan.cost_functions import compute_fidelity_and_cost, compute_cost
from qgan.discriminator import Discriminator
from qgan.generator import Generator
from qgan.target import get_final_target_state
from tools.data.data_managers import print_and_log, save_fidelity_loss
from tools.plot_hub import plt_fidelity_vs_iter
from tools.qobjects.qgates import device # Import the configured device
from time import perf_counter as tpc
from dataclasses import dataclass
from typing import Optional, List

# Set PyTorch seed for reproducibility
#torch.manual_seed(42)

@dataclass
class Results:
    fidelities: List
    losses: List
    runtimes: List
    total_time: float


class Training:
    """
    Manages the training process for the Quantum GAN using PyTorch.
    """
    def __init__(self, config=CFG):
        self.config = config
        # --- 1. Initialize models and data ---
        # The ancilla and target functions now return tensors on the correct device
        initial_state_total, initial_state_final = get_max_entangled_state_with_ancilla_if_needed(self.config)
        self.final_target_state: torch.Tensor = get_final_target_state(initial_state_final, self.config)
        
        # Instantiate the PyTorch models
        self.gen = Generator(config=self.config).to(device)
        self.dis = Discriminator(config=self.config).to(device)
        
        self.initial_state_total = initial_state_total.to(device)
        self.final_target_state = self.final_target_state.to(device)

        # --- 2. Initialize Optimizers ---
        # Use a standard PyTorch optimizer like Adam
        self.optimizer_gen = torch.optim.SGD(self.gen.parameters(), lr=self.config.l_rate, momentum=self.config.momentum_coeff)
        self.optimizer_dis = torch.optim.SGD(self.dis.parameters(), lr=self.config.l_rate, momentum=self.config.momentum_coeff)

    def run(self):
        """Runs the entire QGAN training loop."""
        print_and_log("\n" + self.config.show_data(), self.config.log_path)
        
        # Note: Model loading logic would need to be adapted to use `load_state_dict`
        # For simplicity in this migration, we are starting from scratch.
        # load_models_if_specified(self) 

        fidelities_history, losses_history, runtimes_history = [], [], []
        starttime = tpc()
        
        for epoch in range(1, self.config.epochs + 1):
            epoch_fidelities, epoch_losses, epoch_runtimes = [], [], []
            for i in range(self.config.iterations_epoch):
                # --- Train Discriminator ---
                # We need to detach the generator's output from the computation graph
                # when training the discriminator to avoid backpropagating through the generator.
                
                t0 = tpc()
                # --- Train Generator ---
                for _ in range(self.config.steps_gen):
                    self.optimizer_gen.zero_grad()
                    # Now, we use the generator's output directly to build the graph
                    total_gen_state = self.gen(self.initial_state_total)
                    final_gen_state = get_final_gen_state_for_discriminator(total_gen_state)
                    # The generator aims to minimize the cost function
                    gen_loss = compute_cost(self.dis, self.final_target_state, final_gen_state)
                    gen_loss.backward()
                    self.optimizer_gen.step()

                with torch.no_grad():
                    total_gen_state = self.gen(self.initial_state_total)
                final_gen_state_detached = get_final_gen_state_for_discriminator(total_gen_state)

                for _ in range(self.config.steps_dis):
                    self.optimizer_dis.zero_grad()
                    # The cost function for the discriminator aims to maximize the distance,
                    # so we multiply by -1 to perform gradient descent (minimization).
                    dis_loss = -1 * compute_cost(self.dis, self.final_target_state, final_gen_state_detached)
                    dis_loss.backward()
                    self.optimizer_dis.step()
                tf = tpc()
                # --- Logging and Monitoring ---
                if i % self.config.save_fid_and_loss_every_x_iter == 0:
                    # Use the detached state for fidelity/loss calculation to save memory
                    fid, loss = compute_fidelity_and_cost(self.dis, self.final_target_state, final_gen_state_detached)
                    epoch_fidelities.append(fid)
                    epoch_losses.append(loss.item()) # .item() gets the float value
                    epoch_runtimes.append(tf-t0)
                    if i % self.config.log_every_x_iter == 0:
                        info = f"\nEpoch: {epoch:4d} | Iter: {i+1:4d} | Fidelity: {fid:8f} | Loss: {loss.item():8f}"
                        print_and_log(info, self.config.log_path)
            
            # --- Store and Plot Epoch History ---
            fidelities_history.extend(epoch_fidelities)
            losses_history.extend(epoch_losses)
            runtimes_history.extend(epoch_runtimes)
            plt_fidelity_vs_iter(np.array(fidelities_history), np.array(losses_history), CFG, epoch)

            # --- Stopping Conditions ---
            if epoch_fidelities and epoch_fidelities[-1] > self.config.max_fidelity:
                print_and_log(f"\nFidelity threshold of {self.config.max_fidelity} reached. Stopping training.", self.config.log_path)
                break
        
        # --- End of Training ---
        print_and_log(f"\nTraining finished after {epoch} epochs.", self.config.log_path)
        endtime = tpc()
        print_and_log(f"\nRun took: {endtime - starttime} time.", self.config.log_path)
        
        # --- Save Final Results ---
        save_fidelity_loss(np.array(fidelities_history), np.array(losses_history), self.config.fid_loss_path)
        self.gen.save_model_params(self.config.model_gen_path)
        self.dis.save_model_params(self.config.model_dis_path)

        return Results(fidelities=np.array(fidelities_history), losses=np.array(losses_history), runtimes=np.array(runtimes_history), total_time=endtime-starttime)