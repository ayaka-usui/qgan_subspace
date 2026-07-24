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
"""Training module for the Quantum GAN"""
from qgan.ancilla import compute_negativities

from datetime import datetime

import numpy as np

from config import CFG
from qgan.ancilla import (
    compute_ancilla_entanglement_entropy,
    compute_bipartite_negativity,
    get_max_entangled_state_with_ancilla_if_needed,
)
from qgan.cost_functions import compute_fidelity_and_cost
from qgan.discriminator import Discriminator
from qgan.generator import Generator
from qgan.target import get_final_target_state
from tools.data.data_managers import (
    print_and_log,
    save_entropy,
    save_fidelity_loss,
    save_negativity,
    save_gen_final_params,
    save_model,
)
from tools.data.loading_helpers import load_models_if_specified
from tools.plot_hub import plt_fidelity_vs_iter

np.random.seed()


class Training:
    def __init__(self):
        """Builds the configuration for the Training. You might wanna comment/discomment lines, for changing the model."""

        initial_state = get_max_entangled_state_with_ancilla_if_needed(CFG.system_size)
        """Preparation of max. entgl. state with ancilla qubit if needed, to generate state."""

        self.final_target_state: np.matrix = get_final_target_state(initial_state)
        """Prepare the target state to compare in the Dis, with the size and Target unitary defined in config."""

        self.gen: Generator = Generator(initial_state)
        """Prepares the Generator with the size, ansatz, layers and ancilla, defined in config."""

        self.dis: Discriminator = Discriminator()
        """Prepares the Discriminatos, with the size, and ancilla defined in config."""

    def run(self):
        """Run the training, saving the data, the model, the logs, and the results plots."""

        ###########################################################
        # Initialize training
        ###########################################################
        print_and_log("\n" + CFG.show_data(), CFG.log_path)

        # Load models if specified (only the params, and only if compatible)
        load_models_if_specified(self)

        fidelities_history, losses_history, entropy_history = [], [], []
        neg_history = {pair: [] for pair in ["1-2", "1-3", "1-a", "2-3", "2-a", "3-a"]}
        
        # Pre-load plateau histories for plotting if warm starting
        plat_fids, plat_losses, plat_ents = [], [], []
        plat_negs = {pair: [] for pair in neg_history}
        if CFG.load_timestamp:
            import os
            try:
                p_base = os.path.join("generated_data", CFG.load_timestamp, "fidelities")
                if os.path.exists(os.path.join(p_base, "log_fidelity_loss.txt")):
                    data = np.loadtxt(os.path.join(p_base, "log_fidelity_loss.txt"))
                    plat_fids, plat_losses = list(data[0]), list(data[1])
                if os.path.exists(os.path.join(p_base, "log_entropy.txt")):
                    plat_ents = list(np.loadtxt(os.path.join(p_base, "log_entropy.txt")))
                for pair in plat_negs:
                    path = os.path.join(p_base, f"log_negativity_{pair.replace('-', '')}.txt")
                    if os.path.exists(path):
                        plat_negs[pair] = list(np.loadtxt(path))
            except Exception as e:
                print(f"Warning: could not load plateau data for plotting: {e}")

        starttime = datetime.now()
        num_epochs: int = 0

        ###########################################################
        # Main Training Block
        ###########################################################
        while True:
            # while (f < 0.95):
            fidelities, losses, entropies = [], [], []
            neg_dict = {pair: [] for pair in neg_history}
            num_epochs += 1
            for epoch_iter in range(CFG.iterations_epoch):
                ###########################################################
                # Dis and Gen gradient descent
                ###########################################################

                for _ in range(CFG.steps_dis):
                    self.dis.update_dis(self.final_target_state, self.gen.total_gen_state)

                for _ in range(CFG.steps_gen):
                    self.gen.update_gen(self.dis, self.final_target_state)

                ###########################################################
                # Every X iterations: compute and save fidelity, loss, and entropy
                ###########################################################
                if epoch_iter % CFG.compute_and_save_fid_every_x_iter == 0:
                    fid, loss = compute_fidelity_and_cost(self.dis, self.final_target_state, self.gen.total_gen_state)
                    fidelities.append(fid), losses.append(loss)
                    if CFG.compute_entanglement:
                        if CFG.extra_ancilla:
                            entropy = compute_ancilla_entanglement_entropy(self.gen.total_gen_state)
                            entropies.append(entropy)
                        neg_dict = compute_negativities(self.gen, neg_dict)

                ############################################################
                # Every X iterations: Print and log fidelity, loss, and entropy
                ############################################################
                if epoch_iter % CFG.log_every_x_iter == 0:
                    info = "\nepoch:{:4d} | iters:{:4d} | fidelity:{:8f} | loss:{:8f}".format(
                        num_epochs, epoch_iter + 1, round(fid, 6), round(loss, 6)
                    )
                    if CFG.compute_entanglement and CFG.extra_ancilla:
                        info += " | entropy:{:8f}".format(round(entropies[-1], 6))
                    
                    if len(neg_dict["1-2"]) > 0:
                        info += " | neg(1,2):{:8f}".format(round(neg_dict["1-2"][-1], 6))
                        info += " | neg(1,3):{:8f}".format(round(neg_dict["1-3"][-1], 6))
                        
                    print_and_log(info, CFG.log_path)

            ###########################################################
            # End of epoch, store data and plot
            ###########################################################
            fidelities_history = np.append(fidelities_history, fidelities)
            losses_history = np.append(losses_history, losses)
            if CFG.compute_entanglement and CFG.extra_ancilla:
                entropy_history = np.append(entropy_history, entropies)
            else:
                entropy_history = None
                
            for pair in neg_history:
                neg_history[pair] = np.append(neg_history[pair], neg_dict[pair])

            # Stitch with plateau data for plotting
            plot_fids = plat_fids + list(fidelities_history)
            plot_losses = plat_losses + list(losses_history)
            plot_ents = plat_ents + list(entropy_history) if entropy_history is not None else None
            plot_negs = {p: plat_negs[p] + list(neg_history[p]) for p in neg_history}

            plt_fidelity_vs_iter(
                plot_fids,
                plot_losses,
                CFG,
                num_epochs,
                plot_ents,
                plot_negs,
                len(plat_fids) # passed to draw vertical line
            )

            #############################################################
            # Stopping conditions
            #############################################################
            if num_epochs >= CFG.epochs:
                print_and_log("\n==================================================\n", CFG.log_path)
                print_and_log(f"\nThe number of epochs exceeds {CFG.epochs}.", CFG.log_path)
                break

            if fidelities[-1] > CFG.max_fidelity:  # TODO: Maybe change this cond, to use max(fidelities)?
                print_and_log("\n==================================================\n", CFG.log_path)
                print_and_log(
                    f"\nThe fidelity {fidelities[-1]} exceeds the maximum {CFG.max_fidelity}.",
                    CFG.log_path,
                )
                break

        ###########################################################
        # End training, save all data into files
        ###########################################################
        # Save data of fidelity and loss
        save_fidelity_loss(fidelities_history, losses_history, CFG.fid_loss_path)
        # Save entropy if available
        if CFG.compute_entanglement and CFG.extra_ancilla and entropy_history is not None:
            save_entropy(entropy_history, CFG.entropy_path)
            
        for pair, path in CFG.negativity_paths.items():
            if len(neg_history[pair]) > 0:
                save_negativity(neg_history[pair], path)

        # Save data of the generator and the discriminator
        save_model(self.gen, CFG.model_gen_path)
        save_model(self.dis, CFG.model_dis_path)

        # Output the parameters of the generator
        save_gen_final_params(self.gen, CFG.gen_final_params_path)

        endtime = datetime.now()
        print_and_log(f"\nRun took: {endtime - starttime} time.", CFG.log_path)
