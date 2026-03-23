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
"""
Persistence and file management system — PennyLane version. 
It ensures that we do not lose anything during training 
and that the results are organised so we you can compare them later.

Changes from original:
    - The function 'save_gen_final_params(gen, file_path)'
"""
import os
import pickle

import numpy as np


def train_log(param, file_path):
    """It ensures that the folder where you are going to save the file exists. 
       If it does not exist, it creates it."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "a") as file: #Opens it in 'append' mode
        file.write(param)


def print_and_log(param, file_path):
    """Display the message on terminal (print) and at the same time save it 
    to the file using the previous function."""
    print(param)  # Console feedback
    train_log(param, file_path)  # Logging to file


def print_and_log_with_headers(param, file_path):
    """Simply add aesthetic lines (====) before and after the message."""
    print_and_log(f"\n{'=' * 60}", file_path)
    print_and_log(param, file_path)
    print_and_log(f"\n{'=' * 60}", file_path)

def save_model(model, file_path):
    """ Uses the Pickle library to save the model object (its weights, architecture, etc.) 
    in binary format (‘wb+’).This means that if training is interrupted, we can load the 
    .pkl file and continue where we left
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "wb+") as file:
        pickle.dump(model, file)


def save_fidelity_loss(fidelities_history, losses_history, file_path):
    """Save two lists of numbers: fidelities and losses.
    Use np.savetxt, which saves data in plain text format (a column of numbers). This is ideal for 
    later plotting with Matplotlib (plot_hub.py or replot.py)
    If the file already exists, delete it (os.remove) to write the new data from scratch."""
    if os.path.exists(file_path):
        os.remove(file_path)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "wb") as f:
        np.savetxt(f, fidelities_history)
        np.savetxt(f, losses_history)

def save_gen_final_params(gen, file_path):
    """This function is specific for the Quantum Generator.
    It does not save the entire object, only the final angles of the logic gates.
    Changed because in PennyLane gen.params (gen.qc.gates in previous code) 
    is already a flat array of angles """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    np.savetxt(file_path, gen.params.detach().numpy())
    #np.savetxt(file_path, gen.params)

def get_last_experiment_idx(base_path, common_initial_plateaus):
    """Return the highest experiment index in experimentX or 
    initial_plateau_1/repeated_changed_runX folders under base_path,
    preventing us from overwriting previous experiments if we run the code multiple times."""

    base_path = f"{base_path}/initial_plateau_1" if common_initial_plateaus else base_path
    start_with = "repeated_changed_run" if common_initial_plateaus else "experiment"
    experiment_dirs = [
        d for d in os.listdir(base_path) if d.startswith(start_with) and os.path.isdir(os.path.join(base_path, d))
    ]
    last_idx = 0
    for d in experiment_dirs:
        try:
            idx = int(d.replace(start_with, ""))
            if idx > last_idx:
                last_idx = idx
        except Exception:
            continue
    return last_idx