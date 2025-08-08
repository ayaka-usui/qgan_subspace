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

"""Module, for manual replotting of a generated_data/MULTIPLE_RUNS/timestamp."""

import os

from tools.data.data_managers import get_last_experiment_idx
from tools.plot_hub import find_if_common_initial_plateaus, generate_all_plots

#######################################################################
# Parameters for the replotting script
#######################################################################
time_stamp_to_replot = "2025-07-28__18-50-22_Diff_Hs_complexity_ancilla_vs_no_ancilla-from_start"
max_fidelity = 0.99
x_label = "Hamiltonian"
run_names = [
    "0.2 XZX + 0.8 ZZ",
    "+ ancilla",
    "0.5 XZX + 0.5 ZZ",
    "+ ancilla",
    "0.8 XZX + 0.2 ZZ",
    "+ ancilla",
]

#######################################################################
# Replotting script for the specified experiment
#######################################################################
# Path to the experiment folder
base_path = os.path.join("generated_data", "MULTIPLE_RUNS", time_stamp_to_replot)
log_path = os.path.join(base_path, "replot_log.txt")

# Extract the number of runs and whether there are common initial plateaus
common_initial_plateaus = find_if_common_initial_plateaus(base_path)
n_runs = get_last_experiment_idx(base_path, common_initial_plateaus)

print(f"Replotting for MULTIPLE_RUNS/{time_stamp_to_replot} with {n_runs} experiments")

# Plot:
generate_all_plots(
    base_path,
    log_path,
    n_runs=n_runs,
    max_fidelity=max_fidelity,
    common_initial_plateaus=common_initial_plateaus,
    run_names=run_names,
    x_label=x_label,
)
