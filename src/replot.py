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

max_fidelity = 0.99
x_label = "Ancilla Topology"

# time_stamp_to_replot = "Escaping_plateaus_with_ancilla/2025-07-11__16-58-40_NEW_DATA_FILTERED_NEW_ORDER"
# run_names = [
#     "",
#     "Ansatz",
#     "Bridge No1Q",
#     "Short Bridge",
#     "Bridge",
#     "Total",
# ]

time_stamp_to_replot = "Adding-ancilla-mid-run/2025-08-06__00-23-46-Helps_REORDER"
run_names = ["Bridge NoRand", "Total NoRand", "Bridge Rand", "Total Rand"]

# time_stamp_to_replot = "2025-08-06__00-35-39_Fid-vs-Epochs [DONE]"
# run_names = ["1", "3", "5", "10", "20"]
# x_label = "Epochs"

# time_stamp_to_replot = "2025-08-08__01-11-23"
# x_label = "Gen - Dis steps"
# run_names = ["1-1", "2-1", "1-2", "2-2", "5-1", "1-5", "5-5", "10-10"]

time_stamp_to_replot = "2025-07-28__15-38-23-Diff-ancilla-topologies-from-start [DONE] [NEW]"
x_label = "Ancilla Topology"
run_names = [
    "No Ancilla",
    "Ansatz",
    "Bridge No1Q",
    "Short Bridge",
    "Bridge",
    "Total",
    "PrjBridgeReN",
    "PrjTotalReN",
    "PrjBridgePass",
    "PrjTotalPass",
]

# time_stamp_to_replot = "2025-07-28__18-50-22_Diff_Hs_complexity_ancilla_vs_no_ancilla-from_start [DONE] copy ising"
# x_label = "Target Hamiltonian"
# run_names = [
#     "ising",
#     "ising + bridge",
#     "ising + total",
# ]

# time_stamp_to_replot = "2025-07-28__18-50-22_Diff_Hs_complexity_ancilla_vs_no_ancilla-from_start [DONE] No Ising"
# x_label = "Target Hamiltonian"
# run_names = [
#     "0.2 XZX+0.8 ZZ",
#     "+ bridge",
#     "+ total",
#     "0.5 XZX+0.5 ZZ",
#     "+ bridge",
#     "+ total",
#     "0.8 XZX+0.2 ZZ",
#     "+ bridge",
#     "+ total",
# ]

# time_stamp_to_replot = (
#     "2025-07-11__16-58-40_escaping_plateaus_with_ancilla [MAIN]/2025-07-11__16-58-40_1st_best_plateau(13)"
# )
# x_label = ""
# run_names = ["Adding Ancilla"]

# time_stamp_to_replot = "2025-08-09__13-28-56-Fid-vs-lrate-and-momentum"
# x_label = "Learning Rate, Momentum Coefficient"
# run_names = [
#     "0.01, 0.9",
#     "0.01, 0.5",
#     "0.01, 1.5",
#     "0.2, 0.9",
#     "0.2, 0.5",
#     "0.2, 1.5",
# ]

# time_stamp_to_replot = "2025-08-09__13-28-56-Fid-vs-lrate [DONE]"
# x_label = "Learning Rate"
# run_names = [
#     "0.005",
#     "0.01",
#     "0.05",
#     "0.2",
#     "1.0",
# ]

# time_stamp_to_replot = "Escaping_plateaus_with_ancilla/2025-07-11__16-58-40_All_best_plateau(13 & 19)"
# x_label = ""
# run_names = ["1st plateau escape", "2nd plateau escape"]

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
