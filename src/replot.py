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

"""Module, for manual reploting of a generated_data/MULTIPLE_RUNS/timestamp."""

import os

from tools.plot_hub import generate_all_plots

#######################################################################
# Parameters for the replotting script
#######################################################################
time_stamp_to_replot = "2025-07-03__15-03-02"
n_runs = 5
max_fidelity = 0.99
common_initial_experiment = False

#######################################################################
# Replotting script for the specified experiment
#######################################################################
# Path to the experiment folder
base_path = os.path.join("generated_data", "MULTIPLE_RUNS", time_stamp_to_replot)
log_path = os.path.join(base_path, "replot_log.txt")

print(f"Replotting for MULTIPLE_RUNS/{time_stamp_to_replot} with {n_runs} experiments")

# Plot:
generate_all_plots(
    base_path,
    log_path,
    n_runs=n_runs,
    max_fidelity=max_fidelity,
    common_initial_experiments=common_initial_experiment,
)
