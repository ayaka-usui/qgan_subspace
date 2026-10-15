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
"""Module, for manual replotting of a generated_data/timestamp.

Two modes:
  - Standard mode: same as before, calls generate_all_plots(...).
  - Decile mode: when DECILE_MODE = True, generates one scatter per
    decile_ref using the manifest.csv produced by decile_training_crn.py.
    Each scatter has 4 columns (no_ancilla, ancilla_total, ancilla_bridge,
    ancilla_shortBridge), 20 seeds per column.
"""
import os
from tools.data_managers import get_last_experiment_idx
from tools.plot_hub import (
    find_if_common_initial_plateaus,
    generate_all_plots,
    scatter_plot_by_decile,
)

# ------- Parameters for the replotting script --------------
# EXAMPLE TO EDIT PARAMETERS:
# time_stamp_to_replot = "Batch ZZZ"
# max_fidelity = 0.99
# x_label = "Ancilla Topology"
# run_names = [
#     "Ansatz",
#     "Short Bridge",
#     "Bridge",
#     "Total",
# ]

time_stamp_to_replot = "gradient"
max_fidelity = 0.99

# -- Decile mode --------------------------------------------------------
# Set to True for outputs of decile_training_crn.py.  
# and produces one scatter_plot_D<dd>.png per decile.
DECILE_MODE = False

# Optional: explicit path to manifest.csv. If None, auto-discovered from
# decile_seeds/<VTS>/manifest.csv (sibling of decile_training/<VTS>).
DECILE_MANIFEST_PATH = None

# Display labels for the 4 columns (left to right).
DECILE_CONFIGS_ORDER = [
    "no_ancilla",
    "ancilla_total",
    "ancilla_bridge",
    "ancilla_shortBridge",
]
DECILE_CONFIG_DISPLAY = {
    "no_ancilla":          "no_ancilla",
    "ancilla_total":       "total",
    "ancilla_bridge":      "bridge",
    "ancilla_shortBridge": "shortBridge",
}

# If Decile_MODE is False
# Standard-mode-only fields 
x_label = "Ancilla Topology"
run_names = [
    "Ansatz",
    "ShortBridge",
    "Bridge",
    "Total",
]
# STOP EDITING HERE

# -------------- Replotting script for the specified experiment --------------
base_path = os.path.join("generated_data", time_stamp_to_replot)
log_path = os.path.join(base_path, "replot_log.txt")

if DECILE_MODE:
    # One scatter per decile_ref, 4 configs side-by-side, 20 seeds each.
    print(f"Replotting (decile mode) for {time_stamp_to_replot}")
    scatter_plot_by_decile(
        base_path=base_path,
        log_path=log_path,
        max_fidelity=max_fidelity,
        manifest_path=DECILE_MANIFEST_PATH,
        configs_order=DECILE_CONFIGS_ORDER,
        config_display=DECILE_CONFIG_DISPLAY,
    )
else:
    # Standard mode: histograms, scatter, gradient trajectory plots, etc.
    common_initial_plateaus = find_if_common_initial_plateaus(base_path)
    n_runs = get_last_experiment_idx(base_path, common_initial_plateaus)
    print(f"Replotting for {time_stamp_to_replot} with {n_runs} experiments")
    generate_all_plots(
        base_path,
        log_path,
        n_runs=n_runs,
        max_fidelity=max_fidelity,
        common_initial_plateaus=common_initial_plateaus,
        run_names=run_names,
        x_label=x_label,
    )

    # -- Replotting for the specified Barren Plateau --------------
    focus_plateau_ids = None # None, [1,5]
    if focus_plateau_ids:
        from tools.plot_hub import (plot_grad_trajectory_by_plateau,
                                    plot_grad_norm_trajectory_by_plateau,
                                    plot_grad_pr_trajectory_by_plateau
                                )
        plot_grad_trajectory_by_plateau(
            base_path, log_path, n_runs,
            plateau_ids=focus_plateau_ids,
            run_names=run_names,
            include_control=True,
            include_initial=True,
            fid_stride=None,  # inferred from the runs' save_fid_and_loss_every_x_iter
        )
        plot_grad_norm_trajectory_by_plateau(
            base_path, log_path, n_runs,
            plateau_ids=focus_plateau_ids,
            run_names=run_names,
            include_control=True,
            include_initial=True,
            fid_stride=None,
        )
        plot_grad_pr_trajectory_by_plateau(
            base_path, log_path, n_runs,
            plateau_ids=focus_plateau_ids,
            run_names=run_names,
            include_control=True,
            include_initial=True,
            fid_stride=None,
            window=20,
        )