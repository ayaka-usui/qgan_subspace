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

"""The plot tool"""

import os
import re

import matplotlib as mpl
import numpy as np

from tools.data.data_managers import print_and_log

mpl.use("Agg")
import matplotlib.pyplot as plt


########################################################################
# MAIN PLOTTING FUNCTION
########################################################################
def generate_all_plots(base_path, log_path, n_runs, max_fidelity, common_initial_plateaus):
    # Plot for each run
    for run_idx in range(1, n_runs + 1):
        plot_recurrence_vs_fid(base_path, log_path, run_idx, max_fidelity, common_initial_plateaus)

    # Plot all runs together (overwrites each time)
    plot_comparison_all_runs(base_path, log_path, n_runs, max_fidelity, common_initial_plateaus)

    # Plot average best fidelity per run
    plot_avg_best_fid_per_run(base_path, log_path, n_runs, max_fidelity, common_initial_plateaus)

    # Plot percent of runs above max_fidelity per run
    plot_success_percent_per_run(base_path, log_path, n_runs, max_fidelity, common_initial_plateaus)

    # Plot separated plateaus only when there are initial plateaus
    if common_initial_plateaus:
        # Plot separated plateaus - average best fidelity per run
        plot_avg_best_fid_per_run_separated(base_path, log_path, n_runs, max_fidelity, common_initial_plateaus)

        # Plot separated plateaus - success percent per run
        plot_success_percent_per_run_separated(base_path, log_path, n_runs, max_fidelity, common_initial_plateaus)


########################################################################
# REAL TIME RUN PLOTTING FUNCTION
########################################################################
def plt_fidelity_vs_iter(fidelities, losses, config, indx=0):
    fig, (axs1, axs2) = plt.subplots(1, 2)
    axs1.plot(range(len(fidelities)), fidelities)
    axs1.set_xlabel("Iteration")
    axs1.set_ylabel("Fidelity")
    axs1.set_title("Fidelity <target|gen> vs Iterations")
    axs2.plot(range(len(losses)), losses)
    axs2.set_xlabel("Iteration")
    axs2.set_ylabel("Loss")
    axs2.set_title("Wasserstein Loss vs Iterations")
    plt.tight_layout()

    # Save the figure
    fig_path = f"{config.figure_path}/{config.system_size}qubit_{config.gen_layers}_{indx}.png"
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    plt.savefig(fig_path)
    plt.close()


#########################################################################
# PLOT INDIVIDUAL RUNS HISTOGRAMS
#########################################################################
def plot_recurrence_vs_fid(base_path, log_path, run_idx, max_fidelity, common_initial_plateaus):
    run_colors = plt.cm.tab10.colors  # Consistent palette for control and runs
    control_fids = (
        collect_max_fidelities_nested(base_path, r"repeated_control", None) if common_initial_plateaus else []
    )
    changed_fids = collect_latest_changed_fidelities_nested(base_path, common_initial_plateaus, run_idx)
    # Split last bin at max_fidelity
    bins = [*list(np.linspace(0, max_fidelity, 20)), max_fidelity, 1.0]
    control_hist, _ = np.histogram(control_fids, bins=bins) if control_fids else (np.zeros(len(bins) - 1), bins)
    changed_hist, _ = np.histogram(changed_fids, bins=bins)
    # Renormalize histograms to show distributions
    control_hist = control_hist / control_hist.sum() if control_hist.sum() > 0 else control_hist
    changed_hist = changed_hist / changed_hist.sum() if changed_hist.sum() > 0 else changed_hist
    bin_centers = (np.array(bins[:-1]) + np.array(bins[1:])) / 2
    plt.figure(figsize=(8, 6))
    width = (bins[1] - bins[0]) * 0.4
    bars = []
    if common_initial_plateaus and np.any(control_hist):
        bars.append(
            plt.bar(
                bin_centers - width / 2,
                control_hist,
                width=width,
                label=f"Control (no change) ({len(control_fids)} tries)",
                alpha=0.7,
                color=run_colors[0],
            )
        )
    if np.any(changed_hist):
        # Use the second color from the palette for the first run, or cycle if run_idx is given
        run_color = run_colors[run_idx % len(run_colors)] if run_idx else run_colors[1]
        run_label = (
            f"Run {run_idx} ({len(changed_fids)} tries)" if run_idx else f"Experiment Runs ({len(changed_fids)} tries)"
        )
        bars.append(
            plt.bar(
                bin_centers + width / 2,
                changed_hist,
                width=width,
                label=run_label,
                alpha=0.7,
                color=run_color,
            )
        )
    plt.xlabel("Maximum Fidelity Reached")
    plt.ylabel("Distribution (Fraction)")
    title = "Distribution vs Maximum Fidelity"
    if run_idx:
        title += f" (run {run_idx})"
    elif not common_initial_plateaus:
        title += " (Experiment Mode)"
    plt.title(title)
    if bars:
        plt.legend()
    plt.grid(True)
    save_path = os.path.join(
        base_path,
        f"comparison_distribution_vs_fidelity_run{run_idx}.png" if run_idx else "distribution_vs_fidelity.png",
    )
    plt.tight_layout()
    plt.savefig(save_path)
    print_and_log(f"Saved plot to {save_path}", log_path)
    plt.close()


###########################################################################
# PLOT COMPARISON OF ALL HISTOGRAMS TOGETHER
###########################################################################
def plot_comparison_all_runs(base_path, log_path, n_runs, max_fidelity, common_initial_plateaus):
    run_colors = plt.cm.tab10.colors
    control_fids = (
        collect_max_fidelities_nested(base_path, r"repeated_control", None) if common_initial_plateaus else []
    )
    # Split last bin at max_fidelity
    bins = [*list(np.linspace(0, max_fidelity, 20)), max_fidelity, 1.0]
    bin_centers = (np.array(bins[:-1]) + np.array(bins[1:])) / 2
    plt.figure(figsize=(10, 7))
    all_hists = []
    all_labels = []
    all_colors = []
    # Collect control as first 'run' if present
    if common_initial_plateaus and len(control_fids) > 0:
        control_hist, _ = np.histogram(control_fids, bins=bins)
        control_hist = control_hist / control_hist.sum() if control_hist.sum() > 0 else control_hist
        all_hists.append(control_hist)
        all_labels.append(f"Control (no change) ({len(control_fids)} tries)")
        all_colors.append(run_colors[0])
    # Collect all runs
    for run_idx in range(1, n_runs + 1):
        if common_initial_plateaus:
            changed_fids = collect_latest_changed_fidelities_nested_run(base_path, run_idx)
        else:
            changed_fids = collect_latest_changed_fidelities_nested(base_path, common_initial_plateaus, run_idx)
        changed_hist, _ = np.histogram(changed_fids, bins=bins)
        changed_hist = changed_hist / changed_hist.sum() if changed_hist.sum() > 0 else changed_hist
        all_hists.append(changed_hist)
        all_labels.append(f"Run {run_idx} ({len(changed_fids)} tries)")
        all_colors.append(run_colors[run_idx % len(run_colors)])
    # Plot as grouped bars: each group is a run (control is group 0 if present)
    n_groups = len(all_hists)
    width = (bins[1] - bins[0]) * 0.7 / n_groups
    for i, (hist, label, color) in enumerate(zip(all_hists, all_labels, all_colors)):
        plt.bar(
            bin_centers + width * (i - n_groups / 2 + 0.5),
            hist,
            width=width,
            label=label,
            alpha=0.7,
            color=color,
        )
    plt.xlabel("Maximum Fidelity Reached")
    plt.ylabel("Distribution (Fraction)")
    title = "Comparison: Distribution vs Maximum Fidelity (All Runs)"
    if not common_initial_plateaus:
        title += " (Experiment Mode)"
    plt.title(title)
    if n_groups > 0:
        plt.legend()
    plt.grid(True)
    save_path = os.path.join(base_path, "comparison_distribution_vs_fidelity_all.png")
    plt.tight_layout()
    plt.savefig(save_path)
    print_and_log(f"Saved plot to {save_path}", log_path)
    plt.close()


##########################################################################
# PLOT AVERAGE BEST FIDELITY PER RUN
##########################################################################
def plot_avg_best_fid_per_run(base_path, log_path, n_runs, max_fidelity, common_initial_plateaus):
    avgs = []
    x_ticks = []
    x_labels = []

    for run_idx in range(1, n_runs + 1):
        if common_initial_plateaus:
            changed_fids = collect_latest_changed_fidelities_nested_run(base_path, run_idx)
        else:
            changed_fids = collect_latest_changed_fidelities_nested(base_path, common_initial_plateaus, run_idx)
        if changed_fids:
            avgs.append(np.nanmean(changed_fids))
        else:
            avgs.append(0)

        n_tries = count_tries_for_run(base_path, run_idx, common_initial_plateaus)
        x_ticks.append(run_idx)
        x_labels.append(f"Run {run_idx}\n({n_tries} tries)")

    plt.figure(figsize=(8, 5))
    x = np.arange(1, n_runs + 1)
    plt.plot(x, avgs, "o", color="green", label="Runs Avg", markersize=6)
    # Add value labels above each point
    for xi, yi in zip(x, avgs):
        plt.text(xi, yi + 0.01, f"{yi:.3f}", ha="center", va="bottom", fontsize=9)
    # Add control data as a distinct point if in initial mode
    if common_initial_plateaus:
        if control_fids := collect_max_fidelities_nested(base_path, r"repeated_control", None):
            control_avg = np.nanmean(control_fids)
            plt.plot([0], [control_avg], "s", color="blue", label="Control Avg", markersize=8)
            plt.text(
                0,
                control_avg + 0.01,
                f"{control_avg:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
            # Add control to x-axis labels
            control_tries = count_tries_control(base_path)
            x_ticks.insert(0, 0)
            x_labels.insert(0, f"Control\n({control_tries} tries)")

    plt.axhline(max_fidelity, color="C0", linestyle="--", label=f"max_fidelity={max_fidelity}")
    plt.xlabel("Run index")
    plt.ylabel("Average of Best Fidelity Achieved")

    # Create title with number of plateaus
    if common_initial_plateaus:
        # Count plateaus from any run to determine total number
        plateau_fids = collect_fidelities_by_plateau_for_run(base_path, 1)
        n_plateaus = len(plateau_fids) if plateau_fids else 0
        if n_plateaus > 0:
            plt.title(f"Average Best Fidelity per Run ({n_plateaus} Plateaus Averaged)")
        else:
            plt.title("Average Best Fidelity per Run")
    else:
        plt.title("Average Best Fidelity per Run")

    plt.ylim(0, 1.05)
    plt.xticks(x_ticks, x_labels)
    plt.grid(True, alpha=0.3)
    if common_initial_plateaus:
        plt.legend()
    save_path = os.path.join(base_path, "avg_best_fidelity_per_run.png")
    plt.tight_layout()
    plt.savefig(save_path)
    print_and_log(f"Saved plot to {save_path}", log_path)
    plt.close()


def plot_success_percent_per_run(base_path, log_path, n_runs, max_fidelity, common_initial_plateaus):
    percents = []
    x_ticks = []
    x_labels = []

    for run_idx in range(1, n_runs + 1):
        changed_fids = collect_latest_changed_fidelities_nested(base_path, common_initial_plateaus, run_idx)
        perc = 100 * np.sum(np.array(changed_fids) >= max_fidelity) / len(changed_fids) if changed_fids else 0
        percents.append(perc)

        n_tries = count_tries_for_run(base_path, run_idx, common_initial_plateaus)
        x_ticks.append(run_idx)
        x_labels.append(f"Run {run_idx}\n({n_tries} tries)")

    plt.figure(figsize=(8, 5))
    x = np.arange(1, n_runs + 1)
    plt.plot(x, percents, "o", color="red", label="Runs Success", markersize=6)
    # Add value labels above each point
    for xi, yi in zip(x, percents):
        plt.text(xi, yi + 1, f"{yi:.1f}%", ha="center", va="bottom", fontsize=9)
    # Add control data as a distinct point if in initial mode
    if common_initial_plateaus:
        if control_fids := collect_max_fidelities_nested(base_path, r"repeated_control", None):
            control_success = 100 * np.sum(np.array(control_fids) >= max_fidelity) / len(control_fids)
            plt.plot([0], [control_success], "s", color="blue", label="Control Success", markersize=8)
            plt.text(
                0,
                control_success + 1,
                f"{control_success:.1f}%",
                ha="center",
                va="bottom",
                fontsize=9,
            )
            # Add control to x-axis labels
            control_tries = count_tries_control(base_path)
            x_ticks.insert(0, 0)
            x_labels.insert(0, f"Control\n({control_tries} tries)")

    plt.xlabel("Run index")
    plt.ylabel(f"% of Runs with Fidelity ≥ {max_fidelity}")

    # Create title with number of plateaus
    if common_initial_plateaus:
        # Count plateaus from any run to determine total number
        plateau_fids = collect_fidelities_by_plateau_for_run(base_path, 1)
        n_plateaus = len(plateau_fids) if plateau_fids else 0
        if n_plateaus > 0:
            plt.title(f"Success Rate per Run ({n_plateaus} Plateaus Averaged)")
        else:
            plt.title("Success Rate per Run")
    else:
        plt.title("Success Rate per Run")

    plt.ylim(0, 105)
    plt.xticks(x_ticks, x_labels)
    plt.grid(True, alpha=0.3)
    if common_initial_plateaus:
        plt.legend()
    save_path = os.path.join(base_path, "success_percent_per_run.png")
    plt.savefig(save_path)
    print_and_log(f"Saved plot to {save_path}", log_path)
    plt.close()


##########################################################################
# HELPER FUNCTIONS TO COLLECT MAX FIDELITIES
##########################################################################
def count_tries_for_run(base_path, run_idx, common_initial_plateaus):
    """Count total tries for a specific run."""
    if common_initial_plateaus:
        changed_fids = collect_latest_changed_fidelities_nested_run(base_path, run_idx)
    else:
        changed_fids = collect_latest_changed_fidelities_nested(base_path, common_initial_plateaus, run_idx)
    return len(changed_fids) if changed_fids else 0


def count_tries_control(base_path):
    """Count total tries for control."""
    control_fids = collect_max_fidelities_nested(base_path, r"repeated_control", None)
    return len(control_fids) if control_fids else 0


def get_max_fidelity_from_file(fid_loss_path):
    if not os.path.exists(fid_loss_path):
        return None
    try:
        data = np.loadtxt(fid_loss_path)
        if data.ndim == 1:
            fidelities = data
        else:
            fidelities = data[0] if data.shape[0] < data.shape[1] else data[:, 0]
        return np.max(fidelities)
    except Exception:
        return None


def collect_max_fidelities_nested(base_path, outer_pattern, inner_pattern):
    """
    Collect max fidelities from all outer_pattern/inner_pattern/fidelities/log_fidelity_loss.txt
    """
    max_fids = []
    for root, dirs, files in os.walk(base_path):
        if (
            re.search(outer_pattern, root)
            and (inner_pattern is None or re.search(inner_pattern, root))
            and os.path.basename(root) == "fidelities"
            and "log_fidelity_loss.txt" in files
        ):
            fid_loss_path = os.path.join(root, "log_fidelity_loss.txt")
            max_fid = get_max_fidelity_from_file(fid_loss_path)
            if max_fid is not None:
                max_fids.append(max_fid)
    return max_fids


def collect_latest_changed_fidelities_nested(base_path, common_initial_plateaus, run_idx=None):
    """
    Collect max fidelities for changed runs, supporting both folder structures.
    common_initial_plateaus: boolean, if True, uses the initial plateaus structure.
    If run_idx is not None, only collect for that run.
    """
    run_dirs = {}
    if common_initial_plateaus:
        pattern = (
            f"initial_plateau_(\d+)/repeated_changed_run{run_idx}/(\d+)/fidelities$"
            if run_idx is not None
            else r"initial_plateau_(\d+)/repeated_changed_run(\d+)/(\d+)/fidelities$"
        )
    elif run_idx is not None:
        pattern = rf"experiment{run_idx}/(\d+)/fidelities$"
    else:
        pattern = r"experiment(\d+)/(\d+)/fidelities$"
    for root, dirs, files in os.walk(base_path):
        m = re.search(pattern, root)
        if m and "log_fidelity_loss.txt" in files:
            if common_initial_plateaus:
                if run_idx is not None:
                    run_y = run_idx
                    x_num = int(m[2])
                else:
                    run_y = int(m[2])
                    x_num = int(m[3])
                exp_j = int(m[1])
                key = (exp_j, x_num)
            else:
                if run_idx is not None:
                    run_y = run_idx
                    x_num = int(m[1])
                else:
                    run_y = int(m[1])
                    x_num = int(m[2])
                key = (run_y, x_num)
            if key not in run_dirs or run_y > run_dirs[key][0]:
                run_dirs[key] = (run_y, os.path.join(root, "log_fidelity_loss.txt"))
    max_fids = []
    for run_y, fid_loss_path in run_dirs.values():
        max_fid = get_max_fidelity_from_file(fid_loss_path)
        if max_fid is not None:
            max_fids.append(max_fid)
    return max_fids


def collect_latest_changed_fidelities_nested_run(base_path, run_idx):
    run_dirs = {}
    for root, dirs, files in os.walk(base_path):
        m = re.search(
            r"initial_plateau_(\d+)[/\\]repeated_changed_run" + str(run_idx) + r"[/\\](\d+)[/\\]fidelities$", root
        )
        if m and "log_fidelity_loss.txt" in files:
            exp_j = int(m[1])
            x_num = int(m[2])
            key = (exp_j, x_num)
            run_y = run_idx
            if key not in run_dirs or run_y > run_dirs[key][0]:
                run_dirs[key] = (run_y, os.path.join(root, "log_fidelity_loss.txt"))
    max_fids = []
    for run_y, fid_loss_path in run_dirs.values():
        max_fid = get_max_fidelity_from_file(fid_loss_path)
        if max_fid is not None:
            max_fids.append(max_fid)
    return max_fids


##########################################################################
# PLOT AVERAGE BEST FIDELITY PER RUN - SEPARATED PLATEAUS
##########################################################################
def plot_avg_best_fid_per_run_separated(base_path, log_path, n_runs, max_fidelity, common_initial_plateaus):
    plt.figure(figsize=(10, 6))

    # Collect x-axis positions and labels
    x_ticks = []
    x_labels = []
    plateau_data_by_run = {}

    # Collect and plot data for each run
    for run_idx in range(1, n_runs + 1):
        if common_initial_plateaus:
            # Get fidelities grouped by plateau for this run
            plateau_fids = collect_fidelities_by_plateau_for_run(base_path, run_idx)
            run_avgs = []
            total_tries = sum(len(plateau_data) for plateau_data in plateau_fids.values())
            for plateau_num, plateau_data in sorted(plateau_fids.items()):
                avg = np.nanmean(plateau_data) if plateau_data else 0
                run_avgs.append(avg)
                # Store plateau data for connecting lines
                if plateau_num not in plateau_data_by_run:
                    plateau_data_by_run[plateau_num] = {}
                plateau_data_by_run[plateau_num][run_idx] = avg
        else:
            # For non-plateau mode, same as regular plot
            changed_fids = collect_latest_changed_fidelities_nested(base_path, common_initial_plateaus, run_idx)
            avg = np.nanmean(changed_fids) if changed_fids else 0
            run_avgs = [avg]
            total_tries = len(changed_fids) if changed_fids else 0

        # Plot points for this run with small horizontal offset for visibility
        x_positions = [run_idx + np.random.uniform(-0.1, 0.1) for _ in run_avgs]
        plt.scatter(x_positions, run_avgs, alpha=0.7, s=40)

        x_ticks.append(run_idx)
        x_labels.append(f"Run {run_idx}\n({total_tries} tries)")

    # Add control data if in plateau mode
    if common_initial_plateaus:
        if control_plateau_fids := collect_fidelities_by_plateau_control(base_path):
            control_avgs = []
            total_control_tries = sum(len(plateau_data) for plateau_data in control_plateau_fids.values())
            for plateau_num, plateau_data in sorted(control_plateau_fids.items()):
                avg = np.nanmean(plateau_data) if plateau_data else 0
                control_avgs.append(avg)
                # Store control plateau data for connecting lines
                if plateau_num not in plateau_data_by_run:
                    plateau_data_by_run[plateau_num] = {}
                plateau_data_by_run[plateau_num][0] = avg
            x_positions = [0 + np.random.uniform(-0.1, 0.1) for _ in control_avgs]
            plt.scatter(x_positions, control_avgs, alpha=0.7, s=50, color="blue", marker="s")

            x_ticks.insert(0, 0)
            x_labels.insert(0, f"Control\n({total_control_tries} tries)")

    # Connect same plateaus with lines
    if common_initial_plateaus:
        colors = plt.cm.Set3(np.linspace(0, 1, len(plateau_data_by_run)))
        for i, (plateau_num, plateau_data) in enumerate(sorted(plateau_data_by_run.items())):
            x_coords = []
            y_coords = []
            for run_idx in sorted(plateau_data.keys()):
                x_coords.append(run_idx)
                y_coords.append(plateau_data[run_idx])
            plt.plot(x_coords, y_coords, "--", alpha=0.5, color=colors[i])

    plt.axhline(max_fidelity, color="C0", linestyle="--", label=f"max_fidelity={max_fidelity}")
    plt.xlabel("Run index")
    plt.ylabel("Average of Best Fidelity Achieved")

    # Create title with number of plateaus
    if common_initial_plateaus and plateau_data_by_run:
        n_plateaus = len(plateau_data_by_run)
        plt.title(f"Average Best Fidelity per Run ({n_plateaus} Plateaus Separated)")
    else:
        plt.title("Average Best Fidelity per Run (Plateaus Separated)")

    plt.ylim(0, 1.05)
    plt.xticks(x_ticks, x_labels)
    plt.grid(True, alpha=0.3)

    # Create simple legend like non-separated plots
    legend_elements = []
    legend_elements.append(
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="gray",
            markersize=6,
            alpha=0.7,
            linestyle="None",
            label="Runs Avg",
        )
    )
    if common_initial_plateaus and any("Control" in label for label in x_labels):
        legend_elements.append(
            plt.Line2D(
                [0],
                [0],
                marker="s",
                color="w",
                markerfacecolor="blue",
                markersize=8,
                alpha=0.7,
                linestyle="None",
                label="Control Avg",
            )
        )
    legend_elements.append(plt.Line2D([0], [0], color="C0", linestyle="--", label=f"max_fidelity={max_fidelity}"))
    plt.legend(handles=legend_elements)
    save_path = os.path.join(base_path, "avg_best_fidelity_per_run_separated.png")
    plt.tight_layout()
    plt.savefig(save_path)
    print_and_log(f"Saved plot to {save_path}", log_path)
    plt.close()


def plot_success_percent_per_run_separated(base_path, log_path, n_runs, max_fidelity, common_initial_plateaus):
    plt.figure(figsize=(10, 6))

    # Collect x-axis positions and labels
    x_ticks = []
    x_labels = []
    plateau_data_by_run = {}

    # Collect and plot data for each run
    for run_idx in range(1, n_runs + 1):
        if common_initial_plateaus:
            # Get fidelities grouped by plateau for this run
            plateau_fids = collect_fidelities_by_plateau_for_run(base_path, run_idx)
            run_percents = []
            total_tries = sum(len(plateau_data) for plateau_data in plateau_fids.values())
            for plateau_num, plateau_data in sorted(plateau_fids.items()):
                perc = 100 * np.sum(np.array(plateau_data) >= max_fidelity) / len(plateau_data) if plateau_data else 0
                run_percents.append(perc)
                # Store plateau data for connecting lines
                if plateau_num not in plateau_data_by_run:
                    plateau_data_by_run[plateau_num] = {}
                plateau_data_by_run[plateau_num][run_idx] = perc
        else:
            # For non-plateau mode, same as regular plot
            changed_fids = collect_latest_changed_fidelities_nested(base_path, common_initial_plateaus, run_idx)
            perc = 100 * np.sum(np.array(changed_fids) >= max_fidelity) / len(changed_fids) if changed_fids else 0
            run_percents = [perc]
            total_tries = len(changed_fids) if changed_fids else 0

        # Plot points for this run with small horizontal offset for visibility
        x_positions = [run_idx + np.random.uniform(-0.1, 0.1) for _ in run_percents]
        plt.scatter(x_positions, run_percents, alpha=0.7, s=40)

        x_ticks.append(run_idx)
        x_labels.append(f"Run {run_idx}\n({total_tries} tries)")

    # Add control data if in plateau mode
    if common_initial_plateaus:
        if control_plateau_fids := collect_fidelities_by_plateau_control(base_path):
            control_percents = []
            total_control_tries = sum(len(plateau_data) for plateau_data in control_plateau_fids.values())
            for plateau_num, plateau_data in sorted(control_plateau_fids.items()):
                perc = 100 * np.sum(np.array(plateau_data) >= max_fidelity) / len(plateau_data) if plateau_data else 0
                control_percents.append(perc)
                # Store control plateau data for connecting lines
                if plateau_num not in plateau_data_by_run:
                    plateau_data_by_run[plateau_num] = {}
                plateau_data_by_run[plateau_num][0] = perc
            x_positions = [0 + np.random.uniform(-0.1, 0.1) for _ in control_percents]
            plt.scatter(x_positions, control_percents, alpha=0.7, s=50, color="blue", marker="s")

            x_ticks.insert(0, 0)
            x_labels.insert(0, f"Control\n({total_control_tries} tries)")

    # Connect same plateaus with lines
    if common_initial_plateaus:
        colors = plt.cm.Set3(np.linspace(0, 1, len(plateau_data_by_run)))
        for i, (plateau_num, plateau_data) in enumerate(sorted(plateau_data_by_run.items())):
            x_coords = []
            y_coords = []
            for run_idx in sorted(plateau_data.keys()):
                x_coords.append(run_idx)
                y_coords.append(plateau_data[run_idx])
            plt.plot(x_coords, y_coords, "--", alpha=0.5, color=colors[i])

    plt.xlabel("Run index")
    plt.ylabel(f"% of Runs with Fidelity ≥ {max_fidelity}")

    # Create title with number of plateaus
    if common_initial_plateaus and plateau_data_by_run:
        n_plateaus = len(plateau_data_by_run)
        plt.title(f"Success Rate per Run ({n_plateaus} Plateaus Separated)")
    else:
        plt.title("Success Rate per Run (Plateaus Separated)")

    plt.ylim(0, 105)
    plt.xticks(x_ticks, x_labels)
    plt.grid(True, alpha=0.3)

    # Create simple legend like non-separated plots
    legend_elements = []
    legend_elements.append(
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="red",
            markersize=6,
            alpha=0.7,
            linestyle="None",
            label="Runs Success",
        )
    )
    if common_initial_plateaus and any("Control" in label for label in x_labels):
        legend_elements.append(
            plt.Line2D(
                [0],
                [0],
                marker="s",
                color="w",
                markerfacecolor="blue",
                markersize=8,
                alpha=0.7,
                linestyle="None",
                label="Control Success",
            )
        )
    plt.legend(handles=legend_elements)
    save_path = os.path.join(base_path, "success_percent_per_run_separated.png")
    plt.tight_layout()
    plt.savefig(save_path)
    print_and_log(f"Saved plot to {save_path}", log_path)
    plt.close()


def collect_fidelities_by_plateau_for_run(base_path, run_idx):
    """Collect fidelities grouped by plateau for a specific run."""
    plateau_fids = {}
    for root, dirs, files in os.walk(base_path):
        m = re.search(rf"initial_plateau_(\d+)[/\\]repeated_changed_run{run_idx}[/\\](\d+)[/\\]fidelities$", root)
        if m and "log_fidelity_loss.txt" in files:
            plateau_num = int(m[1])
            fid_loss_path = os.path.join(root, "log_fidelity_loss.txt")
            max_fid = get_max_fidelity_from_file(fid_loss_path)
            if max_fid is not None:
                if plateau_num not in plateau_fids:
                    plateau_fids[plateau_num] = []
                plateau_fids[plateau_num].append(max_fid)
    return plateau_fids


def collect_fidelities_by_plateau_control(base_path):
    """Collect control fidelities grouped by plateau."""
    plateau_fids = {}
    for root, dirs, files in os.walk(base_path):
        m = re.search(r"initial_plateau_(\d+)[/\\]repeated_control[/\\]fidelities$", root)
        if m and "log_fidelity_loss.txt" in files:
            plateau_num = int(m[1])
            fid_loss_path = os.path.join(root, "log_fidelity_loss.txt")
            max_fid = get_max_fidelity_from_file(fid_loss_path)
            if max_fid is not None:
                if plateau_num not in plateau_fids:
                    plateau_fids[plateau_num] = []
                plateau_fids[plateau_num].append(max_fid)
    return plateau_fids
