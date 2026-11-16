"""Scientific plotting engines for simulation dynamics, phase space, and sweeps."""

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_time_series(
    time_series: dict[str, list[float]],
    output_path: Path | str,
    title: str = "Simulation Dynamics",
    dpi: int = 150,
) -> Path:
    """Plot observable trajectories over simulation timesteps.

    Args:
        time_series: Dictionary mapping metric names ('density', 'shannon_entropy', etc.)
                     to lists of values at each step.
        output_path: Destination path for the saved figure.
        title: Overall figure title.
        dpi: Image resolution.

    Returns:
        Path to the saved figure.
    """
    dest = Path(output_path)
    dest.parent.mkdir(parents=True, exist_ok=True)

    metrics = [k for k in ["density", "shannon_entropy", "activity"] if k in time_series]
    if not metrics:
        metrics = list(time_series.keys())

    if not metrics:
        raise ValueError("Cannot plot empty time series data.")

    n_plots = len(metrics)
    fig, axes = plt.subplots(n_plots, 1, figsize=(8, 3 * n_plots), dpi=dpi, sharex=True)
    if n_plots == 1:
        axes = [axes]

    colors = {"density": "#2980b9", "shannon_entropy": "#8e44ad", "activity": "#e67e22"}
    labels = {
        "density": "Active Cell Density (ρ)",
        "shannon_entropy": "Shannon Entropy (bits)",
        "activity": "Activity (cells changed/step)",
    }

    for ax, metric in zip(axes, metrics, strict=True):
        vals = time_series[metric]
        steps = list(range(len(vals)))
        color = colors.get(metric, "#2c3e50")
        label = labels.get(metric, metric.replace("_", " ").title())

        ax.plot(steps, vals, color=color, linewidth=2, label=label)
        ax.set_ylabel(label, fontsize=10, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="upper right", framealpha=0.8)

    axes[-1].set_xlabel("Timestep (t)", fontsize=11, fontweight="bold")
    fig.suptitle(title, fontsize=13, fontweight="bold", y=0.99)
    plt.tight_layout()
    fig.savefig(dest, bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    return dest


def plot_parameter_sweep(
    sweep_results: list[dict[str, Any]],
    parameter_name: str,
    output_path: Path | str,
    title: str = "Parameter Sweep & Phase Transition Analysis",
    dpi: int = 150,
) -> Path:
    """Plot order parameters and macroscopic metrics across a parameter sweep.

    Args:
        sweep_results: List of dictionaries containing evaluated 'value', 'mean_density',
                       'final_entropy', 'mean_activity', etc.
        parameter_name: The name of the parameter varied.
        output_path: Destination figure path.
        title: Figure title.
        dpi: Resolution.

    Returns:
        Path to the saved figure.
    """
    dest = Path(output_path)
    dest.parent.mkdir(parents=True, exist_ok=True)

    if not sweep_results:
        raise ValueError("Sweep results list cannot be empty.")

    values = [r["value"] for r in sweep_results]
    densities = [r.get("mean_density", 0.0) for r in sweep_results]
    entropies = [r.get("final_entropy", 0.0) for r in sweep_results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), dpi=dpi)

    # Subplot 1: Order Parameter - Density
    ax1.plot(values, densities, "o-", color="#2980b9", linewidth=2, markersize=6)
    ax1.set_xlabel(f"Parameter: {parameter_name}", fontsize=10, fontweight="bold")
    ax1.set_ylabel("Mean Density (ρ)", fontsize=10, fontweight="bold")
    ax1.set_title("Order Parameter: Mean Density", fontsize=11, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Subplot 2: Spatial Uncertainty - Entropy
    ax2.plot(values, entropies, "s-", color="#8e44ad", linewidth=2, markersize=6)
    ax2.set_xlabel(f"Parameter: {parameter_name}", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Final Shannon Entropy (bits)", fontsize=10, fontweight="bold")
    ax2.set_title("Macroscopic Entropy Profile", fontsize=11, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.5)

    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(dest, bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    return dest


def plot_damage_spreading(
    hamming_series: list[int],
    output_path: Path | str,
    title: str = "Perturbation Growth & Damage Spreading (Derrida Parameter)",
    dpi: int = 150,
) -> Path:
    """Plot the Hamming distance divergence between twin simulations over time.

    Args:
        hamming_series: Time series of Hamming distances between unperturbed and perturbed runs.
        output_path: Destination file path.
        title: Figure title.
        dpi: Resolution.

    Returns:
        Path to the saved figure.
    """
    dest = Path(output_path)
    dest.parent.mkdir(parents=True, exist_ok=True)

    if not hamming_series:
        raise ValueError("Hamming series cannot be empty.")

    steps = list(range(len(hamming_series)))

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=dpi)
    ax.plot(steps, hamming_series, color="#c0392b", linewidth=2, label="Hamming Distance d(t)")
    ax.fill_between(steps, 0, hamming_series, color="#c0392b", alpha=0.15)

    ax.set_xlabel("Timestep (t)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Perturbation Difference Cells d(t)", fontsize=11, fontweight="bold")
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper left")

    plt.tight_layout()
    fig.savefig(dest, bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    return dest


def plot_phase_space(
    density_series: list[float],
    entropy_series: list[float],
    output_path: Path | str,
    title: str = "Phase Space Trajectory (Density vs. Entropy)",
    dpi: int = 150,
) -> Path:
    """Plot attractor dynamics in the (Density, Shannon Entropy) phase space.

    Args:
        density_series: Sequence of density observables over time.
        entropy_series: Sequence of entropy observables over time.
        output_path: Destination file path.
        title: Figure title.
        dpi: Resolution.

    Returns:
        Path to the saved figure.
    """
    dest = Path(output_path)
    dest.parent.mkdir(parents=True, exist_ok=True)

    if len(density_series) != len(entropy_series) or not density_series:
        raise ValueError("Density and entropy series must have identical non-zero lengths.")

    fig, ax = plt.subplots(figsize=(6, 6), dpi=dpi)
    ax.plot(density_series, entropy_series, "-", color="#2c3e50", alpha=0.6, linewidth=1.5)
    ax.scatter(
        density_series[0],
        entropy_series[0],
        color="#27ae60",
        s=60,
        label="Start (t=0)",
        zorder=5,
    )
    ax.scatter(
        density_series[-1],
        entropy_series[-1],
        color="#e74c3c",
        s=60,
        label="Final State",
        zorder=5,
    )

    ax.set_xlabel("Density (ρ)", fontsize=10, fontweight="bold")
    ax.set_ylabel("Shannon Entropy (bits)", fontsize=10, fontweight="bold")
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="best")

    plt.tight_layout()
    fig.savefig(dest, bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    return dest
