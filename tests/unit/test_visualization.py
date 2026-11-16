"""Unit tests for the scientific visualization and plotting engines."""

from pathlib import Path

import numpy as np
import pytest

from complexity_lab.visualization.plots import (
    plot_damage_spreading,
    plot_parameter_sweep,
    plot_phase_space,
    plot_time_series,
)
from complexity_lab.visualization.renderers import (
    render_grid_image,
    render_trajectory_gif,
)


@pytest.mark.unit
def test_render_grid_image(tmp_path: Path) -> None:
    grid = np.array([
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 0],
    ], dtype=np.int8)

    out_file = tmp_path / "test_grid.png"
    result_path = render_grid_image(grid, out_file, scale=4)
    assert result_path.exists()
    assert result_path.stat().st_size > 50

    # Test with title (matplotlib path)
    out_titled = tmp_path / "test_grid_titled.png"
    result_titled = render_grid_image(grid, out_titled, title="Cross Pattern")
    assert result_titled.exists()
    assert result_titled.stat().st_size > 100


@pytest.mark.unit
def test_render_trajectory_gif(tmp_path: Path) -> None:
    s0 = np.zeros((4, 4), dtype=np.int8)
    s1 = np.ones((4, 4), dtype=np.int8)
    trajectory = [s0, s1, s0, s1]

    out_file = tmp_path / "test_animation.gif"
    result_path = render_trajectory_gif(trajectory, out_file, fps=5, scale=4)
    assert result_path.exists()
    assert result_path.stat().st_size > 100

    with pytest.raises(ValueError, match="empty trajectory"):
        render_trajectory_gif([], tmp_path / "empty.gif")


@pytest.mark.unit
def test_plot_time_series(tmp_path: Path) -> None:
    data = {
        "density": [0.1, 0.15, 0.2, 0.18],
        "shannon_entropy": [0.3, 0.45, 0.6, 0.55],
        "activity": [10.0, 14.0, 8.0, 4.0],
    }
    out_file = tmp_path / "test_dynamics.png"
    result_path = plot_time_series(data, out_file)
    assert result_path.exists()
    assert result_path.stat().st_size > 1000

    with pytest.raises(ValueError, match="empty time series"):
        plot_time_series({}, tmp_path / "empty.png")


@pytest.mark.unit
def test_plot_parameter_sweep(tmp_path: Path) -> None:
    sweep_results = [
        {"value": 0.1, "mean_density": 0.05, "final_entropy": 0.2},
        {"value": 0.2, "mean_density": 0.12, "final_entropy": 0.5},
        {"value": 0.3, "mean_density": 0.22, "final_entropy": 0.8},
    ]
    out_file = tmp_path / "test_sweep.png"
    result_path = plot_parameter_sweep(sweep_results, "init_density", out_file)
    assert result_path.exists()
    assert result_path.stat().st_size > 1000

    with pytest.raises(ValueError, match="cannot be empty"):
        plot_parameter_sweep([], "density", tmp_path / "empty.png")


@pytest.mark.unit
def test_plot_damage_spreading(tmp_path: Path) -> None:
    series = [1, 2, 5, 8, 12, 11, 14]
    out_file = tmp_path / "test_damage.png"
    result_path = plot_damage_spreading(series, out_file)
    assert result_path.exists()
    assert result_path.stat().st_size > 1000

    with pytest.raises(ValueError, match="cannot be empty"):
        plot_damage_spreading([], tmp_path / "empty.png")


@pytest.mark.unit
def test_plot_phase_space(tmp_path: Path) -> None:
    densities = [0.1, 0.15, 0.2, 0.22]
    entropies = [0.3, 0.45, 0.6, 0.65]
    out_file = tmp_path / "test_phase_space.png"
    result_path = plot_phase_space(densities, entropies, out_file)
    assert result_path.exists()
    assert result_path.stat().st_size > 1000

    with pytest.raises(ValueError, match="identical"):
        plot_phase_space([0.1], [0.2, 0.3], tmp_path / "err.png")

