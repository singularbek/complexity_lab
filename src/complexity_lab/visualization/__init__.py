"""Scientific visualization and trajectory rendering package."""

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

__all__ = [
    "plot_damage_spreading",
    "plot_parameter_sweep",
    "plot_phase_space",
    "plot_time_series",
    "render_grid_image",
    "render_trajectory_gif",
]

