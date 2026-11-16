"""Rendering engines for grid state snapshots and animated simulation trajectories."""

from pathlib import Path

import matplotlib
import numpy as np
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def render_grid_image(
    grid: np.ndarray,
    output_path: Path | str,
    title: str | None = None,
    scale: int = 8,
    alive_color: tuple[int, int, int] = (30, 144, 255),  # Dodger blue
    dead_color: tuple[int, int, int] = (245, 245, 245),   # Off-white
) -> Path:
    """Render a 2D binary grid state to an image file.

    Args:
        grid: 2D numpy array of cell states.
        output_path: Destination file path (e.g. PNG).
        title: Optional title rendered above the grid.
        scale: Pixel magnification factor per grid cell.
        alive_color: RGB tuple for active (1) cells.
        dead_color: RGB tuple for inactive (0) cells.

    Returns:
        Path to the saved image file.
    """
    dest = Path(output_path)
    dest.parent.mkdir(parents=True, exist_ok=True)

    if title:
        # Use matplotlib when an explicit title/figure layout is desired
        fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
        ax.imshow(grid, cmap="Blues", interpolation="nearest", vmin=0, vmax=1)
        ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.tight_layout()
        fig.savefig(dest, bbox_inches="tight", dpi=150)
        plt.close(fig)
        return dest

    h, w = grid.shape
    img_array = np.zeros((h, w, 3), dtype=np.uint8)
    img_array[grid == 1] = alive_color
    img_array[grid == 0] = dead_color

    img = Image.fromarray(img_array, mode="RGB")
    if scale > 1:
        img = img.resize((w * scale, h * scale), resample=Image.Resampling.NEAREST)

    img.save(dest)
    return dest


def render_trajectory_gif(
    trajectory: list[np.ndarray],
    output_path: Path | str,
    fps: int = 10,
    scale: int = 8,
    alive_color: tuple[int, int, int] = (41, 128, 185),  # Blue
    dead_color: tuple[int, int, int] = (250, 250, 250),   # Light grey
) -> Path:
    """Render a sequence of 2D grid states into an animated GIF.

    Args:
        trajectory: List of 2D numpy binary grid states over time.
        output_path: Destination GIF file path.
        fps: Frames per second for playback.
        scale: Pixel magnification factor per grid cell.
        alive_color: RGB tuple for active cells.
        dead_color: RGB tuple for inactive cells.

    Returns:
        Path to the saved animated GIF.
    """
    if not trajectory:
        raise ValueError("Cannot render an empty trajectory.")

    dest = Path(output_path)
    dest.parent.mkdir(parents=True, exist_ok=True)

    frames: list[Image.Image] = []
    duration_ms = max(int(1000 / fps), 20)

    for state in trajectory:
        h, w = state.shape
        img_array = np.zeros((h, w, 3), dtype=np.uint8)
        img_array[state == 1] = alive_color
        img_array[state == 0] = dead_color

        img = Image.fromarray(img_array, mode="RGB")
        if scale > 1:
            img = img.resize((w * scale, h * scale), resample=Image.Resampling.NEAREST)
        frames.append(img)

    frames[0].save(
        dest,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
    )
    return dest
