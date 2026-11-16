"""Interactive demonstration script for Cellular Automata experiments."""

import argparse
import time

import numpy as np
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

from complexity_lab.simulations.cellular_automaton import (
    PATTERNS,
    BoundaryCondition,
    CellularAutomatonSimulation,
    InitMode,
    NeighborhoodType,
)

console = Console()


def render_grid(
    grid: np.ndarray,
    alive_char: str = "█",
    dead_char: str = " ",
) -> str:
    """Convert binary 2D numpy array into a visual text block."""
    lines = []
    for row in grid:
        lines.append("".join(alive_char if c else dead_char for c in row))
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Complex Systems Lab - Cellular Automaton Demo")
    parser.add_argument(
        "--pattern",
        type=str,
        default="glider",
        choices=list(PATTERNS.keys()),
        help=f"Pattern to run: {', '.join(sorted(PATTERNS.keys()))}",
    )
    parser.add_argument("--random", action="store_true", help="Use random grid initialization")
    parser.add_argument("--density", type=float, default=0.25, help="Density (0.0 to 1.0)")
    parser.add_argument(
        "--rule",
        type=str,
        default="B3/S23",
        help="Rule string (e.g. B3/S23, B36/S23)",
    )
    parser.add_argument(
        "--boundary",
        type=str,
        default="periodic",
        choices=["periodic", "fixed", "reflective"],
    )
    parser.add_argument("--rows", type=int, default=24, help="Grid rows")
    parser.add_argument("--cols", type=int, default=50, help="Grid columns")
    parser.add_argument("--steps", type=int, default=50, help="Simulation steps")
    parser.add_argument("--delay", type=float, default=0.08, help="Delay per step in seconds")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    sim = CellularAutomatonSimulation()

    if args.random:
        init_mode = InitMode.RANDOM
        pattern = None
        mode_desc = f"random (density={args.density})"
    else:
        init_mode = InitMode.PATTERN
        pattern = args.pattern
        mode_desc = f"pattern: {pattern}"

    sim.initialize(
        seed=args.seed,
        parameters={
            "steps": args.steps,
            "grid_size": (args.rows, args.cols),
            "rule": args.rule,
            "boundary": BoundaryCondition(args.boundary),
            "neighborhood": NeighborhoodType.MOORE,
            "init_mode": init_mode,
            "init_density": args.density,
            "init_pattern": pattern,
        },
    )

    console.print(
        Panel.fit(
            f"[bold green]Starting Cellular Automaton[/bold green]\n"
            f"Mode: [cyan]{mode_desc}[/cyan]\n"
            f"Rule: [yellow]{args.rule}[/yellow] | Boundary: [blue]{args.boundary}[/blue] | "
            f"Size: {args.rows}x{args.cols}",
            title="Complex Systems Lab",
            border_style="green",
        )
    )

    with Live(console=console, screen=False, refresh_per_second=20) as live:
        for _ in range(args.steps + 1):
            assert sim.current_state is not None
            grid_text = render_grid(sim.current_state)

            info_table = Table.grid(padding=1)
            info_table.add_column(style="bold")
            info_table.add_column()
            info_table.add_row("Step:", f"{sim.current_step} / {args.steps}")
            info_table.add_row("Alive Cells:", f"{sim.alive_counts[-1]}")
            info_table.add_row("Activity (changed):", f"{sim.activity_history[-1]}")

            panel = Panel(
                f"[green]{grid_text}[/green]\n\n"
                f"Step: [bold]{sim.current_step}/{args.steps}[/bold] | "
                f"Alive: [cyan]{sim.alive_counts[-1]}[/cyan] | "
                f"Changed: [yellow]{sim.activity_history[-1]}[/yellow]",
                title=f"Simulation: {args.rule}",
                border_style="cyan",
            )
            live.update(panel)

            if sim.current_step < args.steps:
                time.sleep(args.delay)
                sim.step()

    console.print("\n[bold green]Simulation Completed![/bold green]")
    console.print(f"Final Alive Count: [bold cyan]{sim.alive_counts[-1]}[/bold cyan]")
    console.print(f"Total History Recorded: [bold]{len(sim.alive_counts)} states[/bold]")


if __name__ == "__main__":
    main()
