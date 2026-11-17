"""Command Line Interface entry point for the Complex Systems Laboratory."""

import os
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from complexity_lab.application.agent import (
    AutonomousResearchCampaign,
    ExternalAPIProvider,
    ScientificAgent,
)
from complexity_lab.application.experiment_service import ExperimentService
from complexity_lab.domain.value_objects import SimulationType

app = typer.Typer(
    name="scientific-lab",
    help="Complex Systems Laboratory - Scientific Computing & Autonomous AI Agent Engine",
    add_completion=False,
)
console = Console()


def _create_provider() -> ExternalAPIProvider:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise typer.BadParameter("Set the OPENAI_API_KEY environment variable to use the agent.")
    return ExternalAPIProvider(api_key=api_key)


@app.command()
def info() -> None:
    """Display environment, version, and component status."""
    console.print(
        Panel.fit(
            "[bold green]Complex Systems Laboratory[/bold green]\n"
            "[dim]Version 1 — Scientific Computing Engine[/dim]\n"
            "[dim]Version 2 — Autonomous Scientific AI Agent[/dim]\n\n"
            "Engines: [bold cyan]Cellular Automaton (Conway, HighLife, Life-like)[/bold cyan]\n"
            "Analysis: [bold cyan]Spatial Entropy, Density, Periodicity, Activity[/bold cyan]\n"
            "Agent: [bold cyan]Hypothesis-Driven Deterministic Tool Orchestrator[/bold cyan]",
            title="System Info",
            border_style="green",
        )
    )


@app.command()
def version() -> None:
    """Print the system version."""
    console.print("0.2.0")


@app.command()
def simulate(
    rule: str = typer.Option("B3/S23", "--rule", "-r", help="CA rule string"),
    steps: int = typer.Option(50, "--steps", "-s", help="Number of timesteps"),
    rows: int = typer.Option(32, "--rows", help="Grid height"),
    cols: int = typer.Option(32, "--cols", help="Grid width"),
    density: float = typer.Option(0.25, "--density", "-d", help="Random density"),
    seed: int = typer.Option(42, "--seed", help="Random seed"),
    boundary: str = typer.Option("periodic", "--boundary", "-b", help="Boundary condition"),
    gif: Annotated[
        Path | None,
        typer.Option("--gif", help="Optional path to save animated GIF artifact"),
    ] = None,
    plot: Annotated[
        Path | None,
        typer.Option("--plot", help="Optional path to save dynamics plot figure"),
    ] = None,
) -> None:
    """Run a deterministic simulation and print computed observables."""
    service = ExperimentService()
    console.print(
        f"[bold green]Running simulation:[/bold green] Rule={rule}, Size={rows}x{cols}, Seed={seed}"
    )

    art_dir = gif.parent if gif else (plot.parent if plot else None)

    exp = service.run_experiment(
        simulation_type=SimulationType.CELLULAR_AUTOMATON,
        parameters={
            "rule": rule,
            "boundary": boundary,
            "init_mode": "random",
            "init_density": density,
        },
        seed=seed,
        steps=steps,
        grid_size=(rows, cols),
        artifacts_dir=art_dir,
        render_gif=gif is not None,
        render_plot=plot is not None,
    )

    if exp.status != "COMPLETED" or exp.result is None:
        console.print(f"[bold red]Experiment Failed:[/bold red] {exp.error_message}")
        raise typer.Exit(code=1)

    res = exp.result
    obs = res.observables

    table = Table(title="Simulation Observables", border_style="cyan")
    table.add_column("Metric", style="bold")
    table.add_column("Value", style="yellow")

    table.add_row("Experiment ID", exp.experiment_id)
    table.add_row("Status", exp.status.value)
    table.add_row("Execution Time", f"{res.runtime_seconds:.4f} s")
    table.add_row("Initial Alive Cells", str(obs.get("initial_alive")))
    table.add_row("Final Alive Cells", str(obs.get("final_alive")))
    table.add_row("Mean Density", f"{obs.get('mean_density', 0.0):.4f}")
    table.add_row("Final Shannon Entropy", f"{obs.get('final_entropy', 0.0):.4f} bits")
    table.add_row("Periodic Orbit Detected", str(obs.get("is_periodic")))
    table.add_row("Period", str(obs.get("period")))
    table.add_row("Mean Activity", f"{obs.get('mean_activity', 0.0):.2f} cells/step")
    table.add_row("Extinct", str(obs.get("is_extinct")))

    console.print(table)

    if exp.artifacts:
        console.print("\n[bold cyan]Generated Visual Artifacts:[/bold cyan]")
        for art in exp.artifacts:
            console.print(f"  • {art.name} ({art.artifact_type.value}): [green]{art.path}[/green]")


@app.command()
def agent(
    goal: Annotated[
        str,
        typer.Option(
            "--goal",
            "-g",
            help="Research goal for the autonomous scientific agent",
        ),
    ] = "Compare Conway Life B3/S23 vs HighLife B36/S23 activity and density",
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="Optional path to save Markdown scientific report",
        ),
    ] = None,
) -> None:
    """Launch the Autonomous Scientific Agent to investigate a research goal."""
    console.print(
        Panel(
            f"[bold cyan]Scientific Agent Activated[/bold cyan]\n"
            f"Research Objective: [yellow]{goal}[/yellow]",
            title="Complex Systems Lab Agent",
            border_style="cyan",
        )
    )

    def on_event(event: str, payload: dict[str, Any]) -> None:
        if event == "plan_formulated":
            hyp = payload.get("hypothesis", "")
            console.print(f"\n[bold green]Hypothesis Formulated:[/bold green]\n{hyp}\n")
        elif event == "executing_experiment":
            exp_name = payload.get("name")
            console.print(f"[dim]Executing deterministic experiment: {exp_name}...[/dim]")
        elif event == "experiment_completed":
            f = payload.get("finding")
            if f is not None:
                obs = f.key_metrics
                console.print(
                    f"  -> [bold]{f.experiment_name}[/bold]: "
                    f"Density={obs.get('mean_density', 0.0):.4f}, "
                    f"Entropy={obs.get('final_entropy', 0.0):.4f}b, "
                    f"Periodic={obs.get('is_periodic')} (T={obs.get('period')})"
                )
        elif event == "report_saved":
            path = payload.get("path")
            console.print(f"[bold green]Report saved to:[/bold green] [cyan]{path}[/cyan]")

    researcher = ScientificAgent(provider=_create_provider(), observer_callback=on_event)
    report = researcher.investigate(research_goal=goal, output_path=output)

    console.print("\n" + "=" * 60)
    console.print(Panel(report.synthesis, title="Synthesis", border_style="green"))
    console.print(
        Panel(
            f"[bold]{report.conclusion}[/bold]",
            title="Scientific Conclusion",
            border_style="yellow",
        )
    )

    if report.suggested_next_steps:
        console.print("[bold cyan]Recommended Follow-up Experiments:[/bold cyan]")
        for step in report.suggested_next_steps:
            console.print(f"  • {step}")


@app.command()
def campaign(
    goal: Annotated[
        str,
        typer.Option(
            "--goal",
            "-g",
            help="High-level research objective for the multi-round campaign",
        ),
    ] = "Investigate phase transitions and perturbation stability across boundary conditions",
    rounds: Annotated[
        int,
        typer.Option(
            "--rounds",
            "-r",
            help="Number of iterative research rounds to execute",
        ),
    ] = 2,
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="Optional file path to persist the full campaign journal",
        ),
    ] = None,
) -> None:
    """Launch an Autonomous Multi-Round Research Campaign with reflective hypothesis refinement."""
    console.print(
        Panel(
            f"[bold cyan]Autonomous Scientific Campaign Initiated[/bold cyan]\n"
            f"Campaign Goal: [yellow]{goal}[/yellow]\n"
            f"Scheduled Rounds: [magenta]{rounds}[/magenta]",
            title="Complex Systems Research Campaign",
            border_style="cyan",
        )
    )

    def on_event(event: str, payload: dict[str, Any]) -> None:
        if event == "round_started":
            r_num = payload.get("round")
            subgoal = payload.get("subgoal")
            console.print(
                f"\n[bold green]═══ Initiating Round {r_num} ═══[/bold green]\nTarget: {subgoal}"
            )
        elif event == "round_completed":
            r_num = payload.get("round")
            console.print(f"[dim]✔ Round {r_num} completed and synthesized.[/dim]")
        elif event == "campaign_report_saved":
            p = payload.get("path")
            console.print(
                f"\n[bold green]Campaign Journal persisted to:[/bold green] [cyan]{p}[/cyan]"
            )

    runner = AutonomousResearchCampaign(provider=_create_provider(), observer_callback=on_event)
    camp_report = runner.execute_campaign(
        campaign_goal=goal,
        max_rounds=rounds,
        output_path=output,
    )

    console.print("\n" + "=" * 60)
    console.print(Panel(camp_report.meta_synthesis, title="Meta-Synthesis", border_style="green"))
    console.print(
        Panel(
            f"[bold]{camp_report.final_conclusion}[/bold]",
            title="Campaign Conclusion",
            border_style="yellow",
        )
    )


if __name__ == "__main__":
    app()
