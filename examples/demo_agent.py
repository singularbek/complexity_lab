"""Demonstration of the Autonomous Scientific AI Agent in Complexity Lab.

Shows the complete workflow:
1. Formulating a hypothesis based on a complex systems research goal.
2. Generating a controlled experimental plan.
3. Executing deterministic simulations via schema-validated tools.
4. Measuring real macroscopic observables (density, spatial entropy, orbits).
5. Synthesizing evidence into a peer-review-style scientific report.
"""

from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel

from complexity_lab.application.agent import ScientificAgent

console = Console()


def main() -> None:
    goal = "Investigate the impact of boundary conditions on density preservation and entropy"
    output_report = Path("artifacts/research_report_boundaries.md")

    console.print(
        Panel.fit(
            "[bold green]Complex Systems Autonomous Scientific Agent[/bold green]\n"
            f"Research Target: [yellow]{goal}[/yellow]\n"
            f"Report Destination: [cyan]{output_report}[/cyan]",
            title="Complexity Lab",
            border_style="green",
        )
    )

    def on_event(event: str, payload: dict[str, Any]) -> None:
        if event == "plan_formulated":
            hyp = payload.get("hypothesis", "")
            console.print(f"\n[bold green]Hypothesis Formulated:[/bold green]\n{hyp}\n")
        elif event == "executing_experiment":
            exp_name = payload.get("name")
            console.print(f"  [dim]▶ Running simulation: {exp_name}...[/dim]")
        elif event == "experiment_completed":
            f = payload.get("finding")
            if f is not None:
                obs = f.key_metrics
                console.print(
                    f"    ✔ Completed [bold]{f.experiment_name}[/bold]: "
                    f"Mean Density={obs.get('mean_density', 0.0):.4f}, "
                    f"Entropy={obs.get('final_entropy', 0.0):.4f}b, "
                    f"Periodic={obs.get('is_periodic')} (Period={obs.get('period')})"
                )
        elif event == "report_saved":
            path = payload.get("path")
            console.print(f"\n[bold green]Report saved to:[/bold green] [cyan]{path}[/cyan]")

    agent = ScientificAgent(observer_callback=on_event)
    report = agent.investigate(research_goal=goal, output_path=output_report)

    console.print("\n" + "=" * 60)
    console.print(Panel(report.synthesis, title="Theoretical Synthesis", border_style="cyan"))
    console.print(
        Panel(
            f"[bold]{report.conclusion}[/bold]",
            title="Scientific Conclusion",
            border_style="yellow",
        )
    )

    if report.suggested_next_steps:
        console.print("\n[bold cyan]Follow-up Investigations:[/bold cyan]")
        for step in report.suggested_next_steps:
            console.print(f"  • {step}")


if __name__ == "__main__":
    main()
