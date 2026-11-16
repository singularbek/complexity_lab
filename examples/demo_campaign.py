"""Demonstration of a Multi-Round Autonomous Research Campaign with visual artifacts.

Demonstrates:
1. Multi-round hypothesis formulation and empirical execution.
2. Reflective hypothesis refinement from Round 1 to Round 2.
3. Automated generation of visual figures and dynamics plots.
4. Publication-ready campaign journal synthesis.
"""

from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel

from complexity_lab.application.agent import AutonomousResearchCampaign

console = Console()


def main() -> None:
    goal = "Investigate phase transition and perturbation stability in Conway's Game of Life"
    journal_path = Path("artifacts/campaign_journal.md")

    console.print(
        Panel.fit(
            "[bold green]Autonomous Multi-Round Research Campaign[/bold green]\n"
            f"Campaign Goal: [yellow]{goal}[/yellow]\n"
            f"Journal Output: [cyan]{journal_path}[/cyan]",
            title="Complexity Lab - Phase 3",
            border_style="green",
        )
    )

    def on_event(event: str, payload: dict[str, Any]) -> None:
        if event == "round_started":
            r_num = payload.get("round")
            subgoal = payload.get("subgoal")
            console.print(f"\n[bold cyan]▶ Round {r_num} Launched:[/bold cyan] {subgoal}")
        elif event == "plan_formulated":
            hyp = payload.get("hypothesis", "")
            console.print(f"  [green]Hypothesis:[/green] {hyp}")
        elif event == "round_completed":
            r_num = payload.get("round")
            cr = payload.get("campaign_round")
            refl = cr.reflection[:80] if cr else ""
            console.print(f"  [dim]✔ Round {r_num} synthesized. Reflection: {refl}...[/dim]")
        elif event == "campaign_report_saved":
            p = payload.get("path")
            console.print(
                f"\n[bold green]Complete Campaign Journal saved to:[/bold green] [cyan]{p}[/cyan]"
            )

    campaign = AutonomousResearchCampaign(observer_callback=on_event)
    report = campaign.execute_campaign(
        campaign_goal=goal,
        max_rounds=2,
        output_path=journal_path,
        artifacts_dir="artifacts",
    )

    console.print("\n" + "=" * 60)
    console.print(
        Panel(report.meta_synthesis, title="Campaign Meta-Synthesis", border_style="cyan")
    )
    console.print(
        Panel(
            f"[bold]{report.final_conclusion}[/bold]",
            title="Final Conclusion",
            border_style="yellow",
        )
    )

    if report.visual_artifacts:
        console.print("\n[bold cyan]Generated Visual Artifacts:[/bold cyan]")
        for art in report.visual_artifacts:
            console.print(f"  • {art}")


if __name__ == "__main__":
    main()
