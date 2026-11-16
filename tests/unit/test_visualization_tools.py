"""Unit tests for the visualization agent tools."""

from pathlib import Path

import pytest

from complexity_lab.application.agent.tools import (
    PlotDynamicsTool,
    PlotSweepTool,
    RenderAnimationTool,
    ToolRegistry,
)


@pytest.mark.unit
def test_render_animation_tool(tmp_path: Path) -> None:
    tool = RenderAnimationTool()
    out_gif = tmp_path / "glider.gif"
    res = tool.execute(
        rule="B3/S23",
        pattern="glider",
        steps=10,
        grid_rows=16,
        grid_cols=16,
        output_path=str(out_gif),
    )
    assert res["status"] == "COMPLETED"
    assert res["artifact_type"] == "animation"
    assert out_gif.exists()
    assert out_gif.stat().st_size > 100


@pytest.mark.unit
def test_plot_dynamics_tool(tmp_path: Path) -> None:
    tool = PlotDynamicsTool()
    out_png = tmp_path / "dynamics.png"
    res = tool.execute(
        rule="B3/S23",
        density=0.25,
        steps=15,
        grid_rows=16,
        grid_cols=16,
        output_path=str(out_png),
    )
    assert res["status"] == "COMPLETED"
    assert res["artifact_type"] == "figure"
    assert out_png.exists()
    assert "observables" in res


@pytest.mark.unit
def test_plot_sweep_tool(tmp_path: Path) -> None:
    tool = PlotSweepTool()
    out_png = tmp_path / "sweep.png"
    res = tool.execute(
        parameter_name="init_density",
        values=[0.1, 0.3],
        rule="B3/S23",
        steps=10,
        grid_rows=16,
        grid_cols=16,
        output_path=str(out_png),
    )
    assert res["status"] == "COMPLETED"
    assert res["artifact_type"] == "figure"
    assert out_png.exists()
    assert res["values_evaluated"] == 2


@pytest.mark.unit
def test_extended_tool_registry() -> None:
    registry = ToolRegistry()
    tools = registry.list_tools()
    tool_names = [t.name for t in tools]
    assert "render_animation" in tool_names
    assert "plot_dynamics" in tool_names
    assert "plot_sweep" in tool_names
    assert len(tools) >= 9

