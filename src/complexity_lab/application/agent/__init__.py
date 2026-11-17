"""Autonomous Scientific Agent package for complex systems research."""

from complexity_lab.application.agent.campaign import (
    AutonomousResearchCampaign,
    CampaignRound,
    ResearchCampaignReport,
)
from complexity_lab.application.agent.models import (
    ExperimentFinding,
    ExperimentPlan,
    Hypothesis,
    PlannedExperiment,
    ScientificReport,
)
from complexity_lab.application.agent.orchestrator import ScientificAgent
from complexity_lab.application.agent.providers import (
    ExternalAPIProvider,
    LLMProvider,
)
from complexity_lab.application.agent.tools import (
    BaseTool,
    CompareExperimentsTool,
    DamageSpreadingTool,
    InspectSimulationTool,
    ListSimulationsTool,
    ParameterSweepTool,
    PlotDynamicsTool,
    PlotSweepTool,
    RenderAnimationTool,
    RunExperimentTool,
    ToolRegistry,
)

__all__ = [
    "AutonomousResearchCampaign",
    "BaseTool",
    "CampaignRound",
    "CompareExperimentsTool",
    "DamageSpreadingTool",
    "ExperimentFinding",
    "ExperimentPlan",
    "ExternalAPIProvider",
    "Hypothesis",
    "InspectSimulationTool",
    "LLMProvider",
    "ListSimulationsTool",
    "ParameterSweepTool",
    "PlannedExperiment",
    "PlotDynamicsTool",
    "PlotSweepTool",
    "RenderAnimationTool",
    "ResearchCampaignReport",
    "RunExperimentTool",
    "ScientificAgent",
    "ScientificReport",
    "ToolRegistry",
]

