"""Data models representing experiment outputs and outputs metadata."""

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from complexity_lab.domain.value_objects import ArtifactType


class Artifact(BaseModel):
    """Represents a persisted output artifact produced during an experiment run."""

    name: str = Field(..., description="Unique descriptive identifier for the artifact")
    artifact_type: ArtifactType
    path: Path = Field(..., description="File path relative or absolute")
    mime_type: str = Field(
        "application/octet-stream",
        description="MIME type for serialization/HTTP",
    )

    model_config = {"frozen": True}


class ExperimentResult(BaseModel):
    """Numerical metrics and summary data generated from an experiment run."""

    observables: dict[str, Any] = Field(
        default_factory=dict,
        description="Key-value metrics derived from simulation run (e.g. entropy, population)",
    )
    time_series: dict[str, list[float]] = Field(
        default_factory=dict,
        description="Time series trajectory data (e.g. density per timestep)",
    )
    runtime_seconds: float = Field(0.0, ge=0.0)

    model_config = {"frozen": True}
