"""Core Experiment domain entity managing state transitions and parameters."""

import uuid
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field, model_validator

from complexity_lab.domain.artifacts import Artifact, ExperimentResult
from complexity_lab.domain.exceptions import DomainValidationError
from complexity_lab.domain.value_objects import ExperimentStatus, SimulationType


def _utc_now() -> datetime:
    return datetime.now(UTC)


class ExperimentParameters(BaseModel):
    """Generic base parameter container for simulations."""

    steps: int = Field(default=100, ge=1, description="Number of simulation steps")
    grid_size: tuple[int, int] = Field(
        default=(64, 64),
        description="Dimensions of the simulation space",
    )
    extra_params: dict[str, Any] = Field(default_factory=dict)

    model_config = {"frozen": True}


class Experiment(BaseModel):
    """Domain model capturing the entire state, configuration, and result of an experiment."""

    experiment_id: str = Field(default_factory=lambda: f"exp_{uuid.uuid4().hex[:8]}")
    simulation_type: SimulationType
    parameters: ExperimentParameters
    seed: int = Field(default=42, description="Random seed for deterministic execution")
    status: ExperimentStatus = ExperimentStatus.CREATED

    created_at: datetime = Field(default_factory=_utc_now)
    started_at: datetime | None = None
    completed_at: datetime | None = None

    result: ExperimentResult | None = None
    artifacts: list[Artifact] = Field(default_factory=list)
    error_message: str | None = None

    def start(self) -> None:
        """Transition experiment from CREATED to RUNNING."""
        if self.status != ExperimentStatus.CREATED:
            raise DomainValidationError(
                f"Cannot start experiment in status '{self.status}'. Must be in 'CREATED'."
            )
        self.status = ExperimentStatus.RUNNING
        self.started_at = _utc_now()

    def complete(self, result: ExperimentResult, artifacts: list[Artifact] | None = None) -> None:
        """Transition experiment from RUNNING to COMPLETED."""
        if self.status != ExperimentStatus.RUNNING:
            raise DomainValidationError(
                f"Cannot complete experiment in status '{self.status}'. Must be in 'RUNNING'."
            )
        self.status = ExperimentStatus.COMPLETED
        self.completed_at = _utc_now()
        self.result = result
        if artifacts:
            self.artifacts.extend(artifacts)

    def fail(self, error_reason: str) -> None:
        """Transition experiment from any active status to FAILED."""
        if self.status == ExperimentStatus.COMPLETED:
            raise DomainValidationError("Cannot fail an already COMPLETED experiment.")
        self.status = ExperimentStatus.FAILED
        self.completed_at = _utc_now()
        self.error_message = error_reason

    @model_validator(mode="after")
    def validate_timestamps(self) -> "Experiment":
        """Validate logical ordering of timestamps."""
        if self.started_at and self.started_at < self.created_at:
            raise ValueError("started_at cannot precede created_at.")
        if self.completed_at and self.started_at and self.completed_at < self.started_at:
            raise ValueError("completed_at cannot precede started_at.")
        return self
