"""Unit tests for the domain layer entities and value objects."""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from complexity_lab.domain.artifacts import Artifact, ExperimentResult
from complexity_lab.domain.exceptions import DomainValidationError
from complexity_lab.domain.experiments import Experiment, ExperimentParameters
from complexity_lab.domain.value_objects import ArtifactType, ExperimentStatus, SimulationType


@pytest.mark.unit
def test_experiment_creation_default_state() -> None:
    exp = Experiment(
        simulation_type=SimulationType.CELLULAR_AUTOMATON,
        parameters=ExperimentParameters(steps=50, grid_size=(32, 32)),
        seed=123,
    )
    assert exp.status == ExperimentStatus.CREATED
    assert exp.created_at is not None
    assert exp.started_at is None
    assert exp.completed_at is None
    assert exp.result is None
    assert exp.artifacts == []
    assert exp.error_message is None
    assert exp.experiment_id.startswith("exp_")


@pytest.mark.unit
def test_experiment_lifecycle_success() -> None:
    exp = Experiment(
        simulation_type=SimulationType.CELLULAR_AUTOMATON,
        parameters=ExperimentParameters(),
    )
    exp.start()
    assert exp.status == ExperimentStatus.RUNNING
    assert exp.started_at is not None

    result = ExperimentResult(
        observables={"final_alive": 42},
        time_series={"density": [0.2, 0.15, 0.1]},
        runtime_seconds=0.45,
    )
    artifact = Artifact(
        name="final_grid",
        artifact_type=ArtifactType.IMAGE,
        path=Path("artifacts/final_grid.png"),
        mime_type="image/png",
    )
    exp.complete(result=result, artifacts=[artifact])

    final_status: ExperimentStatus = exp.status
    assert final_status == ExperimentStatus.COMPLETED
    assert exp.completed_at is not None
    assert exp.result == result
    assert exp.artifacts == [artifact]


@pytest.mark.unit
def test_experiment_lifecycle_failure() -> None:
    exp = Experiment(
        simulation_type=SimulationType.CELLULAR_AUTOMATON,
        parameters=ExperimentParameters(),
    )
    exp.start()
    exp.fail("Simulation diverged")
    assert exp.status == ExperimentStatus.FAILED
    assert exp.completed_at is not None
    assert exp.error_message == "Simulation diverged"


@pytest.mark.unit
def test_invalid_lifecycle_transitions() -> None:
    exp = Experiment(
        simulation_type=SimulationType.CELLULAR_AUTOMATON,
        parameters=ExperimentParameters(),
    )
    result = ExperimentResult(runtime_seconds=1.0)

    # Cannot complete from CREATED
    with pytest.raises(DomainValidationError):
        exp.complete(result=result)

    # Transition to RUNNING
    exp.start()

    # Cannot start again
    with pytest.raises(DomainValidationError):
        exp.start()

    # Complete the experiment
    exp.complete(result=result)

    # Cannot fail already COMPLETED experiment
    with pytest.raises(DomainValidationError):
        exp.fail("Too late")

    # Cannot start COMPLETED experiment
    with pytest.raises(DomainValidationError):
        exp.start()


@pytest.mark.unit
def test_experiment_timestamp_ordering_validation() -> None:
    now = datetime.now(UTC)
    # started_at before created_at
    with pytest.raises(ValueError, match="started_at cannot precede created_at"):
        Experiment(
            simulation_type=SimulationType.CELLULAR_AUTOMATON,
            parameters=ExperimentParameters(),
            created_at=now,
            started_at=now - timedelta(seconds=10),
        )

    # completed_at before started_at
    with pytest.raises(ValueError, match="completed_at cannot precede started_at"):
        Experiment(
            simulation_type=SimulationType.CELLULAR_AUTOMATON,
            parameters=ExperimentParameters(),
            created_at=now,
            started_at=now + timedelta(seconds=5),
            completed_at=now + timedelta(seconds=2),
        )
