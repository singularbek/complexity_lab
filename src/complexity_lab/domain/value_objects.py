"""Value objects and enumerations for the domain layer."""

from enum import StrEnum


class ExperimentStatus(StrEnum):
    """Lifecycle states of a scientific experiment."""

    CREATED = "CREATED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class SimulationType(StrEnum):
    """Supported simulation classifications."""

    CELLULAR_AUTOMATON = "cellular_automaton"
    ISING_MODEL = "ising_model"
    HOPFIELD_NETWORK = "hopfield_network"


class ArtifactType(StrEnum):
    """Classifications of generated experiment artifacts."""

    IMAGE = "image"
    DATA_CSV = "data_csv"
    DATA_JSON = "data_json"
    LOG = "log"
