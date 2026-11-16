"""Domain-specific exceptions for the Complex Systems Laboratory."""


class LabError(Exception):
    """Base exception for all system errors."""


class DomainValidationError(LabError):
    """Raised when domain constraints or invariants are violated."""


class SimulationError(LabError):
    """Raised when a simulation fails execution or encounters instability."""
