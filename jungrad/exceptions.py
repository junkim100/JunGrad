"""Custom exceptions for jungrad."""


class JungradError(Exception):
    """Base exception for all jungrad errors."""

    pass


class AutogradError(JungradError):
    """Raised when autograd operations fail."""

    pass


class ShapeError(JungradError):
    """Raised when tensor shape operations are invalid."""

    pass


class NumericsError(JungradError):
    """Raised when numerical issues are detected (NaN, Inf, etc.)."""

    pass
