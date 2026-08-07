"""
Custom exceptions used throughout the GraphAgent project.

Having project-specific exceptions makes debugging easier and
allows us to catch only the errors we care about.
"""


class GraphAgentException(Exception):
    """
    Base exception for all GraphAgent errors.
    """

    def __init__(self, message: str = "GraphAgent Error"):
        super().__init__(message)

class ConfigError(GraphAgentException):
    """Raised when there is an issue with configuration files."""
    pass


class DatasetError(GraphAgentException):
    """Raised when dataset loading or processing fails."""
    pass


class GraphConstructionError(GraphAgentException):
    """Raised when graph creation fails."""
    pass

class AgentError(GraphAgentException):
    """Raised when an agent execution fails."""
    pass


class WorkflowError(GraphAgentException):
    """Raised when workflow execution fails."""
    pass

class ModelError(GraphAgentException):
    """Raised when model training or inference fails."""
    pass


class CheckpointError(GraphAgentException):
    """Raised when model checkpoint loading/saving fails."""
    pass

class APIError(GraphAgentException):
    """Raised for API-related errors."""
    pass

class ValidationError(GraphAgentException):
    """Raised when input validation fails."""
    pass