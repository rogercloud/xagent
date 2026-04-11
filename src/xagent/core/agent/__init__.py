"""
Enhanced Agent system with nested agent support and improved patterns.
"""

from .agent import Agent
from .context import AgentContext

# Exceptions
from .exceptions import (
    AgentConfigurationError,
    AgentException,
    AgentToolError,
    ContextCompactionError,
    ContextError,
    DAGDeadlockError,
    DAGDependencyError,
    DAGError,
    DAGPlanGenerationError,
    DAGStepError,
    LLMError,
    LLMNotAvailableError,
    LLMResponseError,
    MaxIterationsError,
    PatternError,
    PatternExecutionError,
    ReActError,
    ReActParsingError,
    ToolError,
    ToolExecutionError,
    ToolNotFoundError,
    create_execution_error,
)

# Patterns
from .pattern.base import AgentPattern
from .pattern.dag_plan_execute import (
    DAGPlanExecutePattern,
    ExecutionPhase,
    ExecutionPlan,
    PlanStep,
    StepInjection,
    StepStatus,
)

# Import ReAct components
from .pattern.react import ReActPattern, ReActStepType
from .precondition import PreconditionResolver
from .runner import AgentRunner

# Utilities
from .utils.context_builder import ContextBuilder, StepExecutionResult

__all__ = [
    # Core agent components
    "Agent",
    "AgentContext",
    "AgentRunner",
    "PreconditionResolver",
    # Base patterns
    "AgentPattern",
    # ReAct pattern
    "ReActPattern",
    "ReActStepType",
    # DAG Plan Execute pattern
    "DAGPlanExecutePattern",
    "PlanStep",
    "ExecutionPlan",
    "StepStatus",
    "ExecutionPhase",
    "StepInjection",
    # Utilities
    "ContextBuilder",
    "StepExecutionResult",
    # Exception hierarchy
    "AgentException",
    "AgentConfigurationError",
    "LLMError",
    "LLMNotAvailableError",
    "LLMResponseError",
    "ToolError",
    "ToolNotFoundError",
    "ToolExecutionError",
    "PatternError",
    "PatternExecutionError",
    "MaxIterationsError",
    "DAGError",
    "DAGPlanGenerationError",
    "DAGStepError",
    "DAGDependencyError",
    "DAGDeadlockError",
    "ReActError",
    "ReActParsingError",
    "ContextError",
    "ContextCompactionError",
    "AgentToolError",
    "create_execution_error",
]
