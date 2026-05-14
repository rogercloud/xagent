"""
Tool Configuration Management

Provides abstract and concrete configuration classes for tool creation.
This allows different contexts (web, standalone) to provide configuration
to the ToolFactory in a unified way.
"""

from abc import ABC, abstractmethod
from typing import Any

from ..... import config as _root_config


class BaseToolConfig(ABC):
    """Abstract base class for tool configuration."""

    @abstractmethod
    def get_workspace_config(self) -> dict[str, Any] | None:
        """Get workspace configuration."""
        pass

    @abstractmethod
    def get_vision_model(self) -> Any | None:
        """Get vision model."""
        pass

    @abstractmethod
    def get_image_models(self) -> dict[str, Any]:
        """Get image models."""
        pass

    @abstractmethod
    def get_asr_models(self) -> dict[str, Any]:
        """Get ASR (speech-to-text) models."""
        pass

    @abstractmethod
    def get_tts_models(self) -> dict[str, Any]:
        """Get TTS (text-to-speech) models."""
        pass

    @abstractmethod
    async def get_mcp_server_configs(self) -> list[dict[str, Any]]:
        """Get MCP server configurations."""
        pass

    @abstractmethod
    def get_file_tools_enabled(self) -> bool:
        """Whether to include file tools."""
        pass

    @abstractmethod
    def get_basic_tools_enabled(self) -> bool:
        """Whether to include basic tools."""
        pass

    @abstractmethod
    def get_embedding_model(self) -> str | None:
        """Get embedding model ID."""
        pass

    @abstractmethod
    def get_browser_tools_enabled(self) -> bool:
        """Whether to include browser automation tools."""
        pass

    @abstractmethod
    def get_task_id(self) -> str | None:
        """Get task ID for session tracking."""
        pass

    @abstractmethod
    def get_allowed_collections(self) -> list[str] | None:
        """Get allowed knowledge base collections. None means all collections are allowed."""
        pass

    @abstractmethod
    def get_allowed_skills(self) -> list[str] | None:
        """Get allowed skill names. None means all skills are allowed."""
        pass

    @abstractmethod
    def get_user_id(self) -> int | None:
        """Get current user ID for multi-tenancy."""
        pass

    @abstractmethod
    def is_admin(self) -> bool:
        """Whether current user is admin."""
        pass

    @abstractmethod
    def get_enable_agent_tools(self) -> bool:
        """Whether to include published agents as tools."""
        pass

    def get_delegate_agent_ids(self) -> list[int] | None:
        """Get explicitly selected delegable agent IDs. None means default behavior."""
        return None

    @abstractmethod
    def get_image_generate_model(self) -> Any | None:
        """Get default image generation model."""
        pass

    @abstractmethod
    def get_custom_api_configs(self) -> list[dict[str, Any]]:
        """Get custom API configurations."""
        pass

    @abstractmethod
    def get_image_edit_model(self) -> Any | None:
        """Get default image editing model."""
        pass

    @abstractmethod
    def get_sandbox(self) -> Any | None:
        """Get sandbox instance for sandboxed executors. Returns None if not available."""
        pass

    def get_tool_credential(self, tool_name: str, field_name: str) -> str | None:
        return None

    def get_sql_connections(self) -> dict[str, str]:
        return {}

    @abstractmethod
    def get_db(self) -> Any | None:
        """Get database session. Returns None for standalone usage."""
        pass

    @abstractmethod
    def get_asr_model(self) -> Any | None:
        """Get default ASR (speech-to-text) model."""
        pass

    @abstractmethod
    def get_tts_model(self) -> Any | None:
        """Get default TTS (text-to-speech) model."""
        pass

    @abstractmethod
    def get_llm(self) -> Any | None:
        """Get default LLM for general tasks."""
        pass

    def get_allowed_tools(self) -> list[str] | None:
        return None

    def get_allowed_agent_ids(self) -> list[int] | None:
        return None

    def get_agent_tool_overrides(self) -> dict[int, dict[str, Any]]:
        return {}

    def get_allow_cross_user_agent_ids(self) -> bool:
        return False

    def get_parent_task_id(self) -> int | None:
        return None

    def get_parent_tracer(self) -> Any | None:
        return None

    def get_max_output_length(self) -> int:
        """Get maximum output length in characters.

        Reads from XAGENT_TOOL_MAX_OUTPUT_LENGTH env var if set.
        See :mod:`xagent.config` for details.
        """
        return _root_config.get_tool_max_output_length()

    def get_max_field_count(self) -> int:
        """Get maximum number of fields/items in dict/list for output filtering.

        Reads from XAGENT_TOOL_MAX_FIELD_COUNT env var if set.
        See :mod:`xagent.config` for details.
        """
        return _root_config.get_tool_max_field_count()

    def get_max_recursion_depth(self) -> int:
        """Get maximum recursion depth for output filtering.

        Reads from XAGENT_TOOL_MAX_RECURSION_DEPTH env var if set.
        See :mod:`xagent.config` for details.
        """
        return _root_config.get_tool_max_recursion_depth()


class ToolConfig(BaseToolConfig):
    """Tool configuration that uses provided config dict for standalone usage."""

    def __init__(self, config_dict: dict[str, Any]):
        # Extract configurations from dict
        workspace_config = config_dict.get("workspace")
        config_dict.get("vision_model")  # Unused in base config
        config_dict.get("image_models", [])  # Unused in base config
        config_dict.get("asr_models", [])  # Unused in base config
        config_dict.get("tts_models", [])  # Unused in base config
        mcp_server_configs = config_dict.get("mcp_servers", [])
        file_tools_enabled = config_dict.get("file_tools_enabled", True)
        basic_tools_enabled = config_dict.get("basic_tools_enabled", True)
        embedding_model = config_dict.get("embedding_model")
        browser_tools_enabled = config_dict.get("browser_tools_enabled", True)
        task_id = config_dict.get("task_id")
        allowed_collections = config_dict.get("allowed_collections")
        allowed_skills = config_dict.get("allowed_skills")
        allowed_tools = config_dict.get("allowed_tools")
        allowed_agent_ids = config_dict.get("allowed_agent_ids")
        user_id = config_dict.get("user_id")
        is_admin = config_dict.get("is_admin", False)
        tool_credentials = config_dict.get("tool_credentials", {})
        agent_tool_overrides = config_dict.get("agent_tool_overrides", {})
        enable_global_agent_tools = config_dict.get("enable_global_agent_tools", True)
        allow_cross_user_agent_ids = config_dict.get(
            "allow_cross_user_agent_ids", False
        )
        parent_task_id = config_dict.get("parent_task_id")
        parent_tracer = config_dict.get("parent_tracer")
        agent_call_stack = config_dict.get("agent_call_stack")

        # Output limit configuration (uses environment variable as default)
        # Store custom values if provided, otherwise use None to fall back to base class defaults
        self._custom_max_output_length: int | None = None
        try:
            self._custom_max_output_length = int(
                config_dict.get("max_output_length")  # type: ignore[arg-type]
            )
        except (TypeError, ValueError):
            pass
        self._custom_max_field_count: int | None = None
        try:
            self._custom_max_field_count = int(
                config_dict.get("max_field_count")  # type: ignore[arg-type]
            )
        except (TypeError, ValueError):
            pass
        self._custom_max_recursion_depth: int | None = None
        try:
            self._custom_max_recursion_depth = int(
                config_dict.get("max_recursion_depth")  # type: ignore[arg-type]
            )
        except (TypeError, ValueError):
            pass

        self.workspace_config: dict[str, Any] | None = workspace_config
        self.vision_model: Any | None = (
            None  # Standalone usage typically doesn't have web context
        )
        self.image_models: dict[
            str, Any
        ] = {}  # Standalone usage typically doesn't have web context
        self.asr_models: dict[
            str, Any
        ] = {}  # Standalone usage typically doesn't have web context
        self.tts_models: dict[
            str, Any
        ] = {}  # Standalone usage typically doesn't have web context
        self.mcp_server_configs: list[dict[str, Any]] = mcp_server_configs
        self.file_tools_enabled: bool = bool(file_tools_enabled)
        self.basic_tools_enabled: bool = bool(basic_tools_enabled)
        self.embedding_model: str | None = embedding_model
        self.browser_tools_enabled: bool = bool(browser_tools_enabled)
        self.task_id: str | None = task_id
        self.allowed_collections: list[str] | None = allowed_collections
        self.allowed_skills: list[str] | None = allowed_skills
        self.allowed_tools: list[str] | None = allowed_tools
        self.allowed_agent_ids: list[int] | None = (
            [value for value in allowed_agent_ids if isinstance(value, int)]
            if isinstance(allowed_agent_ids, list)
            else None
        )
        self.user_id: int | None = user_id
        self.is_admin_value: bool = bool(is_admin)
        self.tool_credentials: dict[str, dict[str, str]] = tool_credentials
        self.agent_tool_overrides: dict[int, dict[str, Any]] = (
            {
                int(key): value
                for key, value in agent_tool_overrides.items()
                if isinstance(key, int) and isinstance(value, dict)
            }
            if isinstance(agent_tool_overrides, dict)
            else {}
        )
        self.enable_global_agent_tools: bool = bool(enable_global_agent_tools)
        self.allow_cross_user_agent_ids: bool = bool(allow_cross_user_agent_ids)
        self.parent_task_id: int | None = (
            parent_task_id if isinstance(parent_task_id, int) else None
        )
        self.parent_tracer: Any | None = parent_tracer
        self.agent_call_stack: list[int] = (
            [int(agent_id) for agent_id in agent_call_stack]
            if isinstance(agent_call_stack, list)
            else []
        )

    def get_workspace_config(self) -> dict[str, Any] | None:
        return self.workspace_config

    def get_vision_model(self) -> Any | None:
        return self.vision_model

    def get_image_models(self) -> dict[str, Any]:
        return self.image_models

    def get_asr_models(self) -> dict[str, Any]:
        return self.asr_models

    def get_tts_models(self) -> dict[str, Any]:
        return self.tts_models

    async def get_mcp_server_configs(self) -> list[dict[str, Any]]:
        return self.mcp_server_configs

    def get_file_tools_enabled(self) -> bool:
        return self.file_tools_enabled

    def get_basic_tools_enabled(self) -> bool:
        return self.basic_tools_enabled

    def get_embedding_model(self) -> str | None:
        return self.embedding_model

    def get_browser_tools_enabled(self) -> bool:
        return self.browser_tools_enabled

    def get_task_id(self) -> str | None:
        return self.task_id

    def get_allowed_collections(self) -> list[str] | None:
        return self.allowed_collections

    def get_allowed_skills(self) -> list[str] | None:
        return self.allowed_skills

    def get_user_id(self) -> int | None:
        return self.user_id

    def is_admin(self) -> bool:
        return self.is_admin_value

    def get_enable_agent_tools(self) -> bool:
        return self.enable_global_agent_tools

    def get_image_generate_model(self) -> Any | None:
        return None  # Standalone config doesn't have web context

    def get_custom_api_configs(self) -> list[dict[str, Any]]:
        return []  # Standalone config doesn't have web context for custom APIs by default

    def get_image_edit_model(self) -> Any | None:
        return None  # Standalone config doesn't have web context

    def get_asr_model(self) -> Any | None:
        return None  # Standalone config doesn't have web context

    def get_tts_model(self) -> Any | None:
        return None  # Standalone config doesn't have web context

    def get_llm(self) -> Any | None:
        return None  # Standalone config doesn't have web context

    def get_allowed_tools(self) -> list[str] | None:
        return self.allowed_tools

    def get_allowed_agent_ids(self) -> list[int] | None:
        return self.allowed_agent_ids

    def get_agent_tool_overrides(self) -> dict[int, dict[str, Any]]:
        return self.agent_tool_overrides

    def get_allow_cross_user_agent_ids(self) -> bool:
        return self.allow_cross_user_agent_ids

    def get_parent_task_id(self) -> int | None:
        return self.parent_task_id

    def get_parent_tracer(self) -> Any | None:
        return self.parent_tracer

    def get_agent_call_stack(self) -> list[int]:
        return self.agent_call_stack

    def get_sandbox(self) -> Any | None:
        return None  # Standalone config doesn't have sandbox

    def get_tool_credential(self, tool_name: str, field_name: str) -> str | None:
        tool_data = self.tool_credentials.get(tool_name)
        if not isinstance(tool_data, dict):
            return None
        value = tool_data.get(field_name)
        return value if isinstance(value, str) and value else None

    def get_sql_connections(self) -> dict[str, str]:
        return {}

    def get_max_output_length(self) -> int:
        if self._custom_max_output_length is not None:
            return self._custom_max_output_length
        return super().get_max_output_length()

    def get_max_field_count(self) -> int:
        if self._custom_max_field_count is not None:
            return self._custom_max_field_count
        return super().get_max_field_count()

    def get_max_recursion_depth(self) -> int:
        if self._custom_max_recursion_depth is not None:
            return self._custom_max_recursion_depth
        return super().get_max_recursion_depth()

    def get_db(self) -> Any | None:
        """ToolConfig (standalone) does not have database access."""
        return None
