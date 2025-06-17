from dataclasses import dataclass, fields, field
from typing import Any, Dict, Literal, Optional
from langchain_core.runnables import RunnableConfig
import os
from enum import Enum

class SearchAPI(Enum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    TAVILY = "tavily"
    DUCKDUCKGO = "duckduckgo"

class ModelsWithWebSearch(Enum):
    ANTHROPIC_CLAUDE_3_7_SONNET_LATEST = "anthropic:claude-3-7-sonnet-latest"
    ANTHROPIC_CLAUDE_4_0_SONNET_LATEST = "anthropic:claude-sonnet-4-20250514"
    ANTHROPIC_CLAUDE_4_0_OPUS_LATEST = "anthropic:claude-opus-4-20250514"
    ANTHROPIC_CLAUDE_3_5_HAIKU_LATEST = "anthropic:claude-3-5-haiku-latest"
    ANTHROPIC_CLAUDE_3_5_SONNET_LATEST = "anthropic:claude-3-5-sonnet-latest"
    OPENAI_GPT_4_1 = "openai:gpt-4.1"
    OPENAI_GPT_4_1_MINI = "openai:gpt-4.1-mini"

MODELS_WITH_WEB_SEARCH = {
    SearchAPI.ANTHROPIC: {
        ModelsWithWebSearch.ANTHROPIC_CLAUDE_3_7_SONNET_LATEST,
        ModelsWithWebSearch.ANTHROPIC_CLAUDE_4_0_SONNET_LATEST,
        ModelsWithWebSearch.ANTHROPIC_CLAUDE_4_0_OPUS_LATEST,
        ModelsWithWebSearch.ANTHROPIC_CLAUDE_3_5_HAIKU_LATEST,
        ModelsWithWebSearch.ANTHROPIC_CLAUDE_3_5_SONNET_LATEST,
    },
    SearchAPI.OPENAI: {
        ModelsWithWebSearch.OPENAI_GPT_4_1,
        ModelsWithWebSearch.OPENAI_GPT_4_1_MINI,
    }
}

@dataclass(kw_only=True)
class WorkflowConfiguration:
    # General Configuration
    max_structured_output_retries: int = 3
    outline_user_approval: bool = False
    # Research Configuration
    search_api: SearchAPI = SearchAPI.TAVILY
    search_api_config: Optional[Dict[str, Any]] = None
    process_search_results: Literal["summarize", "split_and_rerank"] | None = "summarize"
    number_of_queries: int = 4 # Number of search queries to generate per iteration
    max_search_depth: int = 4 # Maximum number of reflection + search iterations
    # Model Configuration
    summarization_model: str = "anthropic:claude-3-5-haiku-latest"
    summarization_model_max_tokens: int = 10000
    research_model: str = "anthropic:claude-3-7-sonnet-latest"
    research_model_max_tokens: int = 10000
    reflection_model: str = "anthropic:claude-3-7-sonnet-latest"
    reflection_model_max_tokens: int = 10000
    outliner_model: str = "openai:gpt-4.1"
    outliner_model_max_tokens: int = 10000
    final_report_model: str = "anthropic:claude-3-7-sonnet-latest"
    final_report_model_max_tokens: int = 10000
    # MCP server configuration
    mcp_server_config: Optional[Dict[str, Any]] = None
    mcp_prompt: Optional[str] = None
    mcp_tools_to_include: Optional[list[str]] = None

    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> "WorkflowConfiguration":
        """Create a WorkflowConfiguration instance from a RunnableConfig."""
        configurable = (
            config["configurable"] if config and "configurable" in config else {}
        )
        values: dict[str, Any] = {
            f.name: os.environ.get(f.name.upper(), configurable.get(f.name))
            for f in fields(cls)
            if f.init
        }
        return cls(**{k: v for k, v in values.items() if v})