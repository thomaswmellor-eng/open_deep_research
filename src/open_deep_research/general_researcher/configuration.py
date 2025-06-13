from dataclasses import dataclass, fields
from typing import Any, Dict, Literal, Optional
from langchain_core.runnables import RunnableConfig
import os
from enum import Enum

class SearchAPI(Enum):
    ANTHROPIC = "anthropic"
    PERPLEXITY = "perplexity"
    TAVILY = "tavily"
    EXA = "exa"
    ARXIV = "arxiv"
    PUBMED = "pubmed"
    LINKUP = "linkup"
    DUCKDUCKGO = "duckduckgo"
    GOOGLESEARCH = "googlesearch"
    NONE = "none"

@dataclass(kw_only=True)
class WorkflowConfiguration:
    # General Configuration
    max_structured_output_retries: int = 3
    sections_user_approval: bool = False
    one_shot_mode: bool = False
    # Search Configuration
    search_api: SearchAPI = SearchAPI.TAVILY
    search_api_config: Optional[Dict[str, Any]] = None
    process_search_results: Literal["summarize", "split_and_rerank"] | None = "summarize"
    number_of_queries: int = 3 # Number of search queries to generate per iteration
    max_search_depth: int = 3 # Maximum number of reflection + search iterations
    # Model Configuration
    summarization_model_provider: str = "anthropic"
    summarization_model: str = "claude-3-5-haiku-latest"
    summarization_model_kwargs: Optional[Dict[str, Any]] = None
    planner_provider: str = "openai"
    planner_model: str = "gpt-4.1"
    planner_model_kwargs: Optional[Dict[str, Any]] = None
    writer_provider: str = "anthropic"
    writer_model: str = "claude-3-7-sonnet-latest"
    writer_model_kwargs: Optional[Dict[str, Any]] = None
    
    

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