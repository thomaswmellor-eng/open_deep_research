import os
from typing import Any, Dict, Literal, Optional

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnableConfig
from langgraph.constants import TAG_NOSTREAM
from pydantic import BaseModel, Field

DEFAULT_REPORT_STRUCTURE = """Use this structure to create a report on the user-provided topic:

1. Introduction (no research needed)
   - Brief overview of the topic area

2. Main Body Sections:
   - Each section should focus on a sub-topic of the user-provided topic
   
3. Conclusion
   - Aim for 1 structural element (either a list of table) that distills the main body sections 
   - Provide a concise summary of the report"""


MODELS = [
    {
        "label": "Claude 3.7 Sonnet",
        "value": "anthropic:claude-3-7-sonnet-latest",
    },
    {
        "label": "Claude 3.5 Sonnet",
        "value": "anthropic:claude-3-5-sonnet-latest",
    },
    {"label": "GPT 4o", "value": "openai:gpt-4o"},
    {"label": "GPT 4o mini", "value": "openai:gpt-4o-mini"},
    {"label": "GPT 4.1", "value": "openai:gpt-4.1"},
    {"label": "o3", "value": "openai:o3"},
    {"label": "o3 mini", "value": "openai:o3-mini"},
    {"label": "o4", "value": "openai:o4"},
]


SEARCH = [
    {
        "label": "Perplexity",
        "value": "perplexity",
    },
    {
        "label": "Tavily",
        "value": "tavily",
    },
    {
        "label": "Exa",
        "value": "exa",
    },
    {
        "label": "Arxiv",
        "value": "arxiv",
    },
    {
        "label": "Pubmed",
        "value": "pubmed",
    },
    {
        "label": "Linkup",
        "value": "linkup",
    },
    {
        "label": "DuckDuckGo",
        "value": "duckduckgo",
    },
    {
        "label": "Google",
        "value": "googlesearch",
    },
]


class Configuration(BaseModel):
    """The configurable fields for the chatbot."""

    # Common configuration
    report_structure: str = Field(
        default=DEFAULT_REPORT_STRUCTURE,
        json_schema_extra={
            "metadata": {
                "x_lg_ui_config": {
                    "type": "textarea",
                    "placeholder": "Enter a report structure...",
                    "description": "The report structure to use for research",
                    "default": DEFAULT_REPORT_STRUCTURE,
                }
            }
        },
    )

    search_api: str = Field(
        default="tavily",
        json_schema_extra={
            "metadata": {
                "x_lg_ui_config": {
                    "type": "select",
                    "default": "tavily",
                    "description": "The search api to use for research",
                    "options": SEARCH,
                }
            }
        },
    )

    search_api_config: Optional[Dict[str, Any]] = Field(
        default=None,
        json_schema_extra={
            "metadata": {
                "x_lg_ui_config": {
                    "type": "json",
                    "placeholder": "Enter a search api config...",
                    "description": "The search api config to use for research",
                }
            }
        },
    )

    # Graph-specific configuration
    number_of_queries: int = Field(
        default=2,
        json_schema_extra={
            "metadata": {
                "x_lg_ui_config": {
                    "type": "slider",
                    "min": 1,
                    "max": 5,
                    "step": 1,
                    "default": 1,
                    "description": "Number of search queries to generate per iteration",
                }
            }
        },
    )

    max_search_depth: int = Field(
        default=2,
        json_schema_extra={
            "metadata": {
                "x_lg_ui_config": {
                    "type": "slider",
                    "min": 1,
                    "max": 5,
                    "step": 1,
                    "default": 2,
                    "description": "Maximum number of reflection + search iterations",
                }
            }
        },
    )

    converse_model: str = Field(
        default="openai:gpt-4o-mini",
        json_schema_extra={
            "metadata": {
                "x_lg_ui_config": {
                    "type": "select",
                    "default": "openai:gpt-4o-mini",
                    "description": "The model to use for conversation",
                    "options": MODELS,
                }
            }
        },
    )

    converse_model_kwargs: Optional[Dict[str, Any]] = Field(
        default=None,
        json_schema_extra={
            "metadata": {
                "x_lg_ui_config": {
                    "type": "json",
                    "placeholder": "Enter converse model kwargs...",
                    "description": "The converse model kwargs to use for research",
                }
            }
        },
    )

    planner_model: str = Field(
        default="anthropic:claude-3-7-sonnet-latest",
        json_schema_extra={
            "metadata": {
                "x_lg_ui_config": {
                    "type": "select",
                    "default": "anthropic:claude-3-7-sonnet-latest",
                    "description": "The model to use for the planner agent",
                    "options": MODELS,
                }
            }
        },
    )

    planner_model_kwargs: Optional[Dict[str, Any]] = Field(
        default=None,
        json_schema_extra={
            "metadata": {
                "x_lg_ui_config": {
                    "type": "json",
                    "placeholder": "Enter planner model kwargs...",
                    "description": "The planner model kwargs to use for research",
                }
            }
        },
    )

    writer_model: str = Field(
        default="anthropic:claude-3-5-sonnet-latest",
        json_schema_extra={
            "metadata": {
                "x_lg_ui_config": {
                    "type": "select",
                    "default": "anthropic:claude-3-5-sonnet-latest",
                    "description": "The model to use for the writer agent",
                    "options": MODELS,
                }
            }
        },
    )

    writer_model_kwargs: Optional[Dict[str, Any]] = Field(
        default=None,
        json_schema_extra={
            "metadata": {
                "x_lg_ui_config": {
                    "type": "json",
                    "placeholder": "Enter a writer model kwargs...",
                    "description": "The writer model kwargs to use for research",
                }
            }
        },
    )

    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> "Configuration":
        """Create a Configuration instance from a RunnableConfig."""
        configurable = (
            config["configurable"] if config and "configurable" in config else {}
        )
        values: dict[str, Any] = {
            f: os.environ.get(f.upper(), configurable.get(f))
            for f in cls.model_fields.keys()
        }
        return cls(**{k: v for k, v in values.items() if v})

    @classmethod
    def init_chat_model(
        cls,
        model_name: Literal["writer_model", "planner_model", "converse_model"],
        config: Optional[RunnableConfig],
    ) -> BaseChatModel:
        configuration = cls.from_runnable_config(config)
        config_model_name = getattr(configuration, model_name)
        config_model_kwargs = {
            **(
                {
                    "max_tokens": 20_000,
                    "thinking": {"type": "enabled", "budget_tokens": 16_000},
                }
                if model_name == "planner_model"
                and config_model_name.endswith("claude-3-7-sonnet-latest")
                else {}
            ),
            **(getattr(configuration, f"{model_name}_kwargs", None) or {}),
        }

        return init_chat_model(model=config_model_name, **config_model_kwargs)
