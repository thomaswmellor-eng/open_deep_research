from langgraph.graph import MessagesState
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, get_buffer_string
from langchain_core.runnables import RunnableConfig
from langgraph.constants import Send
from langgraph.graph import START, END, StateGraph
from langgraph.types import interrupt, Command
from typing import List, Annotated
from pydantic import BaseModel, Field
import operator

from open_deep_research.utils import (
    format_sections, 
    get_config_value, 
    get_search_params, 
    select_and_execute_search,
    get_today_str
)

from open_deep_research.configuration import SearchAPI
from dataclasses import dataclass, fields
from typing import Optional, Dict, Any, Literal
import os
from open_deep_research.general_researcher.prompts import query_writer_instructions, response_structure_instructions

class SearchQuery(BaseModel):
    search_query: str = Field(None, description="Query for web search.")

class Queries(BaseModel):
    queries: List[SearchQuery] = Field(
        description="List of search queries.",
    )

class Section(BaseModel):
    name: str = Field(
        description="Name for this section of the report.",
    )
    description: str = Field(
        description="Brief overview of the main topics and concepts to be covered in this section.",
    )
    content: str = Field(
        description="The content of the section."
    )

class Outline(BaseModel):
    outline: List[Section] = Field(
        description="List of sections for the report.",
    )

@dataclass(kw_only=True)
class WorkflowConfiguration:
    # Common configuration
    search_api: SearchAPI = SearchAPI.TAVILY
    search_api_config: Optional[Dict[str, Any]] = None
    process_search_results: Literal["summarize", "split_and_rerank"] | None = "summarize"
    summarization_model_provider: str = "anthropic"
    summarization_model: str = "claude-3-5-haiku-latest"
    max_structured_output_retries: int = 3
    # Workflow-specific configuration
    number_of_queries: int = 3 # Number of search queries to generate per iteration
    max_search_depth: int = 2 # Maximum number of reflection + search iterations
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

class GeneralResearcherStateInput(MessagesState):
    """InputState is only 'messages'"""

class GeneralResearcherStateOutput(MessagesState):
    final_report: str

class GeneralResearcherState(MessagesState):
    final_report: str
    notes: Annotated[list[str], operator.add]
    historical_queries: Annotated[list[str], operator.add]
    current_queries: list[str]
    search_attempts: int = 0
    outline: list[Section]


async def generate_queries(state: GeneralResearcherState, config: RunnableConfig) -> Command[Literal["search", "generate_outline"]]:
    messages = state["messages"]
    notes = state.get("notes", [])
    historical_queries = state.get("historical_queries", [])
    configurable = WorkflowConfiguration.from_runnable_config(config)
    writer_model = init_chat_model(
        model=configurable.writer_model,
        model_provider=configurable.writer_provider,
        model_kwargs=configurable.writer_model_kwargs or {},
    ).with_structured_output(Queries).with_retry(stop_after_attempt=configurable.max_structured_output_retries)

    try:
        query_results = await writer_model.ainvoke([HumanMessage(content=query_writer_instructions.format(
            messages=get_buffer_string(messages),
            number_of_queries=configurable.number_of_queries,
            query_history="\n".join([f"{i+1}. {q}" for i, q in enumerate(historical_queries)]),
            context="\n".join(notes),
            today=get_today_str()
        ))])
        query_list = [query.search_query for query in query_results.queries]
        if not query_list:
            return Command(
                goto="generate_outline"
            )
        return Command(
            goto="search",
            update={
                "current_queries": query_list,
            }
        )
    except Exception as e:
        print(f"Error generating queries: {e}")
        return Command(
            goto="generate_queries",
            update={"search_attempts": state.get("search_attempts", 0) + 1}
        )


async def search(state: GeneralResearcherState, config: RunnableConfig) -> Command[Literal["generate_queries", "generate_outline"]]:
    current_queries = state.get("current_queries", [])
    search_attempts = state.get("search_attempts", 0)
    configurable = WorkflowConfiguration.from_runnable_config(config)
    try:
        search_results = await select_and_execute_search(
            get_config_value(configurable.search_api),
            current_queries,
            get_search_params(get_config_value(configurable.search_api), configurable.search_api_config or {}))
        # TODO: Potentially format the search results into a more readable and specific format before adding to notes.
        # If we have run out of search attempts, we go to generate an outline. If not, we go back to the query generator.
        return Command(
            goto= "generate_queries" if search_attempts < configurable.max_search_depth else "generate_outline",
            update={
                "notes": ["From these queries: " + "\n".join(current_queries) + "\n" + "We found this information: " + search_results],
                "search_attempts": search_attempts + 1,
                "historical_queries": [current_queries],
                "current_queries": []
            }
        )
    except Exception as e:
        print(f"Error searching: {e}")
        # If we have run out of search attempts, we go to generate an outline. If not, we try again directly in this node.
        return Command(
            goto= "search" if search_attempts < configurable.max_search_depth else "generate_outline",
            update={"search_attempts": search_attempts + 1}
        )


async def generate_outline(state: GeneralResearcherState, config: RunnableConfig):
    messages = state["messages"]
    notes = state.get("notes", [])
    configurable = WorkflowConfiguration.from_runnable_config(config)
    planner_model = init_chat_model(
        model=configurable.planner_model,
        model_provider=configurable.planner_provider,
        model_kwargs=configurable.planner_model_kwargs or {},
    ).with_structured_output(Outline).with_retry(stop_after_attempt=configurable.max_structured_output_retries)

    try:
        outline_results = await planner_model.ainvoke([HumanMessage(content=response_structure_instructions.format(
            messages=get_buffer_string(messages),
            context="\n".join(notes),
            feedback=""
        ))])
        return {
            "outline": outline_results.outline
        }
    except Exception as e:
        print(f"Error generating outline: {e}")


# Write the response in parallel, this is the draft.

# Reflect, and iterate on sections, potentially fetching more information if needed and re-writing some sections if needed.

# Return final report when finished.

builder = StateGraph(GeneralResearcherState, input=GeneralResearcherStateInput, output=GeneralResearcherStateOutput, config_schema=WorkflowConfiguration)
builder.add_node("generate_queries", generate_queries)
builder.add_node("search", search)
builder.add_node("generate_outline", generate_outline)
builder.add_edge(START, "generate_queries")
builder.add_edge("generate_outline", END)

general_researcher = builder.compile()