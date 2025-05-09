from typing import Literal

from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
)
from langchain_core.runnables import RunnableConfig
from langgraph.constants import TAG_NOSTREAM
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from open_deep_research.configuration import Configuration
from open_deep_research.prompts import (
    QUERY_WRITER_INSTRUCTIONS,
    SECTION_GRADER_INSTRUCTIONS,
    SECTION_WRITER_INPUTS,
    SECTION_WRITER_INSTRUCTIONS,
)
from open_deep_research.state import (
    Feedback,
    SearchQueries,
    SectionOutputState,
    SectionState,
)
from open_deep_research.utils import (
    select_and_execute_search,
)

# Used to prevent noisy stream of internal LLM calls
CONFIG_NO_STREAM = {"tags": [TAG_NOSTREAM]}

## Nodes


async def generate_queries(state: SectionState, config: RunnableConfig) -> SectionState:
    """Generate search queries for researching a specific section.

    This node uses an LLM to generate targeted search queries based on the
    section topic and description.

    Args:
        state: Current state containing section details
        config: Configuration including number of queries to generate

    Returns:
        Dict containing the generated search queries
    """
    # Get state
    topic = state["topic"]
    section = state["section"]

    # Get configuration
    configurable = Configuration.from_runnable_config(config)

    # Generate queries
    queries: SearchQueries = (
        await Configuration.init_chat_model("writer_model", config)
        .with_structured_output(SearchQueries)
        .with_config(CONFIG_NO_STREAM)
        .ainvoke(
            [
                SystemMessage(
                    QUERY_WRITER_INSTRUCTIONS.format(
                        topic=topic,
                        section_topic=section.description,
                        number_of_queries=configurable.number_of_queries,
                    )
                ),
                HumanMessage("Generate search queries on the provided topic."),
            ]
        )
    )

    return {"search_queries": queries.queries}


async def search_web(state: SectionState, config: RunnableConfig) -> SectionState:
    """Execute web searches for the section queries.

    This node:
    1. Takes the generated queries
    2. Executes searches using configured search API
    3. Formats results into usable context

    Args:
        state: Current state with search queries
        config: Search API configuration

    Returns:
        Dict with search results and updated iteration count
    """
    # Get state
    search_queries = state["search_queries"]

    # Get configuration
    configurable = Configuration.from_runnable_config(config)

    # Search the web with parameters
    source_str = await select_and_execute_search(
        configurable.search_api,
        configurable.search_api_config,
        [query.search_query for query in search_queries],
    )

    return {
        "source_str": source_str,
        "search_iterations": state["search_iterations"] + 1,
    }


async def write_section(
    state: SectionState, config: RunnableConfig
) -> Command[Literal[END, "search_web"]]:
    """Write a section of the report and evaluate if more research is needed.

    This node:
    1. Writes section content using search results
    2. Evaluates the quality of the section
    3. Either:
       - Completes the section if quality passes
       - Triggers more research if quality fails

    Args:
        state: Current state with search results and section info
        config: Configuration for writing and evaluation

    Returns:
        Command to either complete section or do more research
    """
    # Get state
    topic = state["topic"]
    section = state["section"]
    source_str = state["source_str"]

    # Get configuration
    configurable = Configuration.from_runnable_config(config)

    # Generate section
    section_content = (
        await Configuration.init_chat_model("writer_model", config)
        .with_config(CONFIG_NO_STREAM)
        .ainvoke(
            [
                SystemMessage(SECTION_WRITER_INSTRUCTIONS),
                HumanMessage(
                    SECTION_WRITER_INPUTS.format(
                        topic=topic,
                        section_name=section.name,
                        section_topic=section.description,
                        context=source_str,
                        section_content=section.content,
                    )
                ),
            ]
        )
    )

    # Write content to the section object
    section.content = section_content.text()

    # Generate feedback
    feedback: Feedback = (
        await Configuration.init_chat_model("planner_model", config)
        .with_structured_output(Feedback)
        .with_config(CONFIG_NO_STREAM)
        .ainvoke(
            [
                SystemMessage(
                    SECTION_GRADER_INSTRUCTIONS.format(
                        topic=topic,
                        section_topic=section.description,
                        section=section.content,
                        number_of_follow_up_queries=configurable.number_of_queries,
                    )
                ),
                HumanMessage(
                    "Grade the report and consider follow-up questions for missing information. "
                    "If the grade is 'pass', return empty strings for all follow-up queries. "
                    "If the grade is 'fail', provide specific search queries to gather missing information."
                ),
            ]
        )
    )

    # If the section is passing or the max search depth is reached, publish the section to completed sections
    if (
        feedback.grade == "pass"
        or state["search_iterations"] >= configurable.max_search_depth
    ):
        # Publish the section to completed sections
        return Command(update={"completed_sections": [section]}, goto=END)

    # Update the existing section with new content and update search queries
    else:
        return Command(
            update={"search_queries": feedback.follow_up_queries, "section": section},
            goto="search_web",
        )


# Report section sub-graph

# Add nodes
section_builder = StateGraph(SectionState, output=SectionOutputState)
section_builder.add_node("generate_queries", generate_queries)
section_builder.add_node("search_web", search_web)
section_builder.add_node("write_section", write_section)

# Add edges
section_builder.add_edge(START, "generate_queries")
section_builder.add_edge("generate_queries", "search_web")
section_builder.add_edge("search_web", "write_section")

build_section_with_web_research = section_builder.compile()
