from typing import Literal

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolCall,
)
from langchain_core.runnables import RunnableConfig
from langgraph.constants import TAG_NOSTREAM, Send
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from open_deep_research.configuration import Configuration
from open_deep_research.prompts import (
    FINAL_SECTION_WRITER_INSTRUCTIONS,
    QUERY_WRITER_INSTRUCTIONS,
    REPORT_PLANNER_INSTRUCTIONS,
    REPORT_PLANNER_QUERY_WRITER_INSTRUCTIONS,
    SECTION_GRADER_INSTRUCTIONS,
    SECTION_WRITER_INPUTS,
    SECTION_WRITER_INSTRUCTIONS,
)
from open_deep_research.state import (
    Feedback,
    GenerateOrRefineReport,
    ReportState,
    ReportStateInput,
    ReportStateOutput,
    SearchQueries,
    SectionOutputState,
    Sections,
    SectionState,
    StartResearch,
)
from open_deep_research.utils import (
    find_tool_call,
    format_sections,
    get_search_params,
    select_and_execute_search,
)

# Used to prevent noisy stream of internal LLM calls
CONFIG_NO_STREAM = {"tags": [TAG_NOSTREAM]}

## Nodes


async def converse(state: ReportState, config: RunnableConfig) -> ReportState:
    tools: list[ToolCall] = [GenerateOrRefineReport]
    if state["sections"] or state["topic"]:
        tools.append(StartResearch)

    message: AIMessage = (
        await Configuration.init_chat_model("converse_model", config)
        .bind_tools(tools)
        .ainvoke(
            [
                SystemMessage(
                    "You are an assistant that aids user to create a report."
                ),
                *state["messages"],
            ]
        )
    )

    if report_tool_call := find_tool_call(message, GenerateOrRefineReport):
        return {"message": [message], **report_tool_call["args"]}

    return {"messages": [message]}


async def converse_edge(
    state: ReportState,
) -> Command[Literal["generate_report_plan", "build_section_with_web_research"]]:
    last_message = state["messages"][-1]
    if find_tool_call(last_message, GenerateOrRefineReport):
        return "generate_report_plan"

    if find_tool_call(last_message, StartResearch):
        return Command(
            goto=[
                Send(
                    "build_section_with_web_research",
                    {"topic": state["topic"], "section": s, "search_iterations": 0},
                )
                for s in state["sections"]
                if s.research
            ],
        )

    return END


async def generate_report_plan(
    state: ReportState, config: RunnableConfig
) -> ReportState:
    """Generate the initial report plan with sections.

    This node:
    1. Gets configuration for the report structure and search parameters
    2. Generates search queries to gather context for planning
    3. Performs web searches using those queries
    4. Uses an LLM to generate a structured plan with sections

    Args:
        state: Current graph state containing the report topic
        config: Configuration for models, search APIs, etc.

    Returns:
        Dict containing the generated sections
    """
    # Inputs
    topic = state["topic"]
    feedback = state.get("feedback_on_report_plan", None)

    # Get configuration
    configurable = Configuration.from_runnable_config(config)

    number_of_queries = configurable.number_of_queries
    report_organization = configurable.report_structure
    if isinstance(configurable.report_structure, dict):
        report_organization = str(configurable.report_structure)

    # Generate queries
    search_queries: SearchQueries = (
        await Configuration.init_chat_model("writer_model", config)
        .with_structured_output(SearchQueries)
        .with_config(CONFIG_NO_STREAM)
        .ainvoke(
            [
                SystemMessage(
                    REPORT_PLANNER_QUERY_WRITER_INSTRUCTIONS.format(
                        topic=topic,
                        report_organization=report_organization,
                        number_of_queries=number_of_queries,
                    )
                ),
                HumanMessage(
                    "Generate search queries that will help with planning the sections of the report."
                ),
            ]
        )
    )

    # Search the web with parameters
    source_str = await select_and_execute_search(
        configurable.search_api,
        [query.search_query for query in search_queries.queries],
        get_search_params(
            configurable.search_api, configurable.search_api_config or {}
        ),
    )

    # Generate the report sections
    report_sections = (
        await Configuration.init_chat_model("planner_model", config)
        .with_structured_output(Sections)
        .with_config(CONFIG_NO_STREAM)
        .ainvoke(
            [
                SystemMessage(
                    content=REPORT_PLANNER_INSTRUCTIONS.format(
                        topic=topic,
                        report_organization=report_organization,
                        context=source_str,
                        feedback=feedback,
                    )
                ),
                HumanMessage(
                    "Generate the sections of the report. Your response must include a 'sections' field containing a list of sections. "
                    "Each section must have: name, description, plan, research, and content fields."
                ),
            ]
        )
    )

    return {"sections": report_sections.sections}


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
    queries = (
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
        [query.search_query for query in search_queries],
        get_search_params(
            configurable.search_api, configurable.search_api_config or {}
        ),
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
                SystemMessage(content=SECTION_WRITER_INSTRUCTIONS),
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
    section.content = section_content.content

    # Generate feedback
    feedback = (
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


async def write_final_sections(state: SectionState, config: RunnableConfig):
    """Write sections that don't require research using completed sections as context.

    This node handles sections like conclusions or summaries that build on
    the researched sections rather than requiring direct research.

    Args:
        state: Current state with completed sections as context
        config: Configuration for the writing model

    Returns:
        Dict containing the newly written section
    """
    # Get state
    topic = state["topic"]
    section = state["section"]
    completed_report_sections = state["report_sections_from_research"]

    # Generate section
    section_content = (
        await Configuration.init_chat_model("writer_model", config)
        .with_config(CONFIG_NO_STREAM)
        .ainvoke(
            [
                SystemMessage(
                    FINAL_SECTION_WRITER_INSTRUCTIONS.format(
                        topic=topic,
                        section_name=section.name,
                        section_topic=section.description,
                        context=completed_report_sections,
                    )
                ),
                HumanMessage(
                    "Generate a report section based on the provided sources."
                ),
            ]
        )
    )

    # Write content to section
    section.content = section_content.content

    # Write the updated section to completed sections
    return {"completed_sections": [section]}


def gather_completed_sections(
    state: ReportState,
) -> Command[Literal["write_final_sections"]]:
    """Format completed sections as context for writing final sections.

    This node takes all completed research sections and formats them into
    a single context string for writing summary sections.

    Args:
        state: Current state with completed sections

    Returns:
        Dict with formatted sections as context
    """
    # Format completed section to str to use as context for final sections
    completed_report_sections = format_sections(state["completed_sections"])

    return Command(
        goto=[
            Send(
                "write_final_sections",
                {
                    "topic": state["topic"],
                    "section": s,
                    "report_sections_from_research": completed_report_sections,
                },
            )
            for s in state["sections"]
            if not s.research
        ],
        update={"report_sections_from_research": completed_report_sections},
    )


def compile_final_report(state: ReportState):
    """Compile all sections into the final report.

    This node:
    1. Gets all completed sections
    2. Orders them according to original plan
    3. Combines them into the final report

    Args:
        state: Current state with all completed sections

    Returns:
        Dict containing the complete report
    """
    # Get sections
    sections = state["sections"]
    completed_sections = {s.name: s.content for s in state["completed_sections"]}

    # Update sections with completed content while maintaining original order
    for section in sections:
        section.content = completed_sections[section.name]

    # Compile final report
    all_sections = "\n\n".join([s.content for s in sections])

    # Respond as message
    return {"messages": [AIMessage(content=all_sections)]}


# Report section sub-graph --

# Add nodes
section_builder = StateGraph(SectionState, output=SectionOutputState)
section_builder.add_node("generate_queries", generate_queries)
section_builder.add_node("search_web", search_web)
section_builder.add_node("write_section", write_section)

# Add edges
section_builder.add_edge(START, "generate_queries")
section_builder.add_edge("generate_queries", "search_web")
section_builder.add_edge("search_web", "write_section")

# Outer graph for initial report plan compiling results from each section --

# Add nodes
builder = StateGraph(
    ReportState,
    input=ReportStateInput,
    output=ReportStateOutput,
    config_schema=Configuration,
)
builder.add_node("converse", converse)
builder.add_node("generate_report_plan", generate_report_plan)
builder.add_node("build_section_with_web_research", section_builder.compile())
builder.add_node("gather_completed_sections", gather_completed_sections)
builder.add_node("write_final_sections", write_final_sections)
builder.add_node("compile_final_report", compile_final_report)

# Add edges
builder.add_edge(START, "converse")
builder.add_conditional_edges(
    "converse",
    converse_edge,
    path_map=["build_section_with_web_research", "generate_report_plan", END],
)
builder.add_edge("build_section_with_web_research", "gather_completed_sections")
builder.add_edge("write_final_sections", "compile_final_report")
builder.add_edge("compile_final_report", END)

graph = builder.compile()
