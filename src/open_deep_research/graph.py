from typing import Literal

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolCall,
    ToolMessage,
)
from langchain_core.runnables import RunnableConfig
from langgraph.constants import TAG_NOSTREAM, Send
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from open_deep_research.configuration import Configuration
from open_deep_research.graph_section import (
    build_section_with_web_research,
    write_final_sections,
)
from open_deep_research.prompts import (
    CONVERSE_INSTRUCTIONS,
    REPORT_PLANNER_INSTRUCTIONS,
    REPORT_PLANNER_QUERY_WRITER_INSTRUCTIONS,
)
from open_deep_research.state import (
    GenerateOrRefineReport,
    ReportState,
    ReportStateInput,
    ReportStateOutput,
    SearchQueries,
    Sections,
    StartResearch,
)
from open_deep_research.utils import (
    find_tool_call,
    format_sections,
    select_and_execute_search,
)

# Used to prevent noisy stream of internal LLM calls
# CONFIG_NO_STREAM = {"tags": [TAG_NOSTREAM]}
CONFIG_NO_STREAM = {"tags": []}

## Nodes


# TODO: convert the report plan and build report into tools / subgraphs
# So that we don't need to reinvent the tool workflow
async def converse(
    state: ReportState, config: RunnableConfig
) -> Command[Literal["build_section_with_web_research", "generate_report_plan"]]:
    tools: list[ToolCall] = [GenerateOrRefineReport]

    if (state.get("sections") or state.get("topic")) and isinstance(
        state["messages"][-1], HumanMessage
    ):
        tools.append(StartResearch)

    message: AIMessage = (
        await Configuration.init_chat_model("converse_model", config)
        .bind_tools(tools, parallel_tool_calls=False)
        .ainvoke(
            [
                SystemMessage(CONVERSE_INSTRUCTIONS),
                *state["messages"],
            ],
        )
    )

    if find_tool_call(message, GenerateOrRefineReport):
        return Command(
            goto="generate_report_plan",
            update={"messages": [message]},
        )

    if find_tool_call(message, StartResearch):
        # Start generation of report
        # (1) Schedule research + write of sections that require research
        # (2) After gathering those, write the final sections to complete the artifact
        return Command(
            goto=[
                Send(
                    "build_section_with_web_research",
                    {"topic": state["topic"], "section": s, "search_iterations": 0},
                )
                for s in state["sections"]
                if s.research
            ],
            update={"messages": [message]},
        )

    return {"messages": [message]}


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
    # Last message
    tool_call = find_tool_call(state["messages"][-1], GenerateOrRefineReport)
    topic = tool_call["args"]["topic"]
    feedback = tool_call["args"].get("feedback_on_report_plan", None)

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
        configurable.search_api_config,
        [query.search_query for query in search_queries.queries],
    )

    # Generate the report sections
    report_sections: Sections = (
        await Configuration.init_chat_model("planner_model", config)
        .with_structured_output(Sections)
        .with_config(CONFIG_NO_STREAM)
        .ainvoke(
            [
                SystemMessage(
                    REPORT_PLANNER_INSTRUCTIONS.format(
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

    # Find associate tool message to respond to
    tool_message = ToolMessage(
        content=report_sections.model_dump_json(), tool_call_id=tool_call.get("id")
    )

    return {
        "messages": [tool_message],
        "topic": topic,
        "sections": report_sections.sections,
    }


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


def compile_final_report(state: ReportState) -> ReportState:
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
    messages: list[BaseMessage] = []
    if start_research := find_tool_call(state["messages"][-1], StartResearch):
        messages.append(ToolMessage("Done", tool_call_id=start_research.get("id")))
    messages.append(AIMessage(content=all_sections))

    return {"messages": messages}


# Add nodes
builder = StateGraph(
    ReportState,
    input=ReportStateInput,
    output=ReportStateOutput,
    config_schema=Configuration,
)
builder.add_node("converse", converse)
builder.add_node("generate_report_plan", generate_report_plan)
builder.add_node("build_section_with_web_research", build_section_with_web_research)
builder.add_node("gather_completed_sections", gather_completed_sections)
builder.add_node("write_final_sections", write_final_sections)
builder.add_node("compile_final_report", compile_final_report)

# Add edges
builder.add_edge(START, "converse")
builder.add_edge("generate_report_plan", "converse")
builder.add_edge("build_section_with_web_research", "gather_completed_sections")
builder.add_edge("write_final_sections", "compile_final_report")
builder.add_edge("compile_final_report", END)

graph = builder.compile()
