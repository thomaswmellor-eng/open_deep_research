from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, get_buffer_string
from langchain_core.runnables import RunnableConfig
from langgraph.constants import Send
from langgraph.graph import START, END, StateGraph
from langgraph.types import interrupt, Command
import asyncio
import warnings
from typing import Literal, cast
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

from open_deep_research.general_researcher.configuration import (
    WorkflowConfiguration, 
    SearchAPI, 
    extract_content_from_response,
    call_source_extractor,
    MODELS_WITH_WEB_SEARCH
)
from open_deep_research.general_researcher.state import (
    UpfrontResearcherState,
    GeneralResearcherState,
    GeneralResearcherStateInput,
    GeneralResearcherStateOutput,
    Outline,
    ReflectionResult
)
from open_deep_research.utils import (
    get_config_value, 
    tavily_search,
    duckduckgo_search
)
from open_deep_research.general_researcher.prompts import (
    response_structure_instructions, 
    initial_upfront_model_provider_web_search_system_prompt,
    follow_up_upfront_model_provider_web_search_system_prompt,
    upfront_model_provider_reflection_system_prompt,
    gap_context_prompt,
    final_report_generation_prompt
)

configurable_model = init_chat_model(
    max_tokens=10000,
    configurable_fields=("model", "max_tokens"),
)


async def get_search_tool(search_api: SearchAPI):
    if search_api == SearchAPI.ANTHROPIC:
        return [{"type": "web_search_20250305", "name": "web_search"}]
    elif search_api == SearchAPI.OPENAI:
        return [{"type": "web_search_preview"}]
    elif search_api == SearchAPI.TAVILY:
        search_tool = tavily_search
        tool_metadata = {**(search_tool.metadata or {}), "type": "search", "name": "web_search"}
        search_tool.metadata = tool_metadata
        return [search_tool]
    elif search_api == SearchAPI.DUCKDUCKGO:
        search_tool = duckduckgo_search
        tool_metadata = {**(search_tool.metadata or {}), "type": "search", "name": "web_search"}
        search_tool.metadata = tool_metadata
        return [search_tool]

async def load_mcp_tools(
    config: RunnableConfig,
    existing_tool_names: set[str],
) -> list[BaseTool]:
    configurable = WorkflowConfiguration.from_runnable_config(config)
    if not configurable.mcp_server_config:
        return []
    mcp_server_config = configurable.mcp_server_config
    client = MultiServerMCPClient(mcp_server_config)
    mcp_tools = await client.get_tools()
    filtered_mcp_tools: list[BaseTool] = []
    for tool in mcp_tools:
        if tool.name in existing_tool_names:
            warnings.warn(
                f"Trying to add MCP tool with a name {tool.name} that is already in use - this tool will be ignored."
            )
            continue
        if configurable.mcp_tools_to_include and tool.name not in configurable.mcp_tools_to_include:
            continue
        filtered_mcp_tools.append(tool)
    return filtered_mcp_tools


# Upfront Model Provider Research
async def research(state: UpfrontResearcherState, config: RunnableConfig) -> Command[Literal["reflection", "__end__"]]:
    search_attempts = state.get("search_attempts", 0)
    configurable = WorkflowConfiguration.from_runnable_config(config)
    search_api = SearchAPI(get_config_value(configurable.search_api))
    research_model_config = {
        "model": configurable.research_model,
        "max_tokens": configurable.research_model_max_tokens,
    }
    # Perform web search and then forward to the model_provider_reflection
    tools = []
    search_tool = await get_search_tool(search_api)
    tools.extend(search_tool)
    existing_tool_names = {tool.name if hasattr(tool, "name") else tool.get("name", "web_search") for tool in tools}
    tools_by_name = {tool.name if hasattr(tool, "name") else tool.get("name", "web_search"):tool for tool in tools}
    mcp_tools = await load_mcp_tools(config, existing_tool_names)
    tools.extend(mcp_tools)

    model_with_search = configurable_model.with_config(research_model_config).bind_tools(tools)
    system_prompt = initial_upfront_model_provider_web_search_system_prompt if search_attempts == 0 else follow_up_upfront_model_provider_web_search_system_prompt
    research_messages = state.get("research_messages", [])
    if search_attempts == 0:
        # If this is the first search, use the original messages as a starting point.
        research_messages = state.get("messages").copy()
    while search_attempts < configurable.max_search_depth:
        try:
            response = await model_with_search.ainvoke([SystemMessage(content=system_prompt), *research_messages])
            collected_sources = []
            current_findings = ""
            # If the model supports web search, we can try to extract sources from the response
            if search_api in MODELS_WITH_WEB_SEARCH:
                collected_sources = call_source_extractor(response, search_attempts, search_api)
                current_findings = extract_content_from_response(response)
            # If there are tool calls made, then these are actual tool calls from non-native web search, or MCP tools. We need to add to the notes and sources.
            for tool_call in response.tool_calls:
                tool = tools_by_name[tool_call["name"]]
                try:
                    observation = await tool.ainvoke(tool_call["args"], config)
                except NotImplementedError:
                    observation = tool.invoke(tool_call["args"], config)
                current_findings += f"\n{observation}"
            
            return Command(
                goto="reflection",
                update={
                    "notes": [current_findings],
                    "research_messages": research_messages,
                    "collected_sources": collected_sources,
                    "search_attempts": search_attempts + 1
                }
            )
        except Exception as e:
            print(f"Error in research phase: {e}")
            search_attempts += 1
    return Command(goto="__end__")


async def reflection(state: UpfrontResearcherState, config: RunnableConfig) -> Command[Literal["research", "__end__"]]:
    messages = state["messages"]
    notes = state.get("notes", [])
    configurable = WorkflowConfiguration.from_runnable_config(config)
    if state.get("search_attempts", 0) >= configurable.max_search_depth:
        return Command(goto=END)
    reflection_model_config = {
        "model": configurable.reflection_model,
        "max_tokens": configurable.reflection_model_max_tokens,
    }
    reflection_model = configurable_model.with_config(reflection_model_config).with_structured_output(ReflectionResult).with_retry(stop_after_attempt=configurable.max_structured_output_retries)
    reflection_prompt = upfront_model_provider_reflection_system_prompt.format(
        messages=get_buffer_string(messages),
        findings="\n".join(notes),
    )
    try:
        response = await reflection_model.ainvoke([HumanMessage(content=reflection_prompt)])
        if response["is_satisfied"]:
            return Command(goto=END)
        else:
            knowledge_gaps = "\n".join([f"- {gap}" for gap in response["knowledge_gaps"]])
            focus_areas = "\n".join([f"- {query}" for query in response["suggested_queries"]])
            research_messages = [
                *state.get("messages", []),
                AIMessage(content=f"Current research status:\n\n{notes}"),
                HumanMessage(content=gap_context_prompt.format(
                    knowledge_gaps=knowledge_gaps,
                    focus_areas=focus_areas,
                    reasoning=response["reasoning"]
                ))
            ]
            return Command(
                goto="research",
                update={
                    "research_messages": research_messages,
                }
            )
    except Exception as e:
        print(f"Error in reflection phase: {e}")
        # TODO: Figure out a better way to loop here.
        return Command(goto="model_provider_reflection", update={"search_attempts": state.get("search_attempts", 0) + 1})


upfront_researcher_builder = StateGraph(UpfrontResearcherState)
upfront_researcher_builder.add_node("research", research)
upfront_researcher_builder.add_node("reflection", reflection)
upfront_researcher_builder.add_edge(START, "research")
upfront_researcher = upfront_researcher_builder.compile()


# Outline Generation after Upfront Research
async def generate_outline(state: GeneralResearcherState, config: RunnableConfig) -> Command[Literal["human_feedback", "final_report_generation"]]:
    messages = state["messages"]
    notes = state.get("notes", [])
    feedback_on_outline = state.get("feedback_on_outline", [])
    configurable = WorkflowConfiguration.from_runnable_config(config)
    planner_model_config = {
        "model": configurable.outliner_model,
        "max_tokens": configurable.outliner_model_max_tokens,
    }
    planner_model = configurable_model.with_config(planner_model_config).with_structured_output(Outline).with_retry(stop_after_attempt=configurable.max_structured_output_retries)
    try:
        outline_results = await asyncio.wait_for(
            planner_model.ainvoke(
                [HumanMessage(
                    content=response_structure_instructions.format(
                        messages=get_buffer_string(messages),
                        context="\n".join(notes),
                        feedback="\n".join(feedback_on_outline)))
                ]
            ),
            timeout=45.0
        )
        if configurable.outline_user_approval:
            return Command(goto="human_feedback", update={"outline": outline_results.outline})
        else:
            return Command(goto="final_report_generation", update={"outline": outline_results.outline})
    except Exception as e:
        print(f"Error generating outline: {e}")


async def human_feedback(state: GeneralResearcherState, config: RunnableConfig) -> Command[Literal["generate_outline", "final_report_generation"]]:
    outline = state["outline"]
    outline_str = "\n\n".join(
        f"Section: {section.name}\n"
        f"Description: {section.description}\n"
        for section in outline
    )
    interrupt_message = f"""Please provide feedback on the following outline. 
                        \n\n{outline_str}\n
                        \nDoes the report plan meet your needs?\nPass 'true' to approve the report plan.\nOr, provide feedback to regenerate the report plan:"""
    feedback = interrupt(interrupt_message)
    if (isinstance(feedback, bool) and feedback is True) or (isinstance(feedback, str) and feedback.lower() == "true"):
        return Command(goto="final_report_generation", update={"outline": outline})
    elif isinstance(feedback, str):
        return Command(goto="generate_outline", update={"feedback_on_outline": [feedback]})
    else:
        raise TypeError(f"Interrupt value of type {type(feedback)} is not supported.")


async def final_report_generation(state: GeneralResearcherState, config: RunnableConfig):
    messages = state["messages"]
    outline = state["outline"]
    notes = state.get("notes", [])
    collected_sources = state.get("collected_sources", [])
    configurable = WorkflowConfiguration.from_runnable_config(config)
    writer_model_config = {
        "model": configurable.final_report_model,
        "max_tokens": configurable.final_report_model_max_tokens,
    }
    final_report_prompt = final_report_generation_prompt.format(
        messages=get_buffer_string(messages),
        findings="\n".join(notes),
        source_list="\n".join([f"{source.title} ({source.url})" for source in collected_sources]),
        outline="\n\n".join([f"## {section.name}\n{section.description}" for section in outline])
    )
    try:
        final_report = await asyncio.wait_for(
            configurable_model.with_config(writer_model_config).ainvoke([HumanMessage(content=final_report_prompt)]), 
            timeout=120.0
        )
        return {"final_report": final_report}
    except Exception as e:
        print(f"Error generating final report: {e}")
        return {"final_report": "Error generating final report"}



general_researcher_builder = StateGraph(GeneralResearcherState, input=GeneralResearcherStateInput, output=GeneralResearcherStateOutput, config_schema=WorkflowConfiguration)
general_researcher_builder.add_node("upfront_researcher", upfront_researcher)
general_researcher_builder.add_node("generate_outline", generate_outline)
general_researcher_builder.add_node("human_feedback", human_feedback)
general_researcher_builder.add_node("final_report_generation", final_report_generation)
general_researcher_builder.add_edge(START, "upfront_researcher")
general_researcher_builder.add_edge("upfront_researcher", "generate_outline")
general_researcher_builder.add_edge("final_report_generation", END)

general_researcher = general_researcher_builder.compile()