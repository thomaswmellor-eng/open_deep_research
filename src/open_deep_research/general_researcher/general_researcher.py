from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage, get_buffer_string
from langchain_core.runnables import RunnableConfig
from langgraph.constants import Send
from langgraph.graph import START, END, StateGraph
from langgraph.types import interrupt, Command
import asyncio
from typing import Literal
from open_deep_research.general_researcher.configuration import (
    WorkflowConfiguration, 
    SearchAPI, 
)
from open_deep_research.general_researcher.state import (
    ResearchUnitState,
    GeneralResearcherState,
    GeneralResearcherStateInput,
    GeneralResearcherStateOutput,
    Outline,
    ReflectionResult
)
from open_deep_research.utils import (
    get_config_value,
)
from open_deep_research.general_researcher.prompts import (
    response_structure_instructions, 
    initial_upfront_model_provider_web_search_system_prompt,
    follow_up_upfront_model_provider_web_search_system_prompt,
    upfront_model_provider_reflection_system_prompt,
    gap_context_prompt,
    final_report_generation_prompt
)
from open_deep_research.general_researcher.utils import (
    get_search_tool,
    load_mcp_tools,
    extract_notes_from_research_messages,
)

configurable_model = init_chat_model(
    max_tokens=10000,
    configurable_fields=("model", "max_tokens"),
)

# Upfront Research
async def research(state: ResearchUnitState, config: RunnableConfig) -> Command[Literal["reflection", "__end__"]]:
    configurable = WorkflowConfiguration.from_runnable_config(config)
    research_model_config = {
        "model": configurable.research_model,
        "max_tokens": configurable.research_model_max_tokens,
        "tags": ["langsmith:nostream"]
    }
    search_api = SearchAPI(get_config_value(configurable.search_api))
    research_messages = state.get("research_messages", [])
    num_initial_messages = len(research_messages)
    if len(research_messages) == 0:
        research_messages = state.get("messages", []).copy()
    research_iterations = state.get("research_iterations", 1)
    tools = []
    tools.extend(await get_search_tool(search_api))
    existing_tool_names = {tool.name if hasattr(tool, "name") else tool.get("name", "web_search") for tool in tools}
    mcp_tools = await load_mcp_tools(config, existing_tool_names)
    tools.extend(mcp_tools)
    tools_by_name = {tool.name if hasattr(tool, "name") else tool.get("name", "web_search"):tool for tool in tools}
    research_model = configurable_model.with_config(research_model_config).bind_tools(tools)
    tool_calling_iterations = 0
    system_prompt = initial_upfront_model_provider_web_search_system_prompt.format(
        mcp_prompt=configurable.mcp_prompt or ""
    ) if research_iterations == 0 else follow_up_upfront_model_provider_web_search_system_prompt.format(
        mcp_prompt=configurable.mcp_prompt or ""
    )
    # ReAct Tool Calling Loop
    while tool_calling_iterations < 5: # TODO: Replace with configurable
        response = await research_model.ainvoke([SystemMessage(content=system_prompt), *research_messages])
        research_messages.append(response)
        if len(response.tool_calls) == 0:
            # If no MCP or non-native web search tools were called, then this research iteration is complete.
            break
        tool_calls = response.tool_calls
        web_search_called = False
        for tool_call in tool_calls:
            tool = tools_by_name[tool_call["name"]]
            metadata = getattr(tool, "metadata", {}) or {}
            if metadata.get("type") == "search":
                web_search_called = True
            try:
                observation = await tool.ainvoke(tool_call["args"], config)
            except NotImplementedError:
                observation = tool.invoke(tool_call["args"], config)
            research_messages.append(ToolMessage(
                content=observation,
                name=tool_call["name"],
                tool_call_id=tool_call["id"]
            ))
        # Once we perform a web search, we should exit the loop for feedback.
        if web_search_called:
            break
        tool_calling_iterations += 1
    # We need to assemble clean notes from our research_messages
    new_notes, collected_sources = extract_notes_from_research_messages(research_messages[num_initial_messages:], research_iterations, search_api)
    return Command(
        goto="reflection",
        update={
            "notes": [new_notes],
            "collected_sources": collected_sources,
            "research_iterations": research_iterations + 1,
        }
    )


async def reflection(state: ResearchUnitState, config: RunnableConfig) -> Command[Literal["research", "__end__"]]:
    messages = state["messages"]
    notes = state.get("notes", [])
    configurable = WorkflowConfiguration.from_runnable_config(config)
    if state.get("research_iterations", 1) > configurable.max_search_depth:
        return Command(goto=END)
    reflection_model_config = {
        "model": configurable.reflection_model,
        "max_tokens": configurable.reflection_model_max_tokens,
        "tags": ["langsmith:nostream"]
    }
    reflection_model = configurable_model.with_config(reflection_model_config).with_structured_output(ReflectionResult).with_retry(stop_after_attempt=configurable.max_structured_output_retries)
    findings = "\n".join(notes)
    reflection_prompt = upfront_model_provider_reflection_system_prompt.format(
        messages=get_buffer_string(messages),
        findings=findings,
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
                AIMessage(content=f"Current research status:\n\n{findings}"),
                HumanMessage(content=gap_context_prompt.format(
                    knowledge_gaps=knowledge_gaps,
                    focus_areas=focus_areas,
                    reasoning=response["reasoning"]
                ))
            ]
            return Command(
                goto="research",
                update={
                    "research_messages": research_messages
                }
            )
    except Exception as e:
        print(f"Error in reflection phase: {e}")
        # TODO: Figure out a better way to loop here.
        return Command(goto="model_provider_reflection", update={"research_iterations": state.get("research_iterations", 1) + 1})


upfront_researcher_builder = StateGraph(ResearchUnitState)
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
        "tags": ["langsmith:nostream"]
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
        print("Final report successfully generated")
        return {"final_report": final_report, "messages": [final_report]}
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