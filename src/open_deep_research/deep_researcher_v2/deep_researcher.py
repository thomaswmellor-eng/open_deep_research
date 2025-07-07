from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage, MessageLikeRepresentation, get_buffer_string
from langchain_core.runnables import RunnableConfig
from langgraph.graph import START, END, StateGraph
from langgraph.types import Command, Send
from typing import Literal
from open_deep_research.deep_researcher_v2.configuration import (
    WorkflowConfiguration, 
)
from open_deep_research.deep_researcher_v2.state import (
    GeneralResearcherState,
    GeneralResearcherStateInput,
    GeneralResearcherStateOutput,
    ClarifyWithUser,
    LeadResearcherReflection,
    ResearchQuestion,
    SubResearcherState
)
from open_deep_research.deep_researcher_v2.prompts import (
    clarify_with_user_instructions,
    transform_messages_into_research_topic_prompt,
    research_system_prompt,
    lead_researcher_reflection_prompt,
    sub_researcher_instruction,
    research_unit_condense_output_prompt,
    final_report_generation_prompt,
    initial_researcher_instructions,
)
from open_deep_research.deep_researcher_v2.utils import (
    get_today_str,
    is_token_limit_exceeded,
    get_model_token_limit,
    get_all_tools,
    openai_websearch_called,
    anthropic_websearch_called,
    remove_up_to_last_ai_message,
    get_api_key_for_model
)

# Initialize a configurable model that we will use throughout the agent
configurable_model = init_chat_model(
    configurable_fields=("model", "max_tokens", "api_key"),
)

async def clarify_with_user(state: GeneralResearcherState, config: RunnableConfig) -> Command[Literal["research_supervisor", "__end__"]]:
    """
    This node is responsible for asking the user a clarifying question if the user's request is not clear.
    It will only ask the user a clarifying question if the allow_clarification flag is set to True.
    If the user's request is clear, it will skip the clarifying question and go to the research supervisor.
    """
    configurable = WorkflowConfiguration.from_runnable_config(config)
    if not configurable.allow_clarification:
        return Command(goto="research_supervisor")
    messages = state["messages"]
    model_config = {
        "model": configurable.research_model,
        "api_key": get_api_key_for_model(configurable.research_model, config),
        "tags": ["langsmith:nostream"]
    }
    model = configurable_model.with_config(model_config).with_structured_output(ClarifyWithUser).with_retry(stop_after_attempt=configurable.max_structured_output_retries)
    response = await model.ainvoke([HumanMessage(content=clarify_with_user_instructions.format(messages=get_buffer_string(messages), date=get_today_str()))])
    if response.need_clarification:
        return Command(goto=END, update={"messages": [AIMessage(content=response.question)]})
    else:
        return Command(goto="research_supervisor")


async def extract_research_question(state: GeneralResearcherState, config: RunnableConfig):
    """
    This node is responsible for extracting the research question from the user's request.
    The research_question is the research brief that the researcher will use to conduct research.
    We format the conversation history into a research brief, which is then passed to the researcher.
    """
    configurable = WorkflowConfiguration.from_runnable_config(config)
    research_model_config = {
        "model": configurable.research_model,
        "max_tokens": configurable.research_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.research_model, config),
        "tags": ["langsmith:nostream"]
    }
    research_model = configurable_model.with_config(research_model_config).with_structured_output(ResearchQuestion).with_retry(stop_after_attempt=configurable.max_structured_output_retries)
    response = await research_model.ainvoke([HumanMessage(content=transform_messages_into_research_topic_prompt.format(
        messages=get_buffer_string(state.get("messages", [])),
        date=get_today_str()
    ))])
    return {
        "research_question": response.research_question
    }


async def react_agent_research(system_prompt: str, research_messages: list[MessageLikeRepresentation], config: RunnableConfig):
    """
    This helper function is responsible for conducting a ReAct loop to conduct research.
    It will call the tools provided to it, and return a synthesized report of the research.
    """
    configurable = WorkflowConfiguration.from_runnable_config(config)
    tools = await get_all_tools(config)
    if len(tools) == 0:
        raise ValueError("No tools found to conduct research: Please configure either your search API or add MCP tools to your configuration.")
    tools_by_name = {tool.name if hasattr(tool, "name") else tool.get("name", "web_search"):tool for tool in tools}
    research_model_config = {
        "model": configurable.research_model,
        "max_tokens": configurable.research_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.research_model, config),
        "tags": ["langsmith:nostream"]
    }
    research_model = configurable_model.with_config(research_model_config).bind_tools(tools).with_retry(stop_after_attempt=configurable.max_structured_output_retries)
    tool_calling_iterations = 0
    while tool_calling_iterations < configurable.max_react_tool_calls:
        try:
            response = await research_model.ainvoke([SystemMessage(content=system_prompt), *research_messages])
            research_messages.append(response)
        except Exception as e:
            if is_token_limit_exceeded(e, configurable.research_model):
                print(f"Token limit exceeded while researching: {e}")
                break
            else:
                print(f"Error in research model: {e}")
            tool_calling_iterations += 1
            continue
        # If no tools or native web search tools were called, then this research iteration is complete.
        if len(response.tool_calls) == 0 and not (openai_websearch_called(response) or anthropic_websearch_called(response)):
            break
        tool_calls = response.tool_calls
        finished_research = False
        for tool_call in tool_calls:
            tool = tools_by_name[tool_call["name"]]
            if tool_call["name"] == "ResearchComplete":
                observation = "Research complete."
                finished_research = True
            else:
                try:
                    observation = await tool.ainvoke(tool_call["args"], config)
                except Exception as e:
                    observation = f"Error calling tool {tool_call['name']}: {e}"
            research_messages.append(ToolMessage(
                content=observation,
                name=tool_call["name"],
                tool_call_id=tool_call["id"]
            ))  
        if finished_research:
            break
        tool_calling_iterations += 1

    synthesis_attempts = 0
    synthesizer_model = configurable_model.with_config({
        "model": configurable.research_model,
        "max_tokens": configurable.research_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.research_model, config),
        "tags": ["langsmith:nostream"]
    })
    research_messages.append(HumanMessage(content=research_unit_condense_output_prompt))
    while synthesis_attempts < 3:
        try:
            response = await synthesizer_model.ainvoke([SystemMessage(content=system_prompt), *research_messages])
            return response.content
        except Exception as e:
            synthesis_attempts += 1
            if is_token_limit_exceeded(e, configurable.research_model):
                research_messages = remove_up_to_last_ai_message(research_messages)
                print(f"Token limit exceeded while synthesizing: {e}. Pruning the messages to try again.")
                continue         
            print(f"Error synthesizing research report: {e}")
    return "Error synthesizing research report: Maximum retries exceeded"


async def initial_research(state: GeneralResearcherState, config: RunnableConfig):
    """
    This node is responsible for conducting initial research on the user's request.
    A ReAct loop is used to conduct research on the research_question.
    The ReAct agent is given access to the configured search API and MCP tools.
    """
    configurable = WorkflowConfiguration.from_runnable_config(config)
    system_prompt = research_system_prompt.format(
        mcp_prompt=configurable.mcp_prompt or "",
        date=get_today_str()
    )
    research_messages = [HumanMessage(content=initial_researcher_instructions.format(
        research_question=state.get("research_question", "")
    ))]
    synthesized_report = await react_agent_research(system_prompt, research_messages, config)
    return {
        "notes": [f"Initial research report:\n\n{synthesized_report}"]
    }


async def lead_researcher(state: GeneralResearcherState, config: RunnableConfig) -> Command[Literal["sub_researcher", "__end__"]]:
    notes = state.get("notes", [])
    configurable = WorkflowConfiguration.from_runnable_config(config)
    if state.get("search_attempts", 1) > configurable.max_researcher_iterations:
        return Command(goto=END)
    reflection_model_config = {
        "model": configurable.reflection_model,
        "max_tokens": configurable.reflection_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.research_model, config),
        "tags": ["langsmith:nostream"]
    }
    reflection_model = configurable_model.with_config(reflection_model_config).with_structured_output(LeadResearcherReflection).with_retry(stop_after_attempt=configurable.max_structured_output_retries)
    try:
        response = await reflection_model.ainvoke([
            HumanMessage(content=lead_researcher_reflection_prompt.format(
                date=get_today_str(),
                research_question=state.get("research_question", ""),
                findings="\n".join(notes),
                max_concurrent_research_units=configurable.max_concurrent_research_units
            ))
        ])
        if response.is_satisfied:
            return Command(goto=END)
        else:
            topics_to_research = response.topics_to_research[:configurable.max_concurrent_research_units]
            return Command(
                goto=[
                    Send(
                        "sub_researcher",
                        {
                            "research_messages": [
                                HumanMessage(content=sub_researcher_instruction.format( 
                                    research_question=state.get("research_question", ""),
                                    topic=topic
                                ))
                            ],
                            "specific_topic": topic
                        }
                    ) for topic in topics_to_research
                ],
                update={
                    "search_attempts": state.get("search_attempts", 1) + 1
                }
            )
    except Exception as e:
        if is_token_limit_exceeded(e, configurable.reflection_model):
            print(f"Token limit exceeded while reflecting: {e}")
            return Command(goto=END)
        else:
            print(f"Error in reflection phase: {e}")
        return Command(goto="sub_researcher", update={"research_iterations": state.get("research_iterations", 1) + 1})
    

async def sub_researcher(state: SubResearcherState, config: RunnableConfig) -> Command[Literal["lead_researcher"]]:
    configurable = WorkflowConfiguration.from_runnable_config(config)
    research_messages = state.get("research_messages", [])
    topic = state.get("specific_topic", "")
    system_prompt = research_system_prompt.format(mcp_prompt=configurable.mcp_prompt or "", date=get_today_str())
    synthesized_report = await react_agent_research(system_prompt, research_messages, config)
    return Command(
        goto="lead_researcher",
        update={
            "notes": [f"Completed research on topic {topic}:\n\n{synthesized_report}"]
        }
    )

research_supervisor_builder = StateGraph(GeneralResearcherState)
research_supervisor_builder.add_node("extract_research_question", extract_research_question)
research_supervisor_builder.add_node("initial_research", initial_research)
research_supervisor_builder.add_node("lead_researcher", lead_researcher)
research_supervisor_builder.add_node("sub_researcher", sub_researcher)
research_supervisor_builder.add_edge(START, "extract_research_question")
research_supervisor_builder.add_edge("extract_research_question", "initial_research")
research_supervisor_builder.add_edge("initial_research", "lead_researcher")
research_supervisor = research_supervisor_builder.compile()


async def final_report_generation(state: GeneralResearcherState, config: RunnableConfig):
    notes = state.get("notes", [])
    cleared_state = {
        "notes": {"type": "override", "value": []},
        "collected_sources": {"type": "override", "value": []},
        "search_attempts": 0
    }
    configurable = WorkflowConfiguration.from_runnable_config(config)
    writer_model_config = {
        "model": configurable.final_report_model,
        "max_tokens": configurable.final_report_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.research_model, config),
    }
    
    findings = "\n".join(notes)
    max_retries = 3
    current_retry = 0
    while current_retry <= max_retries:
        final_report_prompt = final_report_generation_prompt.format(
            research_question=state.get("research_question", ""),
            findings=findings,
            date=get_today_str()
        )
        try:
            final_report = await configurable_model.with_config(writer_model_config).ainvoke([HumanMessage(content=final_report_prompt)])
            return {
                "final_report": final_report, 
                "messages": [final_report],
                **cleared_state
            }
        except Exception as e:
            if is_token_limit_exceeded(e, configurable.final_report_model):
                if current_retry == 0:
                    model_token_limit = get_model_token_limit(configurable.final_report_model)
                    if not model_token_limit:
                        return {
                            "final_report": f"Error generating final report: Token limit exceeded, however, we could not determine the model's maximum context length. Please update the model map in deep_researcher/utils.py with this information. {e}",
                            **cleared_state
                        }
                    findings_token_limit = model_token_limit * 4
                else:
                    findings_token_limit = int(findings_token_limit * 0.9)
                print("Reducing the chars to", findings_token_limit)
                findings = findings[:findings_token_limit]
                current_retry += 1
            else:
                # If not a token limit exceeded error, then we just throw an error.
                return {
                    "final_report": f"Error generating final report: {e}",
                    **cleared_state
                }
    return {
        "final_report": "Error generating final report: Maximum retries exceeded",
        **cleared_state
    }

deep_researcher_builder = StateGraph(GeneralResearcherState, input=GeneralResearcherStateInput, output=GeneralResearcherStateOutput, config_schema=WorkflowConfiguration)
deep_researcher_builder.add_node("research_supervisor", research_supervisor)
deep_researcher_builder.add_node("final_report_generation", final_report_generation)
deep_researcher_builder.add_node("clarify_with_user", clarify_with_user)
deep_researcher_builder.add_edge(START, "clarify_with_user")
deep_researcher_builder.add_edge("research_supervisor", "final_report_generation")
deep_researcher_builder.add_edge("clarify_with_user", END)
deep_researcher_builder.add_edge("final_report_generation", END)

deep_researcher = deep_researcher_builder.compile()