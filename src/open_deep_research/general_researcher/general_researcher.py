from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, get_buffer_string
from langchain_core.runnables import RunnableConfig
from langgraph.constants import Send
from langgraph.graph import START, END, StateGraph
from langgraph.types import interrupt, Command
import asyncio
from typing import Literal
from open_deep_research.general_researcher.configuration import (
    WorkflowConfiguration, 
    SearchAPI, 
    MODELS_WITH_WEB_SEARCH, 
    get_model_with_web_search,
    extract_content_from_response,
    call_source_extractor
)
from open_deep_research.general_researcher.state import (
    UpfrontModelProviderResearcherState,
    UpfrontWebResearcherState,
    GeneralResearcherState,
    GeneralResearcherStateInput,
    GeneralResearcherStateOutput,
    Queries,
    Outline,
    ReflectionResult
)
from open_deep_research.utils import (
    get_config_value, 
    get_search_params, 
    select_and_execute_search,
    get_today_str
)
from open_deep_research.general_researcher.prompts import (
    query_writer_instructions, 
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

def initial_router(state: GeneralResearcherState, config: RunnableConfig):
    configurable = WorkflowConfiguration.from_runnable_config(config)
    search_api = SearchAPI(get_config_value(configurable.search_api))
    if (
        search_api in MODELS_WITH_WEB_SEARCH
        and configurable.research_model in {model.value for model in MODELS_WITH_WEB_SEARCH[search_api]}
    ):
        return "upfront_model_provider_researcher"
    else:
        return "upfront_web_researcher"


# Upfront Model Provider Research
async def model_provider_web_search(state: UpfrontModelProviderResearcherState, config: RunnableConfig) -> Command[Literal["model_provider_reflection", "__end__"]]:
    search_attempts = state.get("search_attempts", 0)
    configurable = WorkflowConfiguration.from_runnable_config(config)
    search_api = SearchAPI(get_config_value(configurable.search_api))
    research_model_config = {
        "model": configurable.research_model,
        "max_tokens": configurable.research_model_max_tokens,
    }
    # Perform web search and then forward to the model_provider_reflection
    model_with_websearch = get_model_with_web_search(configurable_model.with_config(research_model_config), search_api)
    system_prompt = initial_upfront_model_provider_web_search_system_prompt if search_attempts == 0 else follow_up_upfront_model_provider_web_search_system_prompt
    research_messages = state.get("research_messages", [])
    if search_attempts == 0:
        # If this is the first search, use the original messages as a starting point.
        research_messages = state.get("messages").copy()
    while search_attempts < configurable.max_search_depth:
        try:
            response = await model_with_websearch.ainvoke([SystemMessage(content=system_prompt), *research_messages])
            collected_sources = call_source_extractor(response, search_attempts, search_api)
            current_findings = extract_content_from_response(response)
            return Command(
                goto="model_provider_reflection",
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


async def model_provider_reflection(state: UpfrontModelProviderResearcherState, config: RunnableConfig) -> Command[Literal["model_provider_web_search", "__end__"]]:
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
                goto="model_provider_web_search",
                update={
                    "research_messages": research_messages,
                }
            )
    except Exception as e:
        print(f"Error in reflection phase: {e}")
        # TODO: Figure out a better way to loop here.
        return Command(goto="model_provider_reflection", update={"search_attempts": state.get("search_attempts", 0) + 1})


upfront_model_provider_researcher_builder = StateGraph(UpfrontModelProviderResearcherState)
upfront_model_provider_researcher_builder.add_node("model_provider_web_search", model_provider_web_search)
upfront_model_provider_researcher_builder.add_node("model_provider_reflection", model_provider_reflection)
upfront_model_provider_researcher_builder.add_edge(START, "model_provider_web_search")
upfront_model_provider_researcher = upfront_model_provider_researcher_builder.compile()


# Upfront Web Research
async def generate_upfront_queries(state: UpfrontWebResearcherState, config: RunnableConfig) -> Command[Literal["upfront_search", "__end__"]]:
    messages = state["messages"]
    notes = state.get("notes", [])
    historical_queries = state.get("historical_queries", [])
    configurable = WorkflowConfiguration.from_runnable_config(config)
    research_model_config = {
        "model": configurable.research_model,
        "max_tokens": configurable.research_model_max_tokens,
    }
    research_model = configurable_model.with_config(research_model_config).with_structured_output(Queries).with_retry(stop_after_attempt=configurable.max_structured_output_retries)

    try:
        query_results = await asyncio.wait_for(
            research_model.ainvoke(
                [HumanMessage(
                    content=query_writer_instructions.format(
                        messages=get_buffer_string(messages),
                        number_of_queries=configurable.number_of_queries,
                        query_history="\n".join([f"{i+1}. {q}" for i, q in enumerate(historical_queries)]),
                        context="\n".join(notes),
                        today=get_today_str())
                    )
                ]
            ),
        timeout=30.0)

        query_list = [query.search_query for query in query_results.queries]
        if not query_list:
            return Command(
                goto="__end__"
            )
        return Command(
            goto="upfront_search",
            update={
                "current_queries": query_list,
            }
        )
    except Exception as e:
        print(f"Error generating queries: {e}")
        return Command(
            goto="generate_upfront_queries",
            update={"search_attempts": state.get("search_attempts", 0) + 1}
        )


async def upfront_search(state: UpfrontWebResearcherState, config: RunnableConfig) -> Command[Literal["generate_upfront_queries", "__end__"]]:
    current_queries = state.get("current_queries", [])
    search_attempts = state.get("search_attempts", 0)
    configurable = WorkflowConfiguration.from_runnable_config(config)
    try:
        search_results = await select_and_execute_search(
            get_config_value(configurable.search_api),
            current_queries,
            get_search_params(get_config_value(configurable.search_api), configurable.search_api_config or {})
        )
        # TODO: Potentially format the search results into a more readable and specific format before adding to notes.
        # If we have run out of search attempts, we go to generate an outline. If not, we go back to the query generator.
        return Command(
            goto= "generate_upfront_queries" if search_attempts < configurable.max_search_depth else "generate_outline",
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
            goto= "upfront_search" if search_attempts < configurable.max_search_depth else "__end__",
            update={"search_attempts": search_attempts + 1}
        )
    
upfront_web_researcher_builder = StateGraph(UpfrontWebResearcherState)
upfront_web_researcher_builder.add_node("generate_upfront_queries", generate_upfront_queries)
upfront_web_researcher_builder.add_node("upfront_search", upfront_search)
upfront_web_researcher_builder.add_edge(START, "generate_upfront_queries")
upfront_web_researcher_builder.add_edge("generate_upfront_queries", "upfront_search")
upfront_web_researcher = upfront_web_researcher_builder.compile()


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
general_researcher_builder.add_node("upfront_web_researcher", upfront_web_researcher)
general_researcher_builder.add_node("upfront_model_provider_researcher", upfront_model_provider_researcher)
general_researcher_builder.add_node("generate_outline", generate_outline)
general_researcher_builder.add_node("human_feedback", human_feedback)
general_researcher_builder.add_node("final_report_generation", final_report_generation)
general_researcher_builder.add_conditional_edges(START, initial_router, {
    "upfront_web_researcher": "upfront_web_researcher",
    "upfront_model_provider_researcher": "upfront_model_provider_researcher"
})
general_researcher_builder.add_edge("upfront_web_researcher", "generate_outline")
general_researcher_builder.add_edge("upfront_model_provider_researcher", "generate_outline")
general_researcher_builder.add_edge("final_report_generation", END)

general_researcher = general_researcher_builder.compile()