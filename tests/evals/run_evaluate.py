from langsmith import Client
from tests.evals.evaluators import eval_overall_quality, eval_relevance, eval_structure
from dotenv import load_dotenv
import asyncio
from typing import Literal
from langchain_core.messages import MessageLikeRepresentation
from open_deep_research.multi_agent import supervisor_builder
from open_deep_research.workflow.workflow import builder
from langgraph.types import Command
from langgraph.checkpoint.memory import MemorySaver
import uuid

load_dotenv("../.env")

client = Client()

# TODO: Configure these variables
dataset_name = "ODR: Workflow Examples"
evaluators = [eval_overall_quality, eval_relevance, eval_structure]
process_search_results = "summarize"
include_source = False
summarization_model = "claude-3-5-haiku-latest"
summarization_model_provider = "anthropic"
supervisor_model = "claude-3-5-sonnet-latest"
researcher_model = "claude-3-5-sonnet-latest"
writer_model = "claude-3-5-sonnet-latest"
writer_model_provider = "anthropic"
clarify_with_user = False
sections_user_approval = False


async def generate_report_multi_agent(
    messages: list[MessageLikeRepresentation],
):
    """Generate a report using the open deep research multi-agent architecture"""
    graph = supervisor_builder.compile()
    config = {"configurable": {}}
    if include_source:
        config["configurable"]["include_source_str"] = True
    if process_search_results:
        config["configurable"]["process_search_results"] = process_search_results
    config["configurable"]["summarization_model"] = summarization_model
    config["configurable"]["summarization_model_provider"] = summarization_model_provider
    config["configurable"]["supervisor_model"] = supervisor_model
    config["configurable"]["researcher_model"] = researcher_model

    final_state = await graph.ainvoke(
        # TODO: This is a hack, find workaround at some point
        {"messages": messages + [{"role": "user", "content": "Generate the report now and don't ask any more follow-up questions"}]},
        config
    )
    return {
        "messages": [
            {"role": "assistant", "content": final_state["final_report"]}
        ]
    }

async def generate_report_workflow(
    messages: list[MessageLikeRepresentation],
):
    """Generate a report using the open deep research workflow"""
    graph = builder.compile(checkpointer=MemorySaver())
    config = {
        "configurable": {
            "thread_id": str(uuid.uuid4()),
        }
    }
    config["configurable"]["include_source_str"] = include_source
    config["configurable"]["process_search_results"] = process_search_results
    config["configurable"]["summarization_model"] = summarization_model
    config["configurable"]["summarization_model_provider"] = summarization_model_provider
    config["configurable"]["writer_model"] = writer_model
    config["configurable"]["writer_model_provider"] = writer_model_provider
    config["configurable"]["clarify_with_user"] = clarify_with_user
    config["configurable"]["sections_user_approval"] = sections_user_approval

    final_state =await graph.ainvoke(
        {"messages": messages},
        config
    )
    return final_state

async def target(inputs: dict):
    # return await generate_report_multi_agent(inputs["messages"])
    return await generate_report_workflow(inputs["messages"])

async def main():
    return await client.aevaluate(
        target,
        data=client.list_examples(dataset_name=dataset_name, splits=["split-1"]),
        evaluators=evaluators,
        experiment_prefix=f"ODR: Workflow - SM:{summarization_model} WM:{writer_model} PSR:{process_search_results} #",
        max_concurrency=1,
        metadata={
            "process_search_results": process_search_results,
            "include_source": include_source,
            "clarify_with_user": clarify_with_user,
            "sections_user_approval": sections_user_approval,
            "summarization_model": summarization_model,
            "writer_model": writer_model,
        },
    )

if __name__ == "__main__":
    results = asyncio.run(main())
    print(results)