from langsmith import Client
from tests.evals.evaluators import eval_overall_quality, eval_relevance, eval_structure
from dotenv import load_dotenv
import asyncio
from langchain_core.messages import MessageLikeRepresentation
from open_deep_research.mcp_workflow.mcp_workflow import mcp_workflow_builder
from open_deep_research.multi_agent import supervisor_builder
from open_deep_research.graph import builder
from langgraph.checkpoint.memory import MemorySaver
import uuid

load_dotenv("../.env")

client = Client()

# TODO: Configure these variables
dataset_name = "ODR: Comprehensive Test"
evaluators = [eval_overall_quality, eval_relevance, eval_structure]

# Target function for open_deep_research/graph.py
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
    config["configurable"]["search_api"] = "tavily"
    config["configurable"]["process_search_results"] = "summarize"
    config["configurable"]["include_source_str"] = False
    config["configurable"]["summarization_model"] = "claude-3-5-haiku-latest"
    config["configurable"]["summarization_provider"] = "anthropic"
    config["configurable"]["max_structured_output_retries"] = 3
    config["configurable"]["number_of_queries"] = 3
    config["configurable"]["max_search_depth"] = 3
    config["configurable"]["planner_model"] = "claude-sonnet-4-20250514"
    config["configurable"]["planner_provider"] = "anthropic"
    config["configurable"]["writer_model"] = "claude-sonnet-4-20250514"
    config["configurable"]["writer_provider"] = "anthropic"

    final_state =await graph.ainvoke(
        {"messages": messages},
        config
    )
    return final_state

# Target function for open_deep_research/multi_agent.py
async def generate_report_multi_agent(
    messages: list[MessageLikeRepresentation],
):
    """Generate a report using the open deep research multi-agent architecture"""
    graph = supervisor_builder.compile()
    config = {
        "configurable": {
            "thread_id": str(uuid.uuid4()),
        }
    }
    config["configurable"]["search_api"] = "tavily"
    config["configurable"]["process_search_results"] = "summarize"
    config["configurable"]["include_source_str"] = False
    config["configurable"]["summarization_model"] = "claude-3-5-haiku-latest"
    config["configurable"]["summarization_provider"] = "anthropic"
    config["configurable"]["number_of_queries"] = 3
    config["configurable"]["supervisor_model"] = "anthropic:claude-3-5-haiku-latest"
    config["configurable"]["researcher_model"] = "anthropic:claude-sonnet-4-20250514"
    config["configurable"]["ask_for_clarification"] = False

    final_state = await graph.ainvoke(
        {"messages": messages + [{"role": "user", "content": "Generate the report now and don't ask any more follow-up questions"}]},
        config
    )
    return {
        "messages": [
            {"role": "assistant", "content": final_state["final_report"]}
        ]
    }

# Target function for open_deep_research/mcp_workflow/mcp_workflow.py
async def generate_report_mcp_workflow(
    messages: list[MessageLikeRepresentation],
):
    """Generate a report using the open deep research general researcher"""
    graph = mcp_workflow_builder.compile(checkpointer=MemorySaver())
    config = {
        "configurable": {
            "thread_id": str(uuid.uuid4()),
        }
    }
    # Configure these variables
    config["configurable"]["max_structured_output_retries"] = 3
    config["configurable"]["clarify_with_user"] = False
    config["configurable"]["search_api"] = "tavily"     # NOTE: We use Tavily to stay consistent
    config["configurable"]["search_api_config"] = None
    config["configurable"]["max_search_depth"] = 4
    config["configurable"]["summarization_model"] = "anthropic:claude-3-5-haiku-latest"
    config["configurable"]["summarization_model_max_tokens"] = 10000
    config["configurable"]["research_model"] = "anthropic:claude-sonnet-4-20250514"
    config["configurable"]["research_model_max_tokens"] = 10000
    config["configurable"]["reflection_model"] = "anthropic:claude-sonnet-4-20250514"
    config["configurable"]["reflection_model_max_tokens"] = 10000
    config["configurable"]["outliner_model"] = "openai:gpt-4.1"
    config["configurable"]["outliner_model_max_tokens"] = 10000
    config["configurable"]["final_report_model"] = "anthropic:claude-sonnet-4-20250514"
    config["configurable"]["final_report_model_max_tokens"] = 10000
    # NOTE: We do not use MCP tools to stay consistent
    final_state = await graph.ainvoke(
        {"messages": messages},
        config
    )
    return final_state

async def target(inputs: dict):
    # return await generate_report_workflow(inputs["messages"])
    # return await generate_report_multi_agent(inputs["messages"])
    return await generate_report_mcp_workflow(inputs["messages"])

async def main():
    return await client.aevaluate(
        target,
        data=dataset_name,
        evaluators=evaluators,
        experiment_prefix=f"MCP Workflow - Tavily #",
        max_concurrency=1,
    )

if __name__ == "__main__":
    results = asyncio.run(main())
    print(results)