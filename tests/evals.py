from typing import cast, Literal
import uuid

from pydantic import BaseModel, Field
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import MessageLikeRepresentation
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from open_deep_research.graph import builder
from open_deep_research.multi_agent import supervisor_builder
from open_deep_research.utils import get_today_str

eval_model = ChatAnthropic(
    model="claude-sonnet-4-0",
    betas=["extended-cache-ttl-2025-04-11"],
)


RELEVANCE_PROMPT = """You are evaluating the relevance of a research report to the user's input topic. Please assess the report against the following criteria, being especially strict about section relevance.

1. Topic Relevance (Overall): Does the report directly address the user's input topic thoroughly?

2. Section Relevance (Critical): CAREFULLY assess each individual section for relevance to the main topic:
   - Identify each section by its ## header
   - For each section, determine if it is directly relevant to the primary topic
   - Flag any sections that seem tangential, off-topic, or only loosely connected to the main topic
   - A high-quality report (score 5) should have NO irrelevant sections

3. Introduction Quality: Does the introduction effectively provide context and set up the scope of the report?

4. Conclusion Quality: Does the conclusion meaningfully summarize key findings and insights from the report?

5. Citations: Does the report properly cite sources in each main body section?

6. Overall Quality: Is the report well-researched, accurate, and professionally written?

Evaluation Instructions:
- Be STRICT about section relevance - ALL sections must clearly connect to the primary topic
- You must individually mention each section by name and assess its relevance
- Provide specific examples from the report to justify your evaluation for each criterion
- A report that is not relevant to the user's input topic should be scored 1
- A report passing all of the above criteria should be scored 5

Today is {today}
"""

class RelevanceScore(BaseModel):
    """Score the response relevance against specific criteria."""
    reasoning: str = Field(description="The reason for the score, including specific examples from the response.")
    score: int = Field(description="Integer score 1-5 showing whether the response meets the provided criteria (1 = doesn't meet at all, 5 = meets all criteria).")

STRUCTURE_PROMPT = """You are evaluating the structure and flow of a research report. Please assess the report against the following criteria:

1. Structure and Flow: Do the sections flow logically from one to the next, creating a cohesive narrative?
2. Structural Elements: Does the report use structural elements (e.g., headers, tables, lists) to effectively convey information?
3. Section Headers: Are section headers properly formatted with Markdown (# for title, ## for sections, ### for subsections)?
4. Citations: Does the report include citations with source URLs?

Today is {today}
"""

class StructureScore(BaseModel):
    """Score the response structure against specific criteria."""
    reasoning: str = Field(description="The reason for the score, including specific examples from the response.")
    score: int = Field(description="Integer score 1-5 showing whether the response meets the provided criteria (1 = doesn't meet at all, 5 = meets all criteria).")


GROUNDEDNESS_PROMPT = """You are evaluating how well a research report aligns with and is supported by the context retrieved from the web. Your evaluation should focus on the following criteria:

<Rubric>
A well-grounded report should:
- Make claims that are directly supported by the retrieved context
- Stay within the scope of information provided in the context
- Maintain the same meaning and intent as the source material
- Not introduce external facts or unsupported assertions outside of basic facts (2 + 2 = 4)

An ungrounded report:
- Makes claims without support from the context
- Contradicts the retrieved information
- Includes speculation or external knowledge outside of basic facts
- Distorts or misrepresents the context
</Rubric>

<Instruction>
- Compare the output against the retrieved context carefully
- Identify claims, statements, and assertions in the output
- For each claim, locate supporting evidence in the context
- Check for:
  - Direct statements from context
  - Valid inferences from context
  - Unsupported additions
  - Contradictions with context

- Note any instances where the output:
  - Extends beyond the context
  - Combines information incorrectly
  - Makes logical leaps
</Instruction>

<Reminder>
- Focus solely on alignment with provided context
- Ignore whether external knowledge suggests different facts
- Consider both explicit and implicit claims
- Provide specific examples of grounded/ungrounded content
- Remember that correct grounding means staying true to the context, even if context conflicts with common knowledge
</Reminder>

<context>
{context}
</context>

<report>
{report}
</report>

Today is {today}
"""


class GroundednessScore(BaseModel):
    """Score the response groundedness against specific criteria."""
    reasoning: str = Field(description="The reason for the score, including specific examples from the response.")
    score: int = Field(description="Integer score 1-5 showing whether the response meets the provided criteria (1 = doesn't meet at all, 5 = meets all criteria).")


def _format_input_query(inputs: dict) -> str:
    messages = inputs["messages"]
    if len(messages) == 1:
        return messages[0]["content"]

    role_to_string_format_map = {
        "user": "<user_input>\n{content}\n</user_input>",
        "assistant": "<assistant_follow_up>\n{content}\n</assistant_follow_up>",
    }

    return "\n\n".join([role_to_string_format_map[message["role"]].format(content=message["content"]) for message in messages])


def eval_relevance(inputs: dict, outputs: dict):
    query = _format_input_query(inputs)
    final_report = outputs["messages"][0]["content"]

    user_input_content = f"""User input: {query}\n\nReport: \n\n{final_report}\n\nEvaluate whether the report meets the criteria and provide detailed justification for your evaluation."""
    if isinstance(eval_model, ChatAnthropic):
        user_input_content = [{
            "type": "text",
            "text": user_input_content,
            "cache_control": {"type": "ephemeral", "ttl": "1h"}
        }]

    eval_result = cast(RelevanceScore, eval_model.with_structured_output(RelevanceScore).invoke([
        {"role": "system", "content": RELEVANCE_PROMPT.format(today=get_today_str())},
        {"role": "user", "content": user_input_content}
    ]))
    # normalize to 0-1
    return {"key": "relevance_score", "score": eval_result.score / 5, "comment": eval_result.reasoning}


def eval_structure(inputs: dict, outputs: dict):
    query = _format_input_query(inputs)
    final_report = outputs["messages"][0]["content"]

    user_input_content = f"""User input: {query}\n\nReport: \n\n{final_report}\n\nEvaluate whether the report meets the criteria and provide detailed justification for your evaluation."""
    if isinstance(eval_model, ChatAnthropic):
        user_input_content = [{
            "type": "text",
            "text": user_input_content,
            "cache_control": {"type": "ephemeral", "ttl": "1h"}
        }]

    eval_result = cast(StructureScore, eval_model.with_structured_output(StructureScore).invoke([
        {"role": "system", "content": STRUCTURE_PROMPT.format(today=get_today_str())},
        {"role": "user", "content": user_input_content}
    ]))
    # normalize to 0-1
    return {"key": "structure_score", "score": eval_result.score / 5, "comment": eval_result.reasoning}


def eval_groundedness(inputs: dict, outputs: dict):
    report = outputs["messages"][0]["content"]
    context = outputs["context"]

    user_input_content = GROUNDEDNESS_PROMPT.format(context=context, report=report, today=get_today_str())
    if isinstance(eval_model, ChatAnthropic):
        user_input_content = [{
            "type": "text",
            "text": user_input_content,
            "cache_control": {"type": "ephemeral", "ttl": "1h"}
        }]

    eval_result = cast(GroundednessScore, eval_model.with_structured_output(GroundednessScore).invoke([
        {"role": "user", "content": user_input_content},
    ]))
    # normalize to 0-1
    return {"key": "groundedness_score", "score": eval_result.score / 5, "comment": eval_result.reasoning}


async def generate_report_workflow(
    query: str,
    process_search_results: Literal["summarize", "split_and_rerank"] | None = None,
    include_source: bool = True
):
    """Generate a report using the open deep research workflow"""
    graph = builder.compile(checkpointer=MemorySaver())
    config = {
        "configurable": {
            "thread_id": str(uuid.uuid4()),
        }
    }
    if include_source:
        config["configurable"]["include_source_str"] = True

    if process_search_results:
        config["configurable"]["process_search_results"] = process_search_results

    # Run the graph until the interruption
    await graph.ainvoke(
        {"topic": query},
        config
    )
    # Pass True to approve the report plan
    final_state = await graph.ainvoke(Command(resume=True), config)
    return final_state


async def generate_report_multi_agent(
    messages: list[MessageLikeRepresentation],
    process_search_results: Literal["summarize", "split_and_rerank"] | None = None,
    include_source: bool = True
):
    """Generate a report using the open deep research multi-agent architecture"""
    graph = supervisor_builder.compile()
    config = {"configurable": {}}
    if include_source:
        config["configurable"]["include_source_str"] = True

    if process_search_results:
        config["configurable"]["process_search_results"] = process_search_results

    final_state = await graph.ainvoke(
        # this is a hack
        {"messages": messages + [{"role": "user", "content": "Generate the report now and don't ask any more follow-up questions"}]},
        config
    )
    return final_state