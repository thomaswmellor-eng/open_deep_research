from typing import cast

from pydantic import BaseModel, Field
from langchain_anthropic import ChatAnthropic

eval_model = ChatAnthropic(
    model="claude-sonnet-4-0",
    betas=["extended-cache-ttl-2025-04-11"],
)


RESPONSE_QUALITY_PROMPT = """You are evaluating the quality of a research report. Please assess the report against the following criteria, being especially strict about section relevance.

1. Topic Relevance (Overall): Does the report directly address the user's input topic thoroughly?

2. Section Relevance (Critical): CAREFULLY assess each individual section for relevance to the main topic:
   - Identify each section by its ## header
   - For each section, determine if it is directly relevant to the primary topic
   - Flag any sections that seem tangential, off-topic, or only loosely connected to the main topic
   - A high-quality report should have NO irrelevant sections

3. Structure and Flow: Do the sections flow logically from one to the next, creating a cohesive narrative?

4. Introduction Quality: Does the introduction effectively provide context and set up the scope of the report?

5. Conclusion Quality: Does the conclusion meaningfully summarize key findings and insights from the report?

6. Structural Elements: Does the report use structural elements (e.g., tables, lists) to effectively convey information?

7. Section Headers: Are section headers properly formatted with Markdown (# for title, ## for sections, ### for subsections)?

8. Citations: Does the report properly cite sources in each main body section?

9. Overall Quality: Is the report well-researched, accurate, and professionally written?

Evaluation Instructions:
- Be STRICT about section relevance - ALL sections must clearly connect to the primary topic
- A report with even ONE irrelevant section should be considered flawed
- You must individually mention each section by name and assess its relevance
- Provide specific examples from the report to justify your evaluation for each criterion
- The report fails if any sections are irrelevant to the main topic, regardless of other qualities
"""


class QualityScore(BaseModel):
    """Score the response quality against specific criteria."""
    # TODO: return grades here
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


def eval_report_quality(inputs: dict, outputs: dict):
    query = _format_input_query(inputs)
    final_report = outputs["messages"][0]["content"]

    user_input_content = f"""User input: {query}\n\nReport: \n\n{final_report}\n\nEvaluate whether the report meets the criteria and provide detailed justification for your evaluation."""
    if isinstance(eval_model, ChatAnthropic):
        user_input_content = [{
            "type": "text",
            "text": user_input_content,
            "cache_control": {"type": "ephemeral", "ttl": "1h"}
        }]

    eval_result = cast(QualityScore, eval_model.with_structured_output(QualityScore).invoke([
        {"role": "system", "content": RESPONSE_QUALITY_PROMPT},
        {"role": "user", "content": user_input_content}
    ]))
    # normalize to 0-1
    return {"key": "eval_report_quality", "score": eval_result.score / 5, "comment": eval_result.reasoning}


def eval_groundedness(inputs: dict, outputs: dict):
    report = outputs["messages"][0]["content"]
    context = outputs["context"]

    user_input_content = GROUNDEDNESS_PROMPT.format(context=context, report=report)
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
    return {"key": "eval_groundedness", "score": eval_result.score / 5, "comment": eval_result.reasoning}