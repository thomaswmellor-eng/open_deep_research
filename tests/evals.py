from typing import cast

from pydantic import BaseModel, Field
from langchain.chat_models import init_chat_model

eval_model = init_chat_model("anthropic:claude-sonnet-4-20250514")


RESPONSE_CRITERIA_SYSTEM_PROMPT = """
You are evaluating the quality of a research report. Please assess the report against the following criteria, being especially strict about section relevance.

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


class CriteriaGrade(BaseModel):
    """Score the response against specific criteria."""
    grade: int = Field(description="Integer score 1-5 showing whether the response meets the provided criteria (1 = doesn't meet at all, 5 = meets all criteria).")
    justification: str = Field(description="The justification for the grade and score, including specific examples from the response.")


def eval_report_quality(inputs: dict, outputs: dict):
    query = inputs["messages"][0]["content"]
    final_report = outputs["messages"][0]["content"]

    eval_result = cast(CriteriaGrade, eval_model.with_structured_output(CriteriaGrade).invoke([
        {"role": "system", "content": RESPONSE_CRITERIA_SYSTEM_PROMPT},
        {"role": "user", "content": f"""User input: {query}\n\nReport: \n\n{final_report}\n\nEvaluate whether the report meets the criteria and provide detailed justification for your evaluation."""}
    ]))
    # normalize to 0-1
    return eval_result.grade / 5