import operator
from typing import Annotated, List, Literal, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages
from pydantic import BaseModel, Field


class Section(BaseModel):
    name: str = Field(
        description="Name for this section of the report.",
    )
    description: str = Field(
        description="Brief overview of the main topics and concepts to be covered in this section.",
    )
    research: bool = Field(
        description="Whether to perform web research for this section of the report."
    )
    content: str = Field(description="The content of the section.")


class Sections(BaseModel):
    sections: List[Section] = Field(
        description="Sections of the report.",
    )


class SearchQuery(BaseModel):
    search_query: str = Field(None, description="Query for web search.")


class SearchQueries(BaseModel):
    queries: List[SearchQuery] = Field(
        description="List of search queries.",
    )


class GenerateOrRefineReport(BaseModel):
    """Generate or refine the research plan used for report."""

    topic: str = Field(
        description="Topic for the report. Prefer using the exact user query for topic."
    )
    feedback_on_report_plan: str | None = Field(
        description="Feedback to be used when modifying an existing report plan"
    )


class StartResearch(BaseModel):
    """Start the research"""

    pass


class Feedback(BaseModel):
    grade: Literal["pass", "fail"] = Field(
        description="Evaluation result indicating whether the response meets requirements ('pass') or needs revision ('fail')."
    )
    follow_up_queries: List[SearchQuery] = Field(
        description="List of follow-up search queries.",
    )


class ReportStateInput(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


class ReportStateOutput(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


class ReportState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    topic: str  # Report topic
    sections: list[Section]  # List of report sections
    completed_sections: Annotated[list[Section], operator.add]  # Send()
    report_sections_from_research: (
        str  # String of any completed sections from research to write final sections
    )


class SectionState(TypedDict):
    topic: str  # Report topic
    section: Section  # Report section
    search_iterations: int  # Number of search iterations done
    search_queries: list[SearchQuery]  # List of search queries
    source_str: str  # String of formatted source content from web search
    report_sections_from_research: (
        str  # String of any completed sections from research to write final sections
    )
    completed_sections: list[
        Section
    ]  # Final key we duplicate in outer state for Send() API


class SectionOutputState(TypedDict):
    completed_sections: list[
        Section
    ]  # Final key we duplicate in outer state for Send() API
