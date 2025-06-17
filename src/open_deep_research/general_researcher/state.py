from typing import Annotated, List, Literal, Optional
from pydantic import BaseModel, Field
from dataclasses import dataclass
import operator
from langgraph.graph import MessagesState
from langchain_core.messages import MessageLikeRepresentation

# Structured output

class SearchQuery(BaseModel):
    search_query: str = Field(None, description="Query for web search.")

class Queries(BaseModel):
    queries: List[SearchQuery] = Field(
        description="List of search queries.",
    )

class Section(BaseModel):
    name: str = Field(
        description="Name for this section of the report.",
    )
    description: str = Field(
        description="Brief overview of the main topics and concepts to be covered in this section.",
    )
    content: str = Field(
        description="The content of the section."
    )

class Outline(BaseModel):
    outline: List[Section] = Field(
        description="List of sections for the report.",
    )

@dataclass
class SearchSource:
    """Represents a source found during web search."""
    url: str
    title: str
    content: str
    search_query: str
    iteration: int

@dataclass 
class ReflectionResult:
    """Result of reflection phase analyzing research completeness."""
    is_satisfied: bool
    knowledge_gaps: List[str]
    suggested_queries: List[str] 
    reasoning: str

# State Definitions

class ResearchUnitState(MessagesState):
    notes: Annotated[Optional[list[str]], operator.add]
    research_iterations: Optional[int] = 0
    research_messages: Optional[list[MessageLikeRepresentation]]
    collected_sources: Annotated[Optional[list[SearchSource]], operator.add]

class GeneralResearcherStateInput(MessagesState):
    """InputState is only 'messages'"""

class GeneralResearcherStateOutput(MessagesState):
    final_report: str

class GeneralResearcherState(MessagesState):
    final_report: str
    notes: Annotated[list[str], operator.add]
    collected_sources: Optional[list[SearchSource]]
    feedback_on_outline: list[str]
    historical_queries: Annotated[list[str], operator.add]
    current_queries: list[str]
    search_attempts: int = 0
    outline: list[Section]
    completed_sections: Annotated[list, operator.add]

class SectionState(MessagesState):
    section: Section
    notes: list[str]
    feedback_on_section: str
    search_iterations: int
    section_search_queries: list[SearchQuery]
    section_notes: list[str]
    completed_sections: list[Section]

class SectionOutputState(MessagesState):
    completed_sections: list[Section]