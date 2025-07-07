from typing import Annotated, List, Optional
from pydantic import BaseModel, Field
import operator
from langgraph.graph import MessagesState
from langchain_core.messages import MessageLikeRepresentation

###################
# Structured Outputs
###################
class Summary(BaseModel):
    summary: str
    key_excerpts: str

class ClarifyWithUser(BaseModel):
    need_clarification: bool = Field(
        description="Whether the user needs to be asked a clarifying question.",
    )
    question: str = Field(
        description="A question to ask the user to clarify the report scope",
    )

class ResearchSynthesis(BaseModel):
    full_report: str = Field(
        description="A fully comprehensive report on the information that the researcher has gathered on this topic in the prior messages.",
    )

class ResearchQuestion(BaseModel):
    research_question: str = Field(
        description="A research question that will be used to guide the research.",
    )

class SearchSource(BaseModel):
    """Represents a source found during web search."""
    url: str
    title: str
    content: str
    search_query: str
    iteration: int

class LeadResearcherReflection(BaseModel):
    """Result of reflection phase analyzing research completeness."""
    topics_to_research: List[str] = Field(
        description="Detailed list of topics to research. Each topic should be discrete and specific. You can return only a single topic if there is only one topic to research. You can also return multiple topics if allowed by the configuration, or topics if is_satisfied is true.",
    )
    reasoning: str = Field(
        description="Reasoning for the reflection.",
    )
    is_satisfied: bool = Field(
        description="Whether the research is complete to answer the user's question.",
    )

###################
# State Definitions
###################
def override_reducer(current_value, new_value):
    if isinstance(new_value, dict) and new_value.get("type") == "override":
        return new_value.get("value", new_value)
    else:
        return operator.add(current_value, new_value)
    
class SubResearcherState:
    specific_topic: str
    research_messages: list[MessageLikeRepresentation]
    notes: list[str]

class GeneralResearcherStateInput(MessagesState):
    """InputState is only 'messages'"""

class GeneralResearcherStateOutput(MessagesState):
    final_report: str

class GeneralResearcherState(MessagesState):
    research_question: Optional[str]
    notes: Annotated[list[str], override_reducer]
    search_attempts: int = 0
    final_report: str
