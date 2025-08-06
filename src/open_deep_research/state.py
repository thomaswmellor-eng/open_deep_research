from typing import Annotated, Optional, List
from pydantic import BaseModel, Field
import operator
from langgraph.graph import MessagesState
from langchain_core.messages import MessageLikeRepresentation
from typing_extensions import TypedDict

###################
# Structured Outputs
###################
class ConductResearch(BaseModel):
    """Call this tool to conduct research on a specific topic."""
    research_topic: str = Field(
        description="The topic to research. Should be a single topic, and should be described in high detail (at least a paragraph).",
    )

class ResearchComplete(BaseModel):
    """Call this tool to indicate that the research is complete."""

class Summary(BaseModel):
    summary: str
    key_excerpts: str

class ClarifyWithUser(BaseModel):
    need_clarification: bool = Field(
        description="Whether the user needs to be asked a clarifying question.",
    )
    question: str = Field(
        description="A question to ask the user to clarify their travel preferences",
    )
    verification: str = Field(
        description="Verify message that we will start research after the user has provided the necessary information.",
    )
    travel_profile_complete: bool = Field(
        description="Whether we have gathered all necessary travel information from the user",
    )

class ResearchQuestion(BaseModel):
    research_brief: str = Field(
        description="A research question that will be used to guide the research.",
    )

class DestinationSuggestion(BaseModel):
    reasoning: str = Field(
        description="Natural, conversational explanation of suggested destinations and why they match the user's preferences. Include destinations, brief reasoning, and any relevant highlights in a flowing, friendly tone.",
    )
    suggested_destinations: List[str] = Field(
        description="List of suggested destinations that match the user's preferences",
        default_factory=list
    )
    budget_considerations: str = Field(
        description="Budget considerations for each suggested destination",
        default=""
    )
    cultural_highlights: str = Field(
        description="Cultural characteristics and highlights of suggested destinations",
        default=""
    )
    best_time_to_visit: str = Field(
        description="Best time to visit each suggested destination",
        default=""
    )


###################
# State Definitions
###################

def override_reducer(current_value, new_value):
    if isinstance(new_value, dict) and new_value.get("type") == "override":
        return new_value.get("value", new_value)
    else:
        return operator.add(current_value, new_value)
    
class AgentInputState(MessagesState):
    """InputState is only 'messages'"""

class AgentState(MessagesState):
    supervisor_messages: Annotated[list[MessageLikeRepresentation], override_reducer]
    research_brief: Optional[str]
    raw_notes: Annotated[list[str], override_reducer] = []
    notes: Annotated[list[str], override_reducer] = []
    final_report: str
    travel_profile: Annotated[dict, override_reducer] = {}

class SupervisorState(TypedDict):
    supervisor_messages: Annotated[list[MessageLikeRepresentation], override_reducer]
    research_brief: str
    notes: Annotated[list[str], override_reducer] = []
    research_iterations: int = 0
    raw_notes: Annotated[list[str], override_reducer] = []

class ResearcherState(TypedDict):
    researcher_messages: Annotated[list[MessageLikeRepresentation], operator.add]
    tool_call_iterations: int = 0
    research_topic: str
    compressed_research: str
    raw_notes: Annotated[list[str], override_reducer] = []

class ResearcherOutputState(BaseModel):
    compressed_research: str
    raw_notes: Annotated[list[str], override_reducer] = []