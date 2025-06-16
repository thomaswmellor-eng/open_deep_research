from dataclasses import dataclass, fields, field
from typing import Any, Dict, Literal, Optional
from langchain_core.runnables import RunnableConfig
import os
from enum import Enum
from open_deep_research.general_researcher.state import SearchSource
from google.ai.generativelanguage_v1beta.types import Tool as GenAITool

class SearchAPI(Enum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    # GOOGLE_GENAI = "google_genai"
    TAVILY = "tavily"
    DUCKDUCKGO = "duckduckgo"

class ModelsWithWebSearch(Enum):
    ANTHROPIC_CLAUDE_3_7_SONNET_LATEST = "anthropic:claude-3-7-sonnet-latest"
    ANTHROPIC_CLAUDE_4_0_SONNET_LATEST = "anthropic:claude-sonnet-4-20250514"
    ANTHROPIC_CLAUDE_4_0_OPUS_LATEST = "anthropic:claude-opus-4-20250514"
    ANTHROPIC_CLAUDE_3_5_HAIKU_LATEST = "anthropic:claude-3-5-haiku-latest"
    ANTHROPIC_CLAUDE_3_5_SONNET_LATEST = "anthropic:claude-3-5-sonnet-latest"
    OPENAI_GPT_4_1 = "openai:gpt-4.1"
    OPENAI_GPT_4_1_MINI = "openai:gpt-4.1-mini"
    # GEMINI_GEMINI_2_5_FLASH_PREVIEW_05_20 = "google_genai:gemini-2.5-flash-preview-05-20"
    # GEMINI_GEMINI_2_5_PRO_PREVIEW_06_05 = "google_genai:gemini-2.5-pro-preview-06-05"
    # GEMINI_GEMINI_2_0_FLASH = "google_genai:gemini-2.0-flash"
    # GEMINI_GEMINI_2_0_FLASH_LITE = "google_genai:gemini-2.0-flash-lite"
    # GEMINI_GEMINI_1_5_FLASH = "google_genai:gemini-1.5-flash"

MODELS_WITH_WEB_SEARCH = {
    SearchAPI.ANTHROPIC: {
        ModelsWithWebSearch.ANTHROPIC_CLAUDE_3_7_SONNET_LATEST,
        ModelsWithWebSearch.ANTHROPIC_CLAUDE_4_0_SONNET_LATEST,
        ModelsWithWebSearch.ANTHROPIC_CLAUDE_4_0_OPUS_LATEST,
        ModelsWithWebSearch.ANTHROPIC_CLAUDE_3_5_HAIKU_LATEST,
        ModelsWithWebSearch.ANTHROPIC_CLAUDE_3_5_SONNET_LATEST,
    },
    SearchAPI.OPENAI: {
        ModelsWithWebSearch.OPENAI_GPT_4_1,
        ModelsWithWebSearch.OPENAI_GPT_4_1_MINI,
    },
    # SearchAPI.GOOGLE_GENAI: {
    #     ModelsWithWebSearch.GEMINI_GEMINI_2_5_FLASH_PREVIEW_05_20,
    #     ModelsWithWebSearch.GEMINI_GEMINI_2_5_PRO_PREVIEW_06_05,
    #     ModelsWithWebSearch.GEMINI_GEMINI_2_0_FLASH,
    #     ModelsWithWebSearch.GEMINI_GEMINI_2_0_FLASH_LITE,
    #     ModelsWithWebSearch.GEMINI_GEMINI_1_5_FLASH,
    # },
}
    
def call_source_extractor(response, iteration, search_api: SearchAPI):
    if search_api == SearchAPI.ANTHROPIC:
        return collect_sources_anthropic(response, iteration)
    elif search_api == SearchAPI.OPENAI:
        return collect_sources_openai(response, iteration)
    else:
        return []
    
def extract_content_from_response(response):
    text_blocks = []
    if hasattr(response, 'content'):
        for content_block in response.content:
            if "type" in content_block and content_block["type"] == "text":
                text_blocks.append(content_block["text"])
    return "\n\n".join(text_blocks) if text_blocks else ""
    
def collect_sources_anthropic(response, iteration):
    collected_sources = []
    if hasattr(response, 'content'):    
        for content_block in response.content:
            if "type" in content_block:
                if content_block["type"] == 'web_search_tool_result':
                    if "content" in content_block:
                        for search_result in content_block["content"]:
                            if "type" in search_result and search_result["type"] == 'web_search_result':
                                source = SearchSource(
                                    url=search_result.get('url', ''),
                                    title=search_result.get('title', ''),
                                    content=f"Source found in iteration {iteration}",
                                    search_query=f"iteration_{iteration}",
                                    iteration=iteration
                                )
                                collected_sources.append(source)
                
                elif content_block["type"] == 'text':
                    if "citations" in content_block:
                        for citation in content_block["citations"]:
                            if "url" in citation and "title" in citation:
                                cited_content = citation.get('cited_text', '')
                                if not cited_content:
                                    cited_content = f"Citation from {citation['title']}"
                                source = SearchSource(
                                    url=citation["url"],
                                    title=citation["title"],
                                    content=cited_content[:1000],
                                    search_query=f"citation_iteration_{iteration}",
                                    iteration=iteration
                                )
                                collected_sources.append(source)
    return collected_sources

def collect_sources_openai(response, iteration):
    """Extract sources from OpenAI response format"""
    collected_sources = []
    if hasattr(response, 'content'):
        for content_block in response.content:
            if "type" in content_block and content_block["type"] == "text":
                if "annotations" in content_block:
                    for annotation in content_block["annotations"]:
                        if annotation.get("type") == "url_citation":
                            # Extract the cited text using start/end indices
                            text = content_block.get("text", "")
                            start_idx = annotation.get("start_index", 0)
                            end_idx = annotation.get("end_index", len(text))
                            cited_text = text[start_idx:end_idx] if text else ""
                            
                            source = SearchSource(
                                url=annotation.get("url", ""),
                                title=annotation.get("title", ""),
                                content=cited_text[:1000] if cited_text else f"Citation from {annotation.get('title', 'Unknown')}",
                                search_query=f"citation_iteration_{iteration}",
                                iteration=iteration
                            )
                            collected_sources.append(source)
            elif content_block.get("type") == 'web_search_tool_result':
                if "content" in content_block:
                    for search_result in content_block["content"]:
                        if "type" in search_result and search_result["type"] == 'web_search_result':
                            source = SearchSource(
                                url=search_result.get('url', ''),
                                title=search_result.get('title', ''),
                                content=f"Source found in iteration {iteration}",
                                search_query=f"iteration_{iteration}",
                                iteration=iteration
                            )
                            collected_sources.append(source)
    
    return collected_sources

@dataclass(kw_only=True)
class WorkflowConfiguration:
    # General Configuration
    max_structured_output_retries: int = 3
    outline_user_approval: bool = False
    # Research Configuration
    search_api: SearchAPI = SearchAPI.TAVILY
    search_api_config: Optional[Dict[str, Any]] = None
    process_search_results: Literal["summarize", "split_and_rerank"] | None = "summarize"
    number_of_queries: int = 4 # Number of search queries to generate per iteration
    max_search_depth: int = 4 # Maximum number of reflection + search iterations
    # Model Configuration
    summarization_model: str = "anthropic:claude-3-5-haiku-latest"
    summarization_model_max_tokens: int = 10000
    research_model: str = "anthropic:claude-3-7-sonnet-latest"
    research_model_max_tokens: int = 10000
    reflection_model: str = "anthropic:claude-3-7-sonnet-latest"
    reflection_model_max_tokens: int = 10000
    outliner_model: str = "openai:gpt-4.1"
    outliner_model_max_tokens: int = 10000
    final_report_model: str = "anthropic:claude-3-7-sonnet-latest"
    final_report_model_max_tokens: int = 10000
    # MCP server configuration
    mcp_server_config: Optional[Dict[str, Any]] = None
    mcp_prompt: Optional[str] = None
    mcp_tools_to_include: Optional[list[str]] = None

    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> "WorkflowConfiguration":
        """Create a WorkflowConfiguration instance from a RunnableConfig."""
        configurable = (
            config["configurable"] if config and "configurable" in config else {}
        )
        values: dict[str, Any] = {
            f.name: os.environ.get(f.name.upper(), configurable.get(f.name))
            for f in fields(cls)
            if f.init
        }
        return cls(**{k: v for k, v in values.items() if v})