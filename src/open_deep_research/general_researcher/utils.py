from open_deep_research.general_researcher.state import SearchSource
from open_deep_research.general_researcher.configuration import SearchAPI, WorkflowConfiguration
from open_deep_research.utils import (
    tavily_search,
    duckduckgo_search
)
from langchain_core.tools import BaseTool
from langchain_core.messages import AIMessage, ToolMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.runnables import RunnableConfig
import json
import warnings

async def get_search_tool(search_api: SearchAPI):
    if search_api == SearchAPI.ANTHROPIC:
        return [{"type": "web_search_20250305", "name": "web_search"}]
    elif search_api == SearchAPI.OPENAI:
        return [{"type": "web_search_preview"}]
    elif search_api == SearchAPI.TAVILY:
        search_tool = tavily_search
        tool_metadata = {**(search_tool.metadata or {}), "type": "search", "name": "web_search"}
        search_tool.metadata = tool_metadata
        return [search_tool]
    elif search_api == SearchAPI.DUCKDUCKGO:
        search_tool = duckduckgo_search
        tool_metadata = {**(search_tool.metadata or {}), "type": "search", "name": "web_search"}
        search_tool.metadata = tool_metadata
        return [search_tool]

async def load_mcp_tools(
    config: RunnableConfig,
    existing_tool_names: set[str],
) -> list[BaseTool]:
    configurable = WorkflowConfiguration.from_runnable_config(config)
    if not configurable.mcp_server_config:
        return []
    mcp_server_config = configurable.mcp_server_config
    client = MultiServerMCPClient(mcp_server_config)
    mcp_tools = await client.get_tools()
    filtered_mcp_tools: list[BaseTool] = []
    for tool in mcp_tools:
        if tool.name in existing_tool_names:
            warnings.warn(
                f"Trying to add MCP tool with a name {tool.name} that is already in use - this tool will be ignored."
            )
            continue
        if configurable.mcp_tools_to_include and tool.name not in configurable.mcp_tools_to_include:
            continue
        filtered_mcp_tools.append(tool)
    return filtered_mcp_tools

def extract_content_from_model_websearch_response(response):
    text_blocks = []
    if hasattr(response, 'content'):
        for content_block in response.content:
            if "type" in content_block and content_block["type"] == "text":
                text_blocks.append(content_block["text"])
    return "\n\n".join(text_blocks) if text_blocks else ""
    
def call_source_extractor(response, iteration, search_api: SearchAPI):
    if search_api == SearchAPI.ANTHROPIC:
        return collect_sources_anthropic(response, iteration)
    elif search_api == SearchAPI.OPENAI:
        return collect_sources_openai(response, iteration)
    else:
        return []
    
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

def anthropic_websearch_called(response):
    try:
        usage = response.response_metadata.get("usage")
        if not usage:
            return False
        server_tool_use = usage.get("server_tool_use")
        if not server_tool_use:
            return False
        web_search_requests = server_tool_use.get("web_search_requests")
        if web_search_requests is None:
            return False
        return web_search_requests > 0
    except (AttributeError, TypeError):
        return False

def openai_websearch_called(response):
    tool_outputs = response.additional_kwargs.get("tool_outputs")
    if tool_outputs:
        for tool_output in tool_outputs:
            if tool_output.get("type") == "web_search_call":
                return True
    return False

def handle_ai_message_response_types(message, search_api: SearchAPI, iteration: int):
    observation = ""
    collected_sources = []
    # If we provided Anthropic Search, we need to inspect the response_metadata to see if the model called a native web search tool.
    if search_api == SearchAPI.ANTHROPIC and anthropic_websearch_called(message):
        observation = extract_content_from_model_websearch_response(message)
        collected_sources = collect_sources_anthropic(message, iteration)
    # If we provided OpenAI Search, we need to inspect the additional_kwargs to see if the model called a native web search tool.
    elif search_api == SearchAPI.OPENAI and openai_websearch_called(message):
        observation = extract_content_from_model_websearch_response(message)
        collected_sources = collect_sources_openai(message, iteration)
    elif len(message.tool_calls) > 0:
        for tool_call in message.tool_calls:
            tool_name = tool_call.get("name")
            tool_args = tool_call.get("args")
            if tool_name and tool_args:
                observation += f"{tool_name}: {json.dumps(tool_args)}\n"
    else:
        observation = message.content
    return observation, collected_sources


def extract_notes_from_research_messages(research_messages, research_iterations, search_api: SearchAPI):
    notes = f"Notes from Research Iteration {research_iterations}:\n\n"
    collected_sources = []
    for message in research_messages:
        if isinstance(message, AIMessage):
            observation, new_sources = handle_ai_message_response_types(message, search_api, research_iterations)
            notes += observation
            if new_sources:
                sources_str = "\n".join([f"- {source.title} ({source.url})" for source in new_sources])
                notes += f"\nSources:\n{sources_str}\n\n"
                collected_sources.extend(new_sources)
        elif isinstance(message, ToolMessage):
            notes += f"{message.name}: {message.content}\n\n"
    return notes, collected_sources
