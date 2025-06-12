"""
Non-LangGraph implementation of deep research agent using native Python and Anthropic APIs.

This module provides an iterative research system that:
1. Performs initial research using Claude's web search tool
2. Reflects on findings to identify knowledge gaps 
3. Conducts follow-up research to fill gaps
4. Synthesizes final report with proper citations
"""

import json
import logging
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from langsmith import traceable

import anthropic
from anthropic.types import Message


logger = logging.getLogger(__name__)


@dataclass
class ResearchConfig:
    """Configuration for the deep research agent."""
    
    # Model settings
    model: str = "claude-3-5-sonnet-latest"  # Web search supported model
    writer_model: str = "claude-opus-4-20250514"  # Citations supported model
    temperature: float = 0.7
    max_tokens: int = 10000
    
    # Research parameters
    max_iterations: int = 3
    max_searches_total: int = 10
    
    # User interaction
    enable_clarification_qa: bool = True  # Disable for evaluations
    clarification_timeout: int = 30  # seconds
    
    # Search settings
    search_domains: Optional[List[str]] = None
    search_location: Optional[str] = None


@dataclass 
class ReflectionResult:
    """Result of reflection phase analyzing research completeness."""
    is_satisfied: bool
    knowledge_gaps: List[str]
    suggested_queries: List[str] 
    reasoning: str


@dataclass
class SearchSource:
    """Represents a source found during web search."""
    url: str
    title: str
    content: str
    search_query: str
    iteration: int


class DeepResearchAgent:
    """
    Deep research agent that iteratively searches, reflects, and synthesizes information.
    
    Uses Anthropic's web search tool for autonomous research and citations API for 
    final report generation with proper source attribution.
    """
    
    def __init__(self, config: ResearchConfig):
        self.config = config
        self.client = anthropic.Anthropic()
        self.collected_sources: List[SearchSource] = []
        self.research_history: List[str] = []
        
    @traceable(run_type="llm",
               name="Overall Research",
               project_name="Anthropic Deep Researcher")
    def research(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Main research pipeline with iterative reflection.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
                     e.g., [{"role": "user", "content": "Research AI safety"}]
        
        Returns:
            Dict with 'content' (final report), 'citations', 'sources', and metadata
        """
        logger.info("Starting deep research process")
        
        # Optional clarification phase
        if self.config.enable_clarification_qa:
            messages = self._clarify_scope(messages)
        
        # Extract research topic for logging
        topic = self._extract_topic(messages)
        logger.info(f"Research topic: {topic}")
        
        # Iterative research loop
        iteration = 0
        current_findings = ""
        
        while iteration < self.config.max_iterations:
            logger.info(f"Starting research iteration {iteration + 1}/{self.config.max_iterations}")
            
            # Step 1: Research phase with web search
            research_response = self._research_phase(messages, current_findings, iteration)
            
            # Step 2: Extract and store sources  
            self._collect_sources(research_response, iteration)
            
            # Extract all text content from the response (including search results)
            current_findings = ""
            if research_response.content:
                text_blocks = []
                for content_block in research_response.content:
                    if hasattr(content_block, 'type') and content_block.type == 'text':
                        text_blocks.append(content_block.text)
                current_findings = "\n\n".join(text_blocks)
            
            self.research_history.append(current_findings)
            
            # Step 3: Reflection phase - analyze gaps
            reflection = self._reflection_phase(messages, current_findings)
            logger.info(f"Reflection satisfied: {reflection.is_satisfied}")
            
            # Step 4: Check if satisfied or continue
            if reflection.is_satisfied or iteration >= self.config.max_iterations - 1:
                logger.info("Research complete - proceeding to final synthesis")
                break
                
            # Step 5: Prepare for next iteration with gap analysis
            messages = self._prepare_next_iteration(messages, current_findings, reflection)
            iteration += 1
        
        # Step 6: Final synthesis with citations - use ALL research findings
        all_findings = "\n\n---ITERATION SEPARATOR---\n\n".join(self.research_history)
        return self._generate_final_report(messages, all_findings)
    
    def _clarify_scope(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Optional interactive clarification of research scope.
        
        Args:
            messages: Original message history
            
        Returns:
            Updated messages with clarification (if any)
        """
        # TODO: Implement interactive clarification
        # For now, return original messages unchanged
        logger.info("Clarification phase skipped - not yet implemented")
        return messages
    
    def _extract_topic(self, messages: List[Dict[str, str]]) -> str:
        """Extract research topic from messages."""
        user_messages = [msg["content"] for msg in messages if msg["role"] == "user"]
        return user_messages[-1] if user_messages else "Unknown topic"
    
    @traceable(run_type="llm",
               name="Web Search",
               project_name="Anthropic Deep Researcher")
    def _research_phase(self, messages: List[Dict[str, str]], previous_findings: str, iteration: int) -> Message:
        """
        Research phase where Claude autonomously searches using web search tool.
        
        Args:
            messages: Conversation history
            previous_findings: Results from previous research iterations
            iteration: Current iteration number
            
        Returns:
            Anthropic Message response with search results
        """
        # Build system prompt based on iteration
        if iteration == 0:
            system_prompt = """You are a research assistant conducting deep research. Use the web search tool to thoroughly research the user's topic.

Focus on:
- Finding authoritative, comprehensive sources
- Covering multiple perspectives and aspects of the topic
- Using specific, targeted search queries
- Gathering sufficient information for a detailed report

Use multiple searches as needed to cover the topic thoroughly."""
        else:
            system_prompt = f"""You are continuing deep research based on previous findings. Use web search to fill knowledge gaps identified in the reflection.

Previous findings:
{previous_findings}

Focus on areas that need deeper investigation. Use targeted searches to find missing information and perspectives."""

        # Create research prompt
        research_messages = messages.copy()
        if iteration > 0:
            research_messages.append({
                "role": "assistant", 
                "content": f"Previous research findings:\n\n{previous_findings}"
            })
            research_messages.append({
                "role": "user",
                "content": "Continue researching to fill the knowledge gaps identified. Focus on the areas that need deeper investigation."
            })
        
        try:
            response = self.client.messages.create(
                model=self.config.model,
                system=system_prompt,
                messages=research_messages,
                tools=[{
                    "type": "web_search_20250305",
                    "name": "web_search"
                }],
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature
            )
            
            logger.info(f"Research phase completed with {len(response.content)} content blocks")
                            
            return response
            
        except Exception as e:
            logger.error(f"Error in research phase: {e}")
            raise
    
    def _collect_sources(self, response: Message, iteration: int):
        """
        Extract and store sources from web search results.
        
        Args:
            response: Anthropic Message response containing tool uses
            iteration: Current research iteration
        """
        sources_found = 0
        
        # Parse content blocks for web search results and citations
        if hasattr(response, 'content') and response.content:
            for content_block in response.content:
                # Look for WebSearchToolResultBlock
                if hasattr(content_block, 'type') and content_block.type == 'web_search_tool_result':
                    if hasattr(content_block, 'content') and content_block.content:
                        for search_result in content_block.content:
                            if hasattr(search_result, 'type') and search_result.type == 'web_search_result':
                                # Extract source information (skip encrypted content, we'll get readable content from citations)
                                source = SearchSource(
                                    url=getattr(search_result, 'url', ''),
                                    title=getattr(search_result, 'title', ''),
                                    content=f"Source found in iteration {iteration}",  # Placeholder - real content comes from citations
                                    search_query=f"iteration_{iteration}",
                                    iteration=iteration
                                )
                                self.collected_sources.append(source)
                                sources_found += 1
                
                # Also collect citation information from TextBlocks
                elif hasattr(content_block, 'type') and content_block.type == 'text':
                    if hasattr(content_block, 'citations') and content_block.citations:
                        for citation in content_block.citations:
                            if hasattr(citation, 'url') and hasattr(citation, 'title'):
                                # Extract the actual cited text content
                                cited_content = getattr(citation, 'cited_text', '')
                                if not cited_content:
                                    cited_content = f"Citation from {citation.title}"
                                
                                source = SearchSource(
                                    url=citation.url,
                                    title=citation.title,
                                    content=cited_content[:1000],  # Keep more content for citations
                                    search_query=f"citation_iteration_{iteration}",
                                    iteration=iteration
                                )
                                self.collected_sources.append(source)
                                sources_found += 1
        
        logger.info(f"Source collection for iteration {iteration} - found {sources_found} sources")
    
    @traceable(run_type="llm",
               name="Reflection",
               project_name="Anthropic Deep Researcher")
    def _reflection_phase(self, messages: List[Dict[str, str]], findings: str) -> ReflectionResult:
        """
        Analyze research completeness and identify knowledge gaps.
        
        Args:
            messages: Original conversation messages
            findings: Current research findings to analyze
            
        Returns:
            ReflectionResult with satisfaction status and gap analysis
        """
        topic = self._extract_topic(messages)
        
        reflection_prompt = f"""Analyze these research findings for completeness and depth:

Original Research Topic: {topic}

Current Research Findings:
{findings}

Please evaluate:
1. Are the findings comprehensive enough for a detailed report?
2. What key knowledge gaps or missing perspectives exist?
3. What specific areas need deeper investigation?
4. What follow-up research queries would be most valuable?

Respond in valid JSON format with these exact keys:
{{
    "is_satisfied": boolean,
    "knowledge_gaps": ["gap1", "gap2", ...],
    "suggested_queries": ["query1", "query2", ...],
    "reasoning": "detailed explanation of the analysis"
}}"""

        try:
            response = self.client.messages.create(
                model=self.config.model,
                messages=[{"role": "user", "content": reflection_prompt}],
                max_tokens=1000,
                temperature=0.3  # Lower temperature for structured analysis
            )
            
            # Parse JSON response
            content = response.content[0].text if response.content else "{}"
            reflection_data = json.loads(content)
            
            return ReflectionResult(
                is_satisfied=reflection_data.get("is_satisfied", False),
                knowledge_gaps=reflection_data.get("knowledge_gaps", []),
                suggested_queries=reflection_data.get("suggested_queries", []),
                reasoning=reflection_data.get("reasoning", "")
            )
            
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Error parsing reflection response: {e}")
            # Return conservative fallback - continue research
            return ReflectionResult(
                is_satisfied=False,
                knowledge_gaps=["Analysis parsing failed"],
                suggested_queries=["Continue general research"],
                reasoning="Failed to parse reflection analysis"
            )
        except Exception as e:
            logger.error(f"Error in reflection phase: {e}")
            raise
    
    def _prepare_next_iteration(self, messages: List[Dict[str, str]], 
                               findings: str, reflection: ReflectionResult) -> List[Dict[str, str]]:
        """
        Prepare messages for next research iteration based on reflection.
        
        Args:
            messages: Original messages
            findings: Current findings
            reflection: Gap analysis results
            
        Returns:
            Updated message list for next iteration
        """
        gap_context = f"""
Knowledge gaps to address:
{chr(10).join('- ' + gap for gap in reflection.knowledge_gaps)}

Suggested focus areas:
{chr(10).join('- ' + query for query in reflection.suggested_queries)}

Reasoning: {reflection.reasoning}
"""
        
        updated_messages = messages.copy()
        updated_messages.append({
            "role": "assistant",
            "content": f"Current research status:\n\n{findings}"
        })
        updated_messages.append({
            "role": "user", 
            "content": f"Please continue research to address these gaps:\n\n{gap_context}"
        })
        
        return updated_messages
    
    def _prepare_citations_documents(self) -> List[Dict[str, Any]]:
        """
        Format collected sources for Anthropic's citations API.
        
        Returns:
            List of document dicts for citations API
        """
        documents = []
        for source in self.collected_sources:
            documents.append({
                "type": "text",
                "text": source.content,
                "source": {
                    "type": "other", 
                    "other": source.url
                }
            })
        
        return documents
    
    @traceable(run_type="llm",
               name="Final Report",
               project_name="Anthropic Deep Researcher")
    def _generate_final_report(self, messages: List[Dict[str, str]], findings: str) -> Dict[str, Any]:
        """
        Generate comprehensive final report with proper citations.
        
        Args:
            messages: Original conversation
            findings: All research findings
            
        Returns:
            Dict with final report content, citations, and metadata
        """
        topic = self._extract_topic(messages)
        
        # Note: Citations API would require restructuring research findings as documents
        
        # Include source information in the prompt
        source_list = "\n".join([f"- {source.title}: {source.url}" for source in self.collected_sources[:10]])  # Limit to first 10 for brevity
        
        final_prompt = f"""Based on all the research conducted, create a comprehensive, well-structured report on: {topic}

Research findings to synthesize:
{findings}

Sources found during research (reference these as appropriate):
{source_list}

Please create a detailed report that:
1. Has a clear introduction and conclusion
2. Is well-organized with proper headings
3. Includes specific facts and insights from the research
4. References relevant sources using [Title](URL) format
5. Provides a balanced, thorough analysis
6. Includes a "Sources" section at the end with all referenced links

Format the report in clear markdown with proper structure and include source references where appropriate."""

        try:
            # NOTE: Citations API requires documents in message content array format:
            # messages=[{"role": "user", "content": [
            #     {"type": "document", "source": {...}, "citations": {"enabled": True}},
            #     {"type": "text", "text": "prompt"}
            # ]}]
            # This would require restructuring our research findings into document format.
            # For now, using manual source references in the prompt.
            
            response = self.client.messages.create(
                model=self.config.writer_model,
                system="You are generating a comprehensive research report. Reference sources appropriately.",
                messages=[{"role": "user", "content": final_prompt}],
                max_tokens=self.config.max_tokens,
                temperature=0.3
            )
            
            content = response.content[0].text if response.content else ""
            
            return {
                "content": content,
                "citations": getattr(response, 'citations', []),
                "sources": [
                    {"url": source.url, "title": source.title} 
                    for source in self.collected_sources
                ],
                "metadata": {
                    "iterations": len(self.research_history),
                    "total_sources": len(self.collected_sources),
                    "model": self.config.model
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating final report: {e}")
            raise


def create_research_agent(config: Optional[ResearchConfig] = None) -> DeepResearchAgent:
    """
    Factory function to create a configured research agent.
    
    Args:
        config: Optional configuration. Uses defaults if not provided.
        
    Returns:
        Configured DeepResearchAgent instance
    """
    if config is None:
        config = ResearchConfig()
    
    return DeepResearchAgent(config)


# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Create agent with default config
    agent = create_research_agent()
    
    # Example research query
    messages = [
        {"role": "user", "content": "Research the latest developments in AI safety research"}
    ]
    
    # Run research
    result = agent.research(messages)
    
    print("Research Complete!")
    print(f"Report length: {len(result['content'])} characters")
    print(f"Sources found: {result['metadata']['total_sources']}")
    print(f"Iterations: {result['metadata']['iterations']}")