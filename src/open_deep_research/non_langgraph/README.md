# Non-LangGraph Deep Research Agent

A native Python implementation of an iterative deep research agent using Anthropic's APIs.

## Overview

This agent provides deep research capabilities through:
- **Autonomous Web Search**: Uses Claude's web search tool for initial research
- **Iterative Reflection**: Analyzes findings and identifies knowledge gaps
- **Gap-Driven Research**: Conducts follow-up searches to fill identified gaps
- **Citation-Powered Reports**: Generates final reports with proper source attribution

## Architecture

The agent follows a simple but effective pipeline:
1. **Research Phase**: Claude autonomously searches using web search tool
2. **Reflection Phase**: Analyzes completeness and identifies gaps
3. **Iteration**: Repeats until satisfied or max iterations reached
4. **Final Synthesis**: Generates comprehensive report with citations

## Usage

### Basic Usage

```python
from open_deep_research.non_langgraph.graph import create_research_agent, ResearchConfig

# Create agent with default settings
agent = create_research_agent()

# Research a topic
messages = [{"role": "user", "content": "Research artificial intelligence safety"}]
result = agent.research(messages)

print(result["content"])  # Final report
print(f"Sources: {len(result['sources'])}")  # Number of sources found
```

### Custom Configuration

```python
# Configure for evaluation (no user interaction)
config = ResearchConfig(
    model="claude-3-5-sonnet-20241022",
    max_iterations=3,
    enable_clarification_qa=False,  # Disable for automated testing
    temperature=0.3
)

agent = create_research_agent(config)
result = agent.research(messages)
```

### Configuration Options

- `model`: Anthropic model to use (default: "claude-3-5-sonnet-20241022")
- `max_iterations`: Maximum research iterations (default: 3)
- `enable_clarification_qa`: Enable interactive scope clarification (default: True)
- `temperature`: Model temperature (default: 0.7)
- `max_tokens`: Maximum tokens per API call (default: 4000)
- `search_domains`: Optional domain filtering for searches
- `search_location`: Optional location-based search results

## Running the Agent

### From Python Script

```python
from open_deep_research.non_langgraph.graph import create_research_agent

agent = create_research_agent()
messages = [{"role": "user", "content": "Your research topic here"}]
result = agent.research(messages)
```

### From Command Line

```bash
# Run the example in the graph.py file
python -m src.open_deep_research.non_langgraph.graph
```

## Testing

### Run the Test Suite

```bash
# From project root
python -m src.open_deep_research.non_langgraph.test_agent
```

### What the Test Validates

The test uses "Model Context Protocol" as a research topic and confirms:

1. **Agent Execution**: Runs without errors through all phases
2. **Report Generation**: Produces substantial content (>100 characters)
3. **Topic Relevance**: Report mentions relevant keywords
4. **Structure Validation**: Proper result format with content, sources, metadata
5. **Configuration**: Custom settings are properly applied
6. **Iteration Logic**: Completes expected number of research iterations

### Test Output

The test provides detailed logging and shows:
- Research progress through iterations
- API calls to Anthropic
- Report preview (first 500 characters)
- Final statistics (length, sources, iterations)

## Dependencies

- `anthropic`: Anthropic Python SDK
- `logging`: Standard Python logging
- `json`: JSON parsing for reflection analysis

## Environment Setup

Ensure your `ANTHROPIC_API_KEY` environment variable is set:

```bash
export ANTHROPIC_API_KEY="your-api-key-here"
```

## Limitations

- Web search tool results depend on Anthropic's API availability
- Citations API integration pending full API release
- Source collection currently uses placeholder logic (will be enhanced when web search returns full results)

## Files

- `graph.py`: Main agent implementation
- `test_agent.py`: Comprehensive test suite
- `README.md`: This documentation