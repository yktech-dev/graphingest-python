"""
GraphIngest + LangGraph — Single Agent Demo
=============================================

Wraps a LangGraph research agent as a GraphIngest @node, giving you:
  - Automatic retries if the LLM call fails
  - Real-time step-by-step logging in the dashboard
  - Input caching (skip re-running identical queries)
  - .map() fan-out (run N agents in parallel on Cloud Run)

Prerequisites:
  pip install graphingest[langgraph] langchain-openai
  export GRAPHINGEST_API_URL=http://localhost:3000
  export GRAPHINGEST_API_KEY=your-key
  export OPENAI_API_KEY=your-openai-key
"""

from graphingest import graph, GraphRunContext
from graphingest.langgraph import agent_node, AgentConfig

# ---------------------------------------------------------------------------
# 1. Define the LangGraph agent builder
# ---------------------------------------------------------------------------

def build_research_agent(config: AgentConfig):
    """Build a simple ReAct-style research agent using LangGraph."""
    from langgraph.graph import StateGraph, END
    from langgraph.prebuilt import ToolNode
    from langchain_openai import ChatOpenAI
    from langchain_core.tools import tool
    from typing import TypedDict, Annotated
    import operator

    # Define tools
    @tool
    def web_search(query: str) -> str:
        """Search the web for information."""
        # Simulate search results
        return f"Search results for '{query}': Found 3 relevant articles about the topic."

    @tool
    def summarize(text: str) -> str:
        """Summarize a piece of text."""
        return f"Summary: {text[:100]}... (condensed to key points)"

    tools = [web_search, summarize]

    # Define state
    class AgentState(TypedDict):
        messages: Annotated[list, operator.add]

    # Build graph
    llm = ChatOpenAI(model=config.model, temperature=config.temperature)
    llm_with_tools = llm.bind_tools(tools)

    def call_model(state: AgentState) -> dict:
        response = llm_with_tools.invoke(state["messages"])
        return {"messages": [response]}

    def should_continue(state: AgentState) -> str:
        last = state["messages"][-1]
        if hasattr(last, "tool_calls") and last.tool_calls:
            return "tools"
        return "done"

    builder = StateGraph(AgentState)
    builder.add_node("agent", call_model)
    builder.add_node("tools", ToolNode(tools))

    builder.set_entry_point("agent")
    builder.add_conditional_edges("agent", should_continue, {
        "tools": "tools",
        "done": END,
    })
    builder.add_edge("tools", "agent")

    return builder.compile(**config.compile_kwargs)


# ---------------------------------------------------------------------------
# 2. Wrap the agent as a GraphIngest node
# ---------------------------------------------------------------------------

researcher = agent_node(
    name="researcher",
    graph_builder=build_research_agent,
    config=AgentConfig(
        model="gpt-4o",
        temperature=0.0,
        max_iterations=10,
        system_prompt="You are a research assistant. Use the available tools to answer questions thoroughly.",
        stream_steps=True,  # Log each agent step to the GraphIngest dashboard
    ),
    cache_ttl=600,   # Cache identical queries for 10 minutes
    retries=2,       # Retry up to 2 times on LLM failures
    retry_delay_seconds=3,
)


# ---------------------------------------------------------------------------
# 3. Use in a GraphIngest pipeline
# ---------------------------------------------------------------------------

@graph(name="single-agent-research", timeout_seconds=120)
def research_single(query: str):
    """Run a single research agent."""
    ctx = GraphRunContext.get()
    print(f"  Graph run: {ctx.graph_run_id}")

    result = researcher(query)
    print(f"  Agent result: {result['result'][:100]}")
    print(f"  Steps taken: {result['steps']}")
    print(f"  Time: {result['elapsed_seconds']}s")
    return result


# ---------------------------------------------------------------------------
# 4. Fan-out: run multiple agents in parallel
# ---------------------------------------------------------------------------

@graph(name="parallel-research", timeout_seconds=300)
def research_parallel(queries: list[str]):
    """Fan-out multiple research agents across Cloud Run workers."""
    ctx = GraphRunContext.get()
    print(f"  Graph run: {ctx.graph_run_id}")
    print(f"  Dispatching {len(queries)} agents in parallel...")

    # .map() dispatches N Cloud Run instances, one per query
    results = researcher.map(queries)

    print(f"  All {len(results)} agents completed!")
    for i, r in enumerate(results):
        print(f"    [{i}] {r['result'][:80]}... ({r['steps']} steps, {r['elapsed_seconds']}s)")

    return {
        "query_count": len(queries),
        "results": results,
        "total_steps": sum(r["steps"] for r in results),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Single Agent ===")
    result = research_single("What are the latest advances in quantum computing?")
    print(f"\nFinal: {result}")

    print("\n=== Parallel Agents (fan-out) ===")
    results = research_parallel([
        "What is quantum computing?",
        "Explain transformer architecture in ML",
        "What are the benefits of serverless computing?",
    ])
    print(f"\nFinal: {results['query_count']} queries, {results['total_steps']} total steps")
