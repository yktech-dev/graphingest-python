"""
GraphIngest + LangGraph — Multi-Agent Pipeline Demo
=====================================================

Demonstrates a supervisor-worker pattern where:
  1. A planner agent breaks a task into sub-tasks
  2. Worker agents execute each sub-task in parallel via .map()
  3. A synthesizer agent combines all results

Each agent runs as a separate Cloud Run instance with full
retry protection, logging, and observability.

Prerequisites:
  pip install graphingest[langgraph] langchain-openai
  export GRAPHINGEST_API_URL=http://localhost:3000
  export GRAPHINGEST_API_KEY=your-key
  export OPENAI_API_KEY=your-openai-key
"""

from graphingest import node, graph, RetryPolicy, GraphRunContext
from graphingest.langgraph import agent_node, agent_graph, AgentConfig


# ---------------------------------------------------------------------------
# Agent builders
# ---------------------------------------------------------------------------

def build_planner(config: AgentConfig):
    """Planner: breaks a complex question into sub-questions."""
    from langgraph.graph import StateGraph, END
    from langchain_openai import ChatOpenAI
    from typing import TypedDict, Annotated
    import operator
    import json

    class PlannerState(TypedDict):
        messages: Annotated[list, operator.add]
        sub_tasks: list[str]

    llm = ChatOpenAI(model=config.model, temperature=config.temperature)

    def plan(state: PlannerState) -> dict:
        prompt = (
            "You are a research planner. Given a complex question, break it into "
            "3-5 independent sub-questions that can be researched in parallel. "
            "Return ONLY a JSON array of strings, e.g.: "
            '["sub-question 1", "sub-question 2", "sub-question 3"]'
        )
        messages = [{"role": "system", "content": prompt}] + state["messages"]
        response = llm.invoke(messages)
        try:
            sub_tasks = json.loads(response.content)
        except json.JSONDecodeError:
            sub_tasks = [response.content]
        return {"messages": [response], "sub_tasks": sub_tasks}

    builder = StateGraph(PlannerState)
    builder.add_node("plan", plan)
    builder.set_entry_point("plan")
    builder.add_edge("plan", END)
    return builder.compile()


def build_researcher(config: AgentConfig):
    """Researcher: answers a specific question using tools."""
    from langgraph.graph import StateGraph, END
    from langgraph.prebuilt import ToolNode
    from langchain_openai import ChatOpenAI
    from langchain_core.tools import tool
    from typing import TypedDict, Annotated
    import operator

    @tool
    def web_search(query: str) -> str:
        """Search the web for information."""
        return f"Found: '{query}' — 3 relevant results with key findings on the topic."

    @tool
    def analyze_data(topic: str) -> str:
        """Analyze available data on a topic."""
        return f"Analysis of '{topic}': Key trends show significant growth and adoption."

    tools = [web_search, analyze_data]

    class ResearchState(TypedDict):
        messages: Annotated[list, operator.add]

    llm = ChatOpenAI(model=config.model, temperature=config.temperature)
    llm_with_tools = llm.bind_tools(tools)

    def research(state: ResearchState) -> dict:
        response = llm_with_tools.invoke(state["messages"])
        return {"messages": [response]}

    def should_continue(state: ResearchState) -> str:
        last = state["messages"][-1]
        if hasattr(last, "tool_calls") and last.tool_calls:
            return "tools"
        return "done"

    builder = StateGraph(ResearchState)
    builder.add_node("research", research)
    builder.add_node("tools", ToolNode(tools))
    builder.set_entry_point("research")
    builder.add_conditional_edges("research", should_continue, {
        "tools": "tools",
        "done": END,
    })
    builder.add_edge("tools", "research")
    return builder.compile()


def build_synthesizer(config: AgentConfig):
    """Synthesizer: combines multiple research results into a cohesive report."""
    from langgraph.graph import StateGraph, END
    from langchain_openai import ChatOpenAI
    from typing import TypedDict, Annotated
    import operator

    class SynthState(TypedDict):
        messages: Annotated[list, operator.add]

    llm = ChatOpenAI(model=config.model, temperature=config.temperature)

    def synthesize(state: SynthState) -> dict:
        prompt = (
            "You are a research synthesizer. Combine the following research findings "
            "into a clear, structured summary with key insights and conclusions."
        )
        messages = [{"role": "system", "content": prompt}] + state["messages"]
        response = llm.invoke(messages)
        return {"messages": [response]}

    builder = StateGraph(SynthState)
    builder.add_node("synthesize", synthesize)
    builder.set_entry_point("synthesize")
    builder.add_edge("synthesize", END)
    return builder.compile()


# ---------------------------------------------------------------------------
# Wrap agents as GraphIngest nodes
# ---------------------------------------------------------------------------

planner = agent_node(
    name="planner",
    graph_builder=build_planner,
    config=AgentConfig(model="gpt-4o", temperature=0.0, max_iterations=3),
    retries=2,
    retry_delay_seconds=2,
)

researcher = agent_node(
    name="researcher",
    graph_builder=build_researcher,
    config=AgentConfig(
        model="gpt-4o",
        temperature=0.0,
        max_iterations=10,
        system_prompt="Answer the given question thoroughly using available tools.",
        stream_steps=True,
    ),
    cache_ttl=600,
    retries=2,
    retry_delay_seconds=3,
)

synthesizer = agent_node(
    name="synthesizer",
    graph_builder=build_synthesizer,
    config=AgentConfig(model="gpt-4o", temperature=0.3, max_iterations=3),
    retries=1,
)


# ---------------------------------------------------------------------------
# Orchestrate: planner → parallel researchers → synthesizer
# ---------------------------------------------------------------------------

@agent_graph(
    name="multi-agent-research",
    timeout_seconds=600,
    tags=["agents", "research"],
)
def multi_agent_research(question: str):
    """
    Multi-agent research pipeline:
      1. Planner breaks the question into sub-tasks
      2. Researchers investigate each sub-task in parallel (.map)
      3. Synthesizer combines all findings
    """
    ctx = GraphRunContext.get()
    print(f"  [pipeline] Run: {ctx.graph_run_id}")
    print(f"  [pipeline] Question: {question}")

    # Step 1: Plan
    plan_result = planner(question)
    sub_tasks = plan_result.get("result", [question])
    if isinstance(sub_tasks, str):
        sub_tasks = [sub_tasks]
    print(f"  [pipeline] Planner created {len(sub_tasks)} sub-tasks:")
    for i, task in enumerate(sub_tasks):
        print(f"    {i+1}. {task}")

    # Step 2: Research in parallel (fan-out across Cloud Run)
    print(f"  [pipeline] Dispatching {len(sub_tasks)} researchers in parallel...")
    research_results = researcher.map(sub_tasks)
    print(f"  [pipeline] All researchers completed!")

    # Step 3: Synthesize
    findings = "\n\n".join(
        f"## Sub-task {i+1}: {sub_tasks[i]}\n{r['result']}"
        for i, r in enumerate(research_results)
    )
    synthesis = synthesizer(findings)

    return {
        "question": question,
        "sub_tasks": sub_tasks,
        "research_results": [r["result"] for r in research_results],
        "synthesis": synthesis["result"],
        "total_agent_steps": (
            plan_result["steps"]
            + sum(r["steps"] for r in research_results)
            + synthesis["steps"]
        ),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Multi-Agent Research Pipeline ===\n")

    result = multi_agent_research(
        "What are the implications of quantum computing on cybersecurity, "
        "and how should organizations prepare?"
    )

    print(f"\n{'='*60}")
    print(f"Sub-tasks: {len(result['sub_tasks'])}")
    print(f"Total agent steps: {result['total_agent_steps']}")
    print(f"\nSynthesis:\n{result['synthesis'][:500]}...")
