"""
GraphIngest + LangGraph Integration

Adapters that bridge LangGraph agents with GraphIngest's orchestration layer,
enabling fan-out of AI agents across Cloud Run, real-time log streaming of
agent reasoning, and retry/timeout protection around agent execution.

Usage:
    from graphingest.langgraph import agent_node, agent_graph

    # Wrap a LangGraph StateGraph as a GraphIngest @node
    researcher = agent_node(
        name="researcher",
        graph_builder=build_research_agent,
        cache_ttl=600,
    )

    # Fan-out agents in parallel
    @graph(name="multi-agent-research")
    def pipeline(queries: list[str]):
        results = researcher.map(queries)
        return synthesize(results)

Requires:
    pip install graphingest[langgraph]
    # which installs: langgraph, langchain-core
"""

from __future__ import annotations

import functools
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Type

from .context import GraphRunContext, NodeRunContext
from .task import node, graph

logger = logging.getLogger("graphingest.langgraph")


@dataclass
class AgentConfig:
    """Configuration for an agent node."""

    # LLM model name (e.g., "gpt-4o", "claude-3-sonnet")
    model: str = "gpt-4o"

    # Temperature for LLM generation
    temperature: float = 0.0

    # Maximum iterations for the agent loop (prevents infinite loops)
    max_iterations: int = 25

    # System prompt prepended to every agent invocation
    system_prompt: str = ""

    # Tool names to make available (empty = all tools from the builder)
    tools: list[str] = field(default_factory=list)

    # Whether to stream agent steps to the GraphIngest dashboard
    stream_steps: bool = True

    # Extra kwargs passed to the LangGraph graph compile()
    compile_kwargs: dict[str, Any] = field(default_factory=dict)


def agent_node(
    name: str,
    graph_builder: Callable[..., Any],
    *,
    config: Optional[AgentConfig] = None,
    state_schema: Optional[Type] = None,
    cache_ttl: int = 0,
    retries: int = 0,
    retry_delay_seconds: float = 0,
) -> Any:
    """
    Wrap a LangGraph agent as a GraphIngest @node.

    The graph_builder is a callable that returns a compiled LangGraph StateGraph.
    It receives an AgentConfig and should return graph_builder.compile().

    Args:
        name: Node name (used in dashboard, dispatch, etc.)
        graph_builder: Callable that builds and returns a compiled LangGraph graph.
                       Signature: (config: AgentConfig) -> CompiledStateGraph
        config: Optional AgentConfig to pass to the builder.
        state_schema: Optional Pydantic model or TypedDict for the agent state.
        cache_ttl: Cache agent results for this many seconds (0 = no cache).
        retries: Number of retries on failure.
        retry_delay_seconds: Delay between retries.

    Returns:
        A GraphIngest @node-decorated function with .map() and .arun() support.

    Example:
        from langgraph.graph import StateGraph, END
        from langchain_openai import ChatOpenAI

        def build_research_agent(config):
            llm = ChatOpenAI(model=config.model, temperature=config.temperature)
            builder = StateGraph(dict)
            builder.add_node("reason", lambda s: reason(s, llm))
            builder.add_node("search", search_tool)
            builder.set_entry_point("reason")
            builder.add_conditional_edges("reason", should_continue, {
                "search": "search",
                "done": END,
            })
            builder.add_edge("search", "reason")
            return builder.compile()

        researcher = agent_node(
            name="researcher",
            graph_builder=build_research_agent,
            config=AgentConfig(model="gpt-4o", max_iterations=10),
        )

        # Use like any other node:
        result = researcher("What is quantum computing?")
        results = researcher.map(["query1", "query2", "query3"])
    """
    if config is None:
        config = AgentConfig()

    @node(
        name=name,
        cache_ttl=cache_ttl,
        retries=retries,
        retry_delay_seconds=retry_delay_seconds,
    )
    def _agent_fn(input_data: Any) -> dict:
        run_logger = _get_streaming_logger()

        if run_logger:
            run_logger.info(f"[agent:{name}] Starting with input: {_truncate(input_data, 200)}")

        # Build the LangGraph agent
        compiled_graph = graph_builder(config)

        # Prepare initial state
        if isinstance(input_data, str):
            initial_state = {"messages": [{"role": "user", "content": input_data}]}
        elif isinstance(input_data, dict):
            initial_state = input_data
        else:
            initial_state = {"input": input_data}

        # Add system prompt if configured
        if config.system_prompt and "messages" in initial_state:
            initial_state["messages"].insert(
                0, {"role": "system", "content": config.system_prompt}
            )

        # Run with iteration limit
        recursion_limit = config.max_iterations * 2  # LangGraph counts edges, not nodes

        start = time.time()
        step_count = 0

        if config.stream_steps:
            # Stream mode: log each step to dashboard
            final_state = None
            for event in compiled_graph.stream(
                initial_state,
                {"recursion_limit": recursion_limit},
                stream_mode="updates",
            ):
                step_count += 1
                if run_logger:
                    run_logger.info(f"[agent:{name}] Step {step_count}: {_summarize_event(event)}")
                final_state = event
        else:
            final_state = compiled_graph.invoke(
                initial_state,
                {"recursion_limit": recursion_limit},
            )
            step_count = 1

        elapsed = time.time() - start
        if run_logger:
            run_logger.info(
                f"[agent:{name}] Completed in {elapsed:.1f}s ({step_count} steps)"
            )

        return {
            "result": _extract_result(final_state),
            "steps": step_count,
            "elapsed_seconds": round(elapsed, 2),
            "model": config.model,
        }

    return _agent_fn


def agent_graph(
    name: str,
    *,
    version: str = "",
    tags: Optional[list[str]] = None,
    timeout_seconds: int = 0,
    retry_policy: Any = None,
    on_completion: Optional[Callable] = None,
    on_failure: Optional[Callable] = None,
) -> Callable:
    """
    Decorator factory for creating a GraphIngest @graph that orchestrates agents.

    This is a convenience wrapper around @graph with agent-friendly defaults
    (longer timeout, structured logging).

    Usage:
        @agent_graph(name="research-pipeline", timeout_seconds=600)
        def research(queries: list[str]):
            results = researcher.map(queries)
            return synthesize(results)
    """
    def decorator(fn: Callable) -> Callable:
        graph_kwargs: dict[str, Any] = {"name": name}
        if version:
            graph_kwargs["version"] = version
        if tags:
            graph_kwargs["tags"] = tags
        if timeout_seconds:
            graph_kwargs["timeout_seconds"] = timeout_seconds
        if retry_policy:
            graph_kwargs["retry_policy"] = retry_policy
        if on_completion:
            graph_kwargs["on_completion"] = on_completion
        if on_failure:
            graph_kwargs["on_failure"] = on_failure

        return graph(**graph_kwargs)(fn)

    return decorator


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_streaming_logger() -> Optional[Any]:
    """Get the current streaming logger if inside a graph/node context."""
    try:
        from .logger import get_run_logger
        return get_run_logger()
    except Exception:
        return None


def _truncate(obj: Any, max_len: int = 200) -> str:
    """Truncate a string representation for logging."""
    s = str(obj)
    if len(s) > max_len:
        return s[:max_len] + "..."
    return s


def _summarize_event(event: Any) -> str:
    """Produce a short summary of a LangGraph stream event for dashboard logs."""
    if isinstance(event, dict):
        # LangGraph stream "updates" mode gives {node_name: state_update}
        keys = list(event.keys())
        if len(keys) == 1:
            node_name = keys[0]
            update = event[node_name]
            # Check for messages in the update
            if isinstance(update, dict) and "messages" in update:
                msgs = update["messages"]
                if msgs and isinstance(msgs, list):
                    last = msgs[-1]
                    if isinstance(last, dict):
                        role = last.get("role", "?")
                        content = _truncate(last.get("content", ""), 100)
                        # Check for tool calls
                        tool_calls = last.get("tool_calls", [])
                        if tool_calls:
                            tool_names = [tc.get("name", "?") for tc in tool_calls]
                            return f"{node_name} → called tools: {tool_names}"
                        return f"{node_name} → {role}: {content}"
                    else:
                        # LangChain message objects
                        role = getattr(last, "type", "?")
                        content = _truncate(getattr(last, "content", ""), 100)
                        tool_calls = getattr(last, "tool_calls", [])
                        if tool_calls:
                            tool_names = [tc.get("name", tc.get("type", "?")) for tc in tool_calls]
                            return f"{node_name} → called tools: {tool_names}"
                        return f"{node_name} → {role}: {content}"
            return f"{node_name} → {_truncate(update, 100)}"
    return _truncate(event, 100)


def _extract_result(state: Any) -> Any:
    """Extract the final result from LangGraph agent state."""
    if isinstance(state, dict):
        # Try common patterns
        if "output" in state:
            return state["output"]
        if "result" in state:
            return state["result"]
        if "messages" in state:
            messages = state["messages"]
            if messages and isinstance(messages, list):
                last = messages[-1]
                if isinstance(last, dict):
                    return last.get("content", state)
                return getattr(last, "content", state)
        # For stream "updates" mode, state is {node_name: update}
        keys = list(state.keys())
        if len(keys) == 1:
            return _extract_result(state[keys[0]])
    return state
