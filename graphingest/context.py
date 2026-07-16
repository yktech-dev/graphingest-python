"""
Graph Run Context

Provides access to the current graph/node run metadata from anywhere
inside a running graph or node function.

Usage:
    from graphingest import GraphRunContext

    @graph(name="my-pipeline")
    def pipeline(source: str):
        ctx = GraphRunContext.get()
        print(ctx.graph_run_id)  # uuid
        print(ctx.graph_name)    # "my-pipeline"
        print(ctx.parameters)    # {"source": "..."}
"""

from __future__ import annotations

import os
import contextvars
from dataclasses import dataclass, field
from typing import Any, Optional


_graph_run_ctx_var: contextvars.ContextVar[Optional["GraphRunContext"]] = contextvars.ContextVar(
    "graph_run_context", default=None
)

_node_run_ctx_var: contextvars.ContextVar[Optional["NodeRunContext"]] = contextvars.ContextVar(
    "node_run_context", default=None
)


@dataclass
class GraphRunContext:
    """Immutable context available inside a running graph."""

    graph_run_id: str
    graph_name: str
    graph_version: Optional[str] = None
    parameters: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    parent_graph_run_id: Optional[str] = None

    @classmethod
    def get(cls) -> Optional["GraphRunContext"]:
        """Get the current graph run context, or None if not inside a graph."""
        return _graph_run_ctx_var.get()

    @classmethod
    def current(cls) -> Optional["GraphRunContext"]:
        """Alias for get() — returns the current context or None."""
        return cls.get()

    @classmethod
    def get_or_raise(cls) -> "GraphRunContext":
        """Get the current graph run context, raising if not inside a graph."""
        ctx = _graph_run_ctx_var.get()
        if ctx is None:
            raise RuntimeError(
                "No active GraphRunContext. "
                "This can only be called from within a @graph-decorated function."
            )
        return ctx

    def _set(self) -> contextvars.Token:
        return _graph_run_ctx_var.set(self)

    @staticmethod
    def _reset(token: contextvars.Token) -> None:
        _graph_run_ctx_var.reset(token)


@dataclass
class NodeRunContext:
    """Immutable context available inside a running node."""

    node_run_id: str
    node_key: str
    graph_run_id: str
    map_index: Optional[int] = None
    retry_count: int = 0

    @classmethod
    def get(cls) -> Optional["NodeRunContext"]:
        """Get the current node run context, or None if not inside a node."""
        return _node_run_ctx_var.get()

    @classmethod
    def get_or_raise(cls) -> "NodeRunContext":
        """Get the current node run context, raising if not inside a node."""
        ctx = _node_run_ctx_var.get()
        if ctx is None:
            raise RuntimeError(
                "No active NodeRunContext. "
                "This can only be called from within a @node-decorated function."
            )
        return ctx

    def _set(self) -> contextvars.Token:
        return _node_run_ctx_var.set(self)

    @staticmethod
    def _reset(token: contextvars.Token) -> None:
        _node_run_ctx_var.reset(token)
