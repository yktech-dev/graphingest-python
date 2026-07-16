"""
GraphIngest SDK: @node and @graph decorators

@node lifecycle:
  On Enter  → Check cache; set NodeRunContext; attach streaming logger.
  On Execute → Run function (sync or async); capture all logs.
  On Exit   → Upload result to GCS; report COMPLETED + result_url.

@graph lifecycle:
  On Enter  → Validate parameters (Pydantic); set GraphRunContext; attach streaming logger.
  On Execute → Run function with timeout + retry loop (sync or async).
  On Exit   → Fire state hooks (on_completion / on_failure / on_cancellation).
"""

from __future__ import annotations

import os
import sys
import json
import time
import signal
import logging
import asyncio
import inspect
import traceback
import functools
import uuid
from contextlib import contextmanager
from typing import Any, Callable, Optional, TypeVar, ParamSpec, Union
from dataclasses import dataclass, field

# Global registry: node_key → original unwrapped function
_node_registry: dict[str, Callable] = {}
from datetime import datetime
import random

from google.cloud import storage as gcs

from .client import GraphIngestClient
from .context import GraphRunContext, NodeRunContext
from .logger import StreamingLogHandler

logger = logging.getLogger("graphingest")

P = ParamSpec("P")
R = TypeVar("R")

_client: Optional[GraphIngestClient] = None


def get_client() -> GraphIngestClient:
    global _client
    if _client is None:
        _client = GraphIngestClient()
    return _client


# ---------------------------------------------------------------------------
# Platform limits (tier-based)
# ---------------------------------------------------------------------------

PLATFORM_LIMITS = {
    "free": {
        "node_default_timeout": 300,       # 5 min
        "node_max_timeout": 600,           # 10 min
        "graph_default_timeout": 900,      # 15 min
        "graph_max_timeout": 1800,         # 30 min
        "monthly_execution_minutes": 60,   # 60 min/mo total
        "max_pipelines": 5,
    },
    "pro": {
        "node_default_timeout": 600,       # 10 min
        "node_max_timeout": 3600,          # 60 min
        "graph_default_timeout": 3600,     # 1 hr
        "graph_max_timeout": 21600,        # 6 hr
        "monthly_execution_minutes": None, # unlimited
        "max_pipelines": None,             # unlimited
    },
    "enterprise": {
        "node_default_timeout": 3600,      # 60 min
        "node_max_timeout": 86400,         # 24 hr
        "graph_default_timeout": 21600,    # 6 hr
        "graph_max_timeout": 86400,        # 24 hr
        "monthly_execution_minutes": None, # unlimited
        "max_pipelines": None,             # unlimited
    },
}


def _get_tier() -> str:
    """Resolve the current platform tier from env. Defaults to 'free'."""
    return os.environ.get("GRAPHINGEST_TIER", "free").lower()


def _get_limits() -> dict:
    """Return the platform limits dict for the current tier."""
    return PLATFORM_LIMITS.get(_get_tier(), PLATFORM_LIMITS["free"])


def _clamp_timeout(requested: Optional[int], default_key: str, max_key: str) -> int:
    """Clamp a user-requested timeout to the platform limit for the current tier.

    If requested is None, returns the tier default. If requested exceeds the
    tier max, it is clamped to the max and a warning is logged.
    """
    limits = _get_limits()
    if requested is None:
        return limits[default_key]
    clamped = min(requested, limits[max_key])
    if clamped < requested:
        logger.warning(
            f"Requested timeout {requested}s exceeds {_get_tier()} tier max "
            f"({limits[max_key]}s). Clamped to {clamped}s. "
            f"Upgrade your plan for longer execution limits."
        )
    return clamped


# ---------------------------------------------------------------------------
# Timeout helper
# ---------------------------------------------------------------------------

class NodeTimeoutError(TimeoutError):
    """Raised when a node exceeds its timeout_seconds."""


class GraphTimeoutError(TimeoutError):
    """Raised when a graph exceeds its timeout_seconds."""


def _timeout_handler(signum: int, frame: Any) -> None:
    raise GraphTimeoutError("Graph execution timed out")


def _node_timeout_handler(signum: int, frame: Any) -> None:
    raise NodeTimeoutError("Node execution timed out")


@contextmanager
def _apply_timeout(timeout_seconds: Optional[int], error_cls: type = GraphTimeoutError):
    """Apply a SIGALRM-based timeout (Unix only). No-op on Windows or if None."""
    if timeout_seconds is None or sys.platform == "win32":
        yield
        return

    handler = _node_timeout_handler if error_cls is NodeTimeoutError else _timeout_handler
    old_handler = signal.signal(signal.SIGALRM, handler)
    signal.alarm(timeout_seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


# ---------------------------------------------------------------------------
# Task internals
# ---------------------------------------------------------------------------

class NodeLogHandler(logging.Handler):
    """Captures log records during node execution for batch upload."""

    def __init__(self) -> None:
        super().__init__()
        self.records: list[dict] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append({
            "level": record.levelname,
            "message": self.format(record),
            "worker_id": os.environ.get("WORKER_ID", "unknown"),
            "metadata": {
                "logger": record.name,
                "filename": record.filename,
                "lineno": record.lineno,
            },
        })


@contextmanager
def _node_lifecycle(node_key: str, node_run_id: str, graph_run_id: str):
    """Context manager that wraps node execution with logging, context, and reporting."""
    client = get_client()

    # Set NodeRunContext
    ctx = NodeRunContext(
        node_run_id=node_run_id,
        node_key=node_key,
        graph_run_id=graph_run_id,
    )
    token = ctx._set()

    # Attach streaming log handler if we have a graph_run_id
    streaming_handler: Optional[StreamingLogHandler] = None
    batch_handler = NodeLogHandler()
    batch_handler.setFormatter(logging.Formatter("%(asctime)s [%(name)s] %(message)s"))
    root_logger = logging.getLogger()
    root_logger.addHandler(batch_handler)

    if graph_run_id:
        streaming_handler = StreamingLogHandler(
            flow_run_id=graph_run_id,
            task_run_id=node_run_id,
            client=client,
        )
        streaming_handler.setFormatter(logging.Formatter("%(asctime)s [%(name)s] %(message)s"))
        root_logger.addHandler(streaming_handler)

    start_time = time.monotonic()

    try:
        yield batch_handler
    except Exception:
        duration_ms = int((time.monotonic() - start_time) * 1000)
        tb = traceback.format_exc()
        logger.error(f"Node {node_key} failed: {tb}")

        try:
            client.report_task_failed(
                task_run_id=node_run_id,
                flow_run_id=graph_run_id,
                error_message=str(tb.splitlines()[-1]),
                error_traceback=tb,
                logs=batch_handler.records,
            )
        except Exception as report_err:
            logger.error(f"Failed to report node failure: {report_err}")
        raise
    finally:
        root_logger.removeHandler(batch_handler)
        if streaming_handler:
            streaming_handler.close()
            root_logger.removeHandler(streaming_handler)
        NodeRunContext._reset(token)


def _upload_to_gcs(node_key: str, node_run_id: str, result: Any) -> Optional[str]:
    """Upload node result to Google Cloud Storage and return the URL."""
    bucket_name = os.environ.get("GCS_BUCKET")
    if not bucket_name:
        return None

    try:
        client = gcs.Client()
        bucket = client.bucket(bucket_name)
        blob_path = f"results/{node_key}/{node_run_id}.json"
        blob = bucket.blob(blob_path)
        blob.upload_from_string(
            json.dumps(result, default=str),
            content_type="application/json",
        )
        return f"gs://{bucket_name}/{blob_path}"
    except Exception as e:
        logger.warning(f"GCS upload failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Pydantic parameter validation helper
# ---------------------------------------------------------------------------

def _validate_parameters(fn: Callable, args: tuple, kwargs: dict) -> dict:
    """
    If Pydantic is installed and the function has type annotations,
    build a Pydantic model from the signature and validate inputs.
    Returns the validated parameter dict.
    """
    try:
        from pydantic import create_model, ValidationError  # type: ignore
    except ImportError:
        # Pydantic not installed — skip validation
        sig = inspect.signature(fn)
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        return dict(bound.arguments)

    sig = inspect.signature(fn)
    bound = sig.bind(*args, **kwargs)
    bound.apply_defaults()

    fields: dict[str, Any] = {}
    for param_name, param in sig.parameters.items():
        annotation = param.annotation if param.annotation != inspect.Parameter.empty else Any
        default = param.default if param.default != inspect.Parameter.empty else ...
        fields[param_name] = (annotation, default)

    if not fields:
        return dict(bound.arguments)

    Model = create_model(f"{fn.__name__}_params", **fields)  # type: ignore
    try:
        validated = Model(**bound.arguments)
        return validated.model_dump()
    except Exception as e:
        raise TypeError(f"Parameter validation failed: {e}") from e


# ---------------------------------------------------------------------------
# Retry policy
# ---------------------------------------------------------------------------

@dataclass
class RetryPolicy:
    """Configurable retry strategy with exponential backoff and jitter.

    Args:
        max_retries:      Total retry attempts (0 = no retries).
        delay_seconds:    Initial delay between retries.
        backoff_factor:   Multiplier applied to delay after each attempt.
        max_delay_seconds: Upper bound on the computed delay.
        jitter:           If True, adds random jitter (0 to delay) to avoid thundering herd.

    Examples:
        RetryPolicy(max_retries=3, delay_seconds=1, backoff_factor=2)
            → delays: ~1s, ~2s, ~4s

        RetryPolicy(max_retries=5, delay_seconds=0.5, backoff_factor=3, max_delay_seconds=30, jitter=True)
            → delays: ~0.5s, ~1.5s, ~4.5s, ~13.5s, ~30s (each ± jitter)
    """
    max_retries: int = 0
    delay_seconds: float = 0
    backoff_factor: float = 2.0
    max_delay_seconds: float = 120.0
    jitter: bool = True


# ---------------------------------------------------------------------------
# Flow control policies
# ---------------------------------------------------------------------------

@dataclass
class ConcurrencyPolicy:
    """Limits how many graph runs can execute simultaneously.

    Args:
        limit: Maximum number of concurrent runs. 0 = unlimited.
        key:   Optional parameter key for per-key concurrency.
               E.g., key="user_id" limits concurrency per user.
        wait_timeout_seconds: Max time to wait for a slot (0 = fail immediately).
        poll_interval_seconds: How often to re-check for a free slot.

    Examples:
        ConcurrencyPolicy(limit=5)
            → max 5 concurrent runs of this graph globally

        ConcurrencyPolicy(limit=3, key="user_id")
            → max 3 concurrent runs per user_id
    """
    limit: int = 0
    key: str = ""
    wait_timeout_seconds: float = 0
    poll_interval_seconds: float = 2.0


@dataclass
class ThrottlePolicy:
    """Limits how many graph runs can start within a time window.

    Args:
        limit:          Maximum number of runs allowed in the window.
        period_seconds: Size of the sliding window in seconds.

    Examples:
        ThrottlePolicy(limit=10, period_seconds=60)
            → max 10 runs per minute

        ThrottlePolicy(limit=100, period_seconds=3600)
            → max 100 runs per hour
    """
    limit: int = 0
    period_seconds: int = 60


def _compute_retry_delay(policy: RetryPolicy, attempt: int) -> float:
    """Compute the delay for a given attempt using exponential backoff + jitter."""
    delay = policy.delay_seconds * (policy.backoff_factor ** attempt)
    delay = min(delay, policy.max_delay_seconds)
    if policy.jitter:
        delay = delay * (0.5 + random.random())  # jitter: 50%-150% of delay
    return delay


# ---------------------------------------------------------------------------
# Hook types
# ---------------------------------------------------------------------------

StateHook = Callable[["GraphRunContext", Any], None]
AsyncStateHook = Callable[["GraphRunContext", Any], Any]


def _call_hooks(
    hooks: list[Union[StateHook, AsyncStateHook]],
    ctx: GraphRunContext,
    result_or_error: Any,
) -> None:
    """Fire a list of hooks (sync or async)."""
    for hook in hooks:
        try:
            if asyncio.iscoroutinefunction(hook):
                asyncio.get_event_loop().run_until_complete(hook(ctx, result_or_error))
            else:
                hook(ctx, result_or_error)
        except Exception as hook_err:
            logger.warning(f"State hook {hook.__name__} raised: {hook_err}")


# ---------------------------------------------------------------------------
# get_status: check job status by ID
# ---------------------------------------------------------------------------

def get_status(job_id: str) -> dict:
    """Check the status of a dispatched job by its ID.

    Args:
        job_id: The job ID returned by .arun() (i.e. future.task_run_id).

    Returns:
        dict with keys:
            - state: "PENDING" | "RUNNING" | "COMPLETED" | "FAILED"
            - result: The result data (only when state is COMPLETED)
            - error: Error message (only when state is FAILED)
    """
    client = get_client()
    status = client.poll_task_runs([job_id])
    results = status.get("results", [])

    if not results:
        return {"state": "PENDING", "result": None, "error": None}

    r = results[0]
    return {
        "state": r.get("state", "PENDING"),
        "result": r.get("resultData") if r.get("state") == "COMPLETED" else None,
        "error": r.get("errorMessage") if r.get("state") == "FAILED" else None,
    }


# ---------------------------------------------------------------------------
# NodeFuture: handle for async node execution
# ---------------------------------------------------------------------------

class NodeFuture:
    """A handle to a dispatched node execution. Use .result() to block until completion."""

    def __init__(self, task_run_id: str, node_key: str):
        self._task_run_id = task_run_id
        self._node_key = node_key
        self._result: Any = None
        self._error: Optional[str] = None
        self._resolved = False

    @property
    def task_run_id(self) -> str:
        return self._task_run_id

    def result(self, poll_interval: float = 1.0, timeout: Optional[float] = None) -> Any:
        """Block until the node completes and return its result.

        Args:
            poll_interval: Seconds between status polls (default 1s).
            timeout:       Max seconds to wait (None = wait forever).

        Raises:
            RuntimeError: If the node failed.
            TimeoutError: If timeout exceeded.
        """
        if self._resolved:
            if self._error:
                raise RuntimeError(f"Node {self._node_key} failed: {self._error}")
            return self._result

        client = get_client()
        start = time.monotonic()

        while True:
            status = client.poll_task_runs([self._task_run_id])
            results = status.get("results", [])
            if results:
                r = results[0]
                if r["state"] == "COMPLETED":
                    self._result = r.get("resultData")
                    self._resolved = True
                    return self._result
                elif r["state"] == "FAILED":
                    self._error = r.get("errorMessage", "Unknown error")
                    self._resolved = True
                    raise RuntimeError(f"Node {self._node_key} failed: {self._error}")

            if timeout and (time.monotonic() - start) > timeout:
                raise TimeoutError(f"NodeFuture timed out after {timeout}s waiting for {self._node_key}")

            time.sleep(poll_interval)


def _map_nodes(node_key: str, items: list[Any], poll_interval: float = 1.0, timeout: Optional[float] = None) -> list[Any]:
    """Fan-out: dispatch N parallel node executions and collect results in order.

    Works both inside a @graph (uses existing run context) and standalone
    (auto-generates a run ID for tracking).

    Args:
        node_key:       The node key to invoke.
        items:          List of inputs — one per invocation.
        poll_interval:  Seconds between status polls.
        timeout:        Max seconds to wait for all results.

    Returns:
        Ordered list of results matching the input order.
    """
    ctx = GraphRunContext.current()
    graph_run_id = ctx.graph_run_id if ctx else f"standalone-{uuid.uuid4()}"

    client = get_client()
    dispatch_result = client.dispatch_nodes(
        graph_run_id=graph_run_id,
        node_key=node_key,
        inputs=items,
    )
    task_run_ids: list[str] = dispatch_result["taskRunIds"]

    logger.info(f"Mapped {len(task_run_ids)} invocations of node '{node_key}'")

    # Poll until all complete
    start = time.monotonic()
    while True:
        status = client.poll_task_runs(task_run_ids)
        if status.get("allCompleted"):
            break
        if timeout and (time.monotonic() - start) > timeout:
            raise TimeoutError(f".map() timed out after {timeout}s waiting for {node_key}")
        time.sleep(poll_interval)

    # Collect results in order
    results_by_id = {r["id"]: r for r in status["results"]}
    ordered: list[Any] = []
    for tid in task_run_ids:
        r = results_by_id.get(tid, {})
        if r.get("state") == "FAILED":
            raise RuntimeError(f"Node {node_key} (map index) failed: {r.get('errorMessage')}")
        ordered.append(r.get("resultData"))

    return ordered


def _arun_node(node_key: str, input_data: Any) -> NodeFuture:
    """Dispatch a single node execution asynchronously and return a future.

    Works both inside a @graph (uses existing run context) and standalone
    (auto-generates a run ID). Returns a NodeFuture with a .task_run_id
    that can be used to poll for status and retrieve results.

    Args:
        node_key:    The node key to invoke.
        input_data:  Input payload for the invocation.

    Returns:
        NodeFuture that can be .result()'d later, or whose .task_run_id
        can be returned to a frontend for polling.
    """
    ctx = GraphRunContext.current()
    graph_run_id = ctx.graph_run_id if ctx else f"standalone-{uuid.uuid4()}"

    client = get_client()
    dispatch_result = client.dispatch_nodes(
        graph_run_id=graph_run_id,
        node_key=node_key,
        inputs=[input_data],
    )
    task_run_id = dispatch_result["taskRunIds"][0]

    logger.info(f"Dispatched node '{node_key}' → {task_run_id}")
    return NodeFuture(task_run_id, node_key)


# ---------------------------------------------------------------------------
# @node decorator
# ---------------------------------------------------------------------------

def node(
    name: Optional[str] = None,
    cache_ttl: Optional[int] = None,
    max_retries: int = 3,
    tags: Optional[list[str]] = None,
    version: Optional[str] = None,
    timeout_seconds: Optional[int] = None,
):
    """
    Decorator that wraps a function as an Ingest graph node.

    Args:
        name:            Node key override (defaults to function name).
        cache_ttl:       Cache TTL in seconds (None = no caching).
        max_retries:     Max retry attempts before DLQ.
        tags:            Metadata tags for filtering.
        version:         Semantic version string.
        timeout_seconds: Max execution time per node run. Defaults to tier limit
                         (Free: 5min, Pro: 10min, Enterprise: 60min).
                         Clamped to tier max (Free: 10min, Pro: 60min, Enterprise: 24hr).

    Supports both sync and async functions.

    Usage:
        @node(name="extract-data", cache_ttl=3600)
        def extract(url: str) -> dict:
            ...

        @node(name="async-fetch", timeout_seconds=120)
        async def fetch(url: str) -> dict:
            ...
    """

    def decorator(fn: Callable[P, R]) -> Callable[P, R]:
        node_key = name or fn.__name__
        is_async = asyncio.iscoroutinefunction(fn)
        _node_timeout = _clamp_timeout(
            timeout_seconds, "node_default_timeout", "node_max_timeout"
        )

        def _execute_sync(*args: P.args, **kwargs: P.kwargs) -> R:
            node_run_id = os.environ.get("NODE_RUN_ID") or os.environ.get("TASK_RUN_ID", "")
            graph_run_id = os.environ.get("GRAPH_RUN_ID") or os.environ.get("FLOW_RUN_ID", "")
            client = get_client()

            with _node_lifecycle(node_key, node_run_id, graph_run_id) as log_handler:
                start_time = time.monotonic()
                logger.info(f"Starting node: {node_key} (timeout={_node_timeout}s)")
                with _apply_timeout(_node_timeout, error_cls=NodeTimeoutError):
                    result = fn(*args, **kwargs)
                logger.info(f"Node {node_key} completed successfully")
                duration_ms = int((time.monotonic() - start_time) * 1000)

                result_url = _upload_to_gcs(node_key, node_run_id, result)

                result_data = None
                try:
                    serialized = json.dumps(result, default=str)
                    if len(serialized) <= 65536:
                        result_data = json.loads(serialized)
                except (TypeError, ValueError):
                    pass

                try:
                    client.report_task_completed(
                        task_run_id=node_run_id,
                        flow_run_id=graph_run_id,
                        result_url=result_url,
                        result_data=result_data,
                        duration_ms=duration_ms,
                        logs=log_handler.records,
                    )
                except Exception as report_err:
                    logger.error(f"Failed to report completion: {report_err}")

                return result

        async def _execute_async(*args: P.args, **kwargs: P.kwargs) -> R:
            node_run_id = os.environ.get("NODE_RUN_ID") or os.environ.get("TASK_RUN_ID", "")
            graph_run_id = os.environ.get("GRAPH_RUN_ID") or os.environ.get("FLOW_RUN_ID", "")
            client = get_client()

            with _node_lifecycle(node_key, node_run_id, graph_run_id) as log_handler:
                start_time = time.monotonic()
                logger.info(f"Starting async node: {node_key} (timeout={_node_timeout}s)")
                try:
                    result = await asyncio.wait_for(
                        fn(*args, **kwargs),  # type: ignore
                        timeout=_node_timeout,
                    )
                except asyncio.TimeoutError:
                    raise NodeTimeoutError(
                        f"Node {node_key} timed out after {_node_timeout}s"
                    )
                logger.info(f"Node {node_key} completed successfully")
                duration_ms = int((time.monotonic() - start_time) * 1000)

                result_url = _upload_to_gcs(node_key, node_run_id, result)

                result_data = None
                try:
                    serialized = json.dumps(result, default=str)
                    if len(serialized) <= 65536:
                        result_data = json.loads(serialized)
                except (TypeError, ValueError):
                    pass

                try:
                    client.report_task_completed(
                        task_run_id=node_run_id,
                        flow_run_id=graph_run_id,
                        result_url=result_url,
                        result_data=result_data,
                        duration_ms=duration_ms,
                        logs=log_handler.records,
                    )
                except Exception as report_err:
                    logger.error(f"Failed to report completion: {report_err}")

                return result

        wrapper = functools.wraps(fn)(_execute_async if is_async else _execute_sync)

        wrapper._node_key = node_key  # type: ignore
        wrapper._original_fn = fn  # type: ignore

        # Register in global registry so run() can find it
        _node_registry[node_key] = fn
        wrapper._cache_ttl = cache_ttl  # type: ignore
        wrapper._max_retries = max_retries  # type: ignore
        wrapper._tags = tags or []  # type: ignore
        wrapper._version = version  # type: ignore
        wrapper._is_async = is_async  # type: ignore

        # ── .run(), .map() and .arun() ──
        def _run(input_data: Any, poll_interval: float = 1.0, timeout: Optional[float] = None) -> Any:
            """Execute this node on managed infrastructure and wait for the result.

            Usage:
                result = extract.run("https://api.example.com")
            """
            results = _map_nodes(node_key, [input_data], poll_interval=poll_interval, timeout=timeout)
            return results[0] if results else None

        def _map(items: list[Any], poll_interval: float = 1.0, timeout: Optional[float] = None) -> list[Any]:
            """Fan-out: dispatch N parallel invocations and collect results.

            Usage (inside a @graph function):
                results = extract.map(["url1", "url2", "url3"])
            """
            return _map_nodes(node_key, items, poll_interval=poll_interval, timeout=timeout)

        def _arun(input_data: Any) -> NodeFuture:
            """Dispatch a single async invocation and return a NodeFuture.

            Usage (inside a @graph function):
                future = extract.arun("url1")
                result = future.result()
            """
            return _arun_node(node_key, input_data)

        wrapper.run = _run  # type: ignore
        wrapper.map = _map  # type: ignore
        wrapper.arun = _arun  # type: ignore

        return wrapper  # type: ignore

    return decorator


# ---------------------------------------------------------------------------
# @graph decorator
# ---------------------------------------------------------------------------

class FlowControlError(Exception):
    """Raised when flow control prevents graph execution (concurrency/throttle limit hit)."""
    def __init__(self, reason: str, details: dict):
        self.reason = reason
        self.details = details
        super().__init__(f"Flow control blocked: {reason} — {details}")


def _acquire_flow_control(
    graph_name: str,
    graph_run_id: str,
    concurrency: Optional[ConcurrencyPolicy],
    throttle: Optional[ThrottlePolicy],
    validated_params: dict,
    priority: int,
) -> bool:
    """Acquire flow control slot. Returns True if acquired, raises on hard failure."""
    has_concurrency = concurrency and concurrency.limit > 0
    has_throttle = throttle and throttle.limit > 0
    if not has_concurrency and not has_throttle:
        return True

    client = get_client()

    # Resolve concurrency key from parameters if needed
    concurrency_key = ""
    if has_concurrency and concurrency.key:
        concurrency_key = str(validated_params.get(concurrency.key, ""))

    result = client.acquire_flow_control(
        graph_name=graph_name,
        graph_run_id=graph_run_id,
        concurrency_limit=concurrency.limit if has_concurrency else 0,
        concurrency_key=concurrency_key,
        throttle_limit=throttle.limit if has_throttle else 0,
        throttle_period_seconds=throttle.period_seconds if has_throttle else 0,
        priority=priority,
    )

    if result.get("acquired"):
        return True

    # If wait is configured, poll until a slot opens
    if has_concurrency and concurrency.wait_timeout_seconds > 0:
        start = time.monotonic()
        while time.monotonic() - start < concurrency.wait_timeout_seconds:
            time.sleep(concurrency.poll_interval_seconds)
            result = client.acquire_flow_control(
                graph_name=graph_name,
                graph_run_id=graph_run_id,
                concurrency_limit=concurrency.limit if has_concurrency else 0,
                concurrency_key=concurrency_key,
                throttle_limit=throttle.limit if has_throttle else 0,
                throttle_period_seconds=throttle.period_seconds if has_throttle else 0,
                priority=priority,
            )
            if result.get("acquired"):
                return True
            logger.info(
                f"Flow control: waiting for slot (reason={result.get('reason')}, "
                f"elapsed={time.monotonic() - start:.1f}s / {concurrency.wait_timeout_seconds}s)"
            )

    raise FlowControlError(result.get("reason", "unknown"), result)


def _release_flow_control(graph_name: str, graph_run_id: str) -> None:
    """Release flow control slot (best-effort, don't fail the graph if this errors)."""
    try:
        client = get_client()
        client.release_flow_control(graph_name, graph_run_id)
    except Exception as e:
        logger.warning(f"Failed to release flow control slot: {e}")


def graph(
    name: Optional[str] = None,
    description: Optional[str] = None,
    version: Optional[str] = None,
    tags: Optional[list[str]] = None,
    retries: int = 0,
    retry_delay_seconds: float = 0,
    timeout_seconds: Optional[int] = None,
    validate_parameters: bool = True,
    on_completion: Optional[list[StateHook]] = None,
    on_failure: Optional[list[StateHook]] = None,
    on_cancellation: Optional[list[StateHook]] = None,
    retry_policy: Optional[RetryPolicy] = None,
    concurrency: Optional[ConcurrencyPolicy] = None,
    throttle: Optional[ThrottlePolicy] = None,
    priority: int = 0,
):
    """
    Decorator that marks a function as a GraphIngest graph entrypoint.

    Args:
        name:                Graph name override (defaults to function name).
        description:         Human-readable description (or extracted from docstring).
        version:             Semantic version string.
        tags:                Metadata tags for filtering.
        retries:             Number of graph-level retry attempts on failure (legacy, prefer retry_policy).
        retry_delay_seconds: Fixed delay between retries (legacy, prefer retry_policy).
        timeout_seconds:     Max execution time. Defaults to tier limit
                             (Free: 15min, Pro: 1hr, Enterprise: 6hr).
                             Clamped to tier max (Free: 30min, Pro: 6hr, Enterprise: 24hr).
        validate_parameters: Validate args against Pydantic if installed.
        on_completion:       Hooks called with (GraphRunContext, result) on success.
        on_failure:          Hooks called with (GraphRunContext, exception) on failure.
        on_cancellation:     Hooks called with (GraphRunContext, None) on timeout/cancel.
        retry_policy:        RetryPolicy for exponential backoff + jitter. Overrides retries/retry_delay_seconds.
        concurrency:         ConcurrencyPolicy to limit parallel runs. None = unlimited.
        throttle:            ThrottlePolicy to limit run rate. None = unlimited.
        priority:            Priority level (higher = higher priority). Used when waiting for concurrency slots.

    Supports both sync and async functions.

    Usage:
        @graph(name="etl-pipeline", retry_policy=RetryPolicy(max_retries=3, delay_seconds=1, backoff_factor=2))
        def my_pipeline(source: str):
            data = extract(source)
            load(data)

        @graph(
            name="process-upload",
            concurrency=ConcurrencyPolicy(limit=5, key="user_id"),
            throttle=ThrottlePolicy(limit=100, period_seconds=60),
        )
        def process_upload(user_id: str, file_url: str):
            ...
    """
    # Build effective retry config: retry_policy takes priority over legacy args
    if retry_policy is not None:
        _effective_retries = retry_policy.max_retries
        _effective_policy = retry_policy
    elif retries > 0:
        _effective_retries = retries
        _effective_policy = RetryPolicy(
            max_retries=retries,
            delay_seconds=retry_delay_seconds,
            backoff_factor=1.0,  # fixed delay when using legacy args
            jitter=False,
        )
    else:
        _effective_retries = 0
        _effective_policy = RetryPolicy()

    _on_completion = on_completion or []
    _on_failure = on_failure or []
    _on_cancellation = on_cancellation or []
    _graph_timeout = _clamp_timeout(
        timeout_seconds, "graph_default_timeout", "graph_max_timeout"
    )

    def decorator(fn: Callable[P, R]) -> Callable[P, R]:
        graph_name = name or fn.__name__
        graph_desc = description or (fn.__doc__.strip().split("\n")[0] if fn.__doc__ else None)
        is_async = asyncio.iscoroutinefunction(fn)

        def _run_sync_with_retries(
            ctx: GraphRunContext,
            validated_kwargs: dict,
            streaming_handler: Optional[StreamingLogHandler],
        ) -> R:
            last_exc: Optional[Exception] = None
            attempts = 1 + _effective_retries

            for attempt in range(attempts):
                try:
                    with _apply_timeout(_graph_timeout):
                        result = fn(**validated_kwargs)

                    # Success
                    _call_hooks(_on_completion, ctx, result)
                    return result

                except GraphTimeoutError as e:
                    logger.error(f"Graph {graph_name} timed out after {_graph_timeout}s")
                    _call_hooks(_on_cancellation, ctx, e)
                    raise

                except KeyboardInterrupt:
                    _call_hooks(_on_cancellation, ctx, None)
                    raise

                except Exception as e:
                    last_exc = e
                    if attempt < _effective_retries:
                        delay = _compute_retry_delay(_effective_policy, attempt)
                        logger.warning(
                            f"Graph {graph_name} attempt {attempt + 1}/{attempts} failed: {e}. "
                            f"Retrying in {delay:.2f}s..."
                        )
                        if delay > 0:
                            time.sleep(delay)
                    else:
                        _call_hooks(_on_failure, ctx, e)
                        raise

            raise last_exc  # type: ignore  # unreachable but satisfies mypy

        async def _run_async_with_retries(
            ctx: GraphRunContext,
            validated_kwargs: dict,
            streaming_handler: Optional[StreamingLogHandler],
        ) -> R:
            last_exc: Optional[Exception] = None
            attempts = 1 + _effective_retries

            for attempt in range(attempts):
                try:
                    result = await asyncio.wait_for(
                        fn(**validated_kwargs),  # type: ignore
                        timeout=_graph_timeout,
                    )

                    _call_hooks(_on_completion, ctx, result)
                    return result

                except asyncio.TimeoutError as e:
                    logger.error(f"Graph {graph_name} timed out after {_graph_timeout}s")
                    _call_hooks(_on_cancellation, ctx, e)
                    raise GraphTimeoutError("Graph execution timed out") from e

                except asyncio.CancelledError:
                    _call_hooks(_on_cancellation, ctx, None)
                    raise

                except Exception as e:
                    last_exc = e
                    if attempt < _effective_retries:
                        delay = _compute_retry_delay(_effective_policy, attempt)
                        logger.warning(
                            f"Graph {graph_name} attempt {attempt + 1}/{attempts} failed: {e}. "
                            f"Retrying in {delay:.2f}s..."
                        )
                        if delay > 0:
                            await asyncio.sleep(delay)
                    else:
                        _call_hooks(_on_failure, ctx, e)
                        raise

            raise last_exc  # type: ignore

        # Flow control config captured from decorator args
        _concurrency = concurrency
        _throttle = throttle
        _priority = priority

        @functools.wraps(fn)
        def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            # Detect parent graph context for subgraph nesting
            parent_ctx = GraphRunContext.current()
            if parent_ctx is not None:
                # Subgraph: generate a new run ID, link to parent
                graph_run_id = str(uuid.uuid4())
                parent_graph_run_id = parent_ctx.graph_run_id
                logger.info(f"Subgraph '{graph_name}' spawned from parent graph '{parent_ctx.graph_name}'")
            else:
                graph_run_id = os.environ.get("GRAPH_RUN_ID") or os.environ.get("FLOW_RUN_ID", str(uuid.uuid4()))
                parent_graph_run_id = None

            # Validate parameters
            if validate_parameters:
                validated = _validate_parameters(fn, args, kwargs)
            else:
                sig = inspect.signature(fn)
                bound = sig.bind(*args, **kwargs)
                bound.apply_defaults()
                validated = dict(bound.arguments)

            # Flow control: acquire concurrency slot / check throttle
            _has_flow_control = (
                (_concurrency and _concurrency.limit > 0)
                or (_throttle and _throttle.limit > 0)
            )
            if _has_flow_control:
                _acquire_flow_control(
                    graph_name, graph_run_id, _concurrency, _throttle, validated, _priority
                )
                logger.info(f"Flow control: slot acquired for {graph_name} (run={graph_run_id})")

            # Build context
            ctx = GraphRunContext(
                graph_run_id=graph_run_id,
                graph_name=graph_name,
                graph_version=version,
                parameters=validated,
                tags=tags or [],
                parent_graph_run_id=parent_graph_run_id,
            )
            token = ctx._set()

            # Attach streaming logger
            streaming_handler: Optional[StreamingLogHandler] = None
            root_logger = logging.getLogger()
            if graph_run_id:
                client = get_client()
                streaming_handler = StreamingLogHandler(
                    flow_run_id=graph_run_id,
                    client=client,
                )
                streaming_handler.setFormatter(
                    logging.Formatter("%(asctime)s [%(name)s] %(message)s")
                )
                root_logger.addHandler(streaming_handler)

            logger.info(f"Starting graph: {graph_name} (run={graph_run_id})")
            start = time.monotonic()

            try:
                result = _run_sync_with_retries(ctx, validated, streaming_handler)
                duration = time.monotonic() - start
                logger.info(f"Graph {graph_name} completed in {duration:.2f}s")
                return result
            except Exception:
                duration = time.monotonic() - start
                logger.error(f"Graph {graph_name} failed after {duration:.2f}s")
                raise
            finally:
                # Release flow control slot
                if _has_flow_control:
                    _release_flow_control(graph_name, graph_run_id)
                if streaming_handler:
                    streaming_handler.close()
                    root_logger.removeHandler(streaming_handler)
                GraphRunContext._reset(token)

        @functools.wraps(fn)
        async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            # Detect parent graph context for subgraph nesting
            parent_ctx = GraphRunContext.current()
            if parent_ctx is not None:
                graph_run_id = str(uuid.uuid4())
                parent_graph_run_id = parent_ctx.graph_run_id
                logger.info(f"Subgraph '{graph_name}' spawned from parent graph '{parent_ctx.graph_name}'")
            else:
                graph_run_id = os.environ.get("GRAPH_RUN_ID") or os.environ.get("FLOW_RUN_ID", str(uuid.uuid4()))
                parent_graph_run_id = None

            if validate_parameters:
                validated = _validate_parameters(fn, args, kwargs)
            else:
                sig = inspect.signature(fn)
                bound = sig.bind(*args, **kwargs)
                bound.apply_defaults()
                validated = dict(bound.arguments)

            # Flow control: acquire concurrency slot / check throttle
            _has_flow_control = (
                (_concurrency and _concurrency.limit > 0)
                or (_throttle and _throttle.limit > 0)
            )
            if _has_flow_control:
                _acquire_flow_control(
                    graph_name, graph_run_id, _concurrency, _throttle, validated, _priority
                )
                logger.info(f"Flow control: slot acquired for {graph_name} (run={graph_run_id})")

            ctx = GraphRunContext(
                graph_run_id=graph_run_id,
                graph_name=graph_name,
                graph_version=version,
                parameters=validated,
                tags=tags or [],
                parent_graph_run_id=parent_graph_run_id,
            )
            token = ctx._set()

            streaming_handler: Optional[StreamingLogHandler] = None
            root_logger = logging.getLogger()
            if graph_run_id:
                client = get_client()
                streaming_handler = StreamingLogHandler(
                    flow_run_id=graph_run_id,
                    client=client,
                )
                streaming_handler.setFormatter(
                    logging.Formatter("%(asctime)s [%(name)s] %(message)s")
                )
                root_logger.addHandler(streaming_handler)

            logger.info(f"Starting async graph: {graph_name} (run={graph_run_id})")
            start = time.monotonic()

            try:
                result = await _run_async_with_retries(ctx, validated, streaming_handler)
                duration = time.monotonic() - start
                logger.info(f"Graph {graph_name} completed in {duration:.2f}s")
                return result
            except Exception:
                duration = time.monotonic() - start
                logger.error(f"Graph {graph_name} failed after {duration:.2f}s")
                raise
            finally:
                # Release flow control slot
                if _has_flow_control:
                    _release_flow_control(graph_name, graph_run_id)
                if streaming_handler:
                    streaming_handler.close()
                    root_logger.removeHandler(streaming_handler)
                GraphRunContext._reset(token)

        wrapper = async_wrapper if is_async else sync_wrapper

        wrapper._graph_name = graph_name  # type: ignore
        wrapper._graph_description = graph_desc  # type: ignore
        wrapper._graph_version = version  # type: ignore
        wrapper._graph_tags = tags or []  # type: ignore
        wrapper._graph_retries = retries  # type: ignore
        wrapper._graph_timeout_seconds = timeout_seconds  # type: ignore
        wrapper._is_async = is_async  # type: ignore
        return wrapper  # type: ignore

    return decorator
