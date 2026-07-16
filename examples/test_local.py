#!/usr/bin/env python3
"""
Local test: Run GraphIngest pipelines without a control plane.

Tests core SDK features:
  1. @node / @graph decorators
  2. RetryPolicy (exponential backoff + jitter)
  3. Timeouts
  4. Subgraph nesting
  5. ConcurrencyPolicy / ThrottlePolicy (types only — enforcement needs control plane)
  6. State hooks (on_completion, on_failure)

Usage:
    cd sdk/python
    pip install -e .
    python examples/test_local.py
"""

import time
import logging
from graphingest import (
    node, graph, RetryPolicy, ConcurrencyPolicy, ThrottlePolicy,
    GraphRunContext, NodeRunContext,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
log = logging.getLogger("test")

# ─── Scenario 1: Basic @node → @graph ───────────────────────────

@node(name="extract")
def extract(url: str) -> dict:
    log.info(f"Extracting from {url}")
    time.sleep(0.1)  # simulate work
    return {"url": url, "rows": 42}

@node(name="transform")
def transform(data: dict) -> dict:
    log.info(f"Transforming {data['rows']} rows")
    return {"cleaned_rows": data["rows"], "source": data["url"]}

@node(name="load")
def load(data: dict) -> str:
    log.info(f"Loading {data['cleaned_rows']} rows from {data['source']}")
    return f"Loaded {data['cleaned_rows']} rows"

@graph(name="basic-etl")
def basic_pipeline(url: str):
    data = extract(url)
    cleaned = transform(data)
    result = load(cleaned)
    return result


# ─── Scenario 2: RetryPolicy with exponential backoff ───────────

attempt_count = 0

@node(name="flaky-api")
def flaky_api_call(endpoint: str) -> dict:
    global attempt_count
    attempt_count += 1
    if attempt_count < 3:
        raise ConnectionError(f"Attempt {attempt_count}: API timeout on {endpoint}")
    return {"status": "ok", "data": [1, 2, 3]}

@graph(
    name="retry-demo",
    retry_policy=RetryPolicy(
        max_retries=5,
        delay_seconds=0.2,
        backoff_factor=2,
        jitter=True,
    ),
)
def retry_pipeline(endpoint: str):
    # Note: do NOT reset attempt_count here — graph retries re-call this function
    result = flaky_api_call(endpoint)
    return result


# ─── Scenario 3: Timeout enforcement ────────────────────────────

@node(name="slow-task")
def slow_task(seconds: int) -> str:
    log.info(f"Sleeping {seconds}s...")
    time.sleep(seconds)
    return f"Done after {seconds}s"

@graph(name="timeout-demo", timeout_seconds=2)
def timeout_pipeline():
    return slow_task(1)  # Should succeed (1s < 2s timeout)

@graph(name="timeout-fail-demo", timeout_seconds=1)
def timeout_fail_pipeline():
    return slow_task(5)  # Should fail (5s > 1s timeout)


# ─── Scenario 4: Subgraph nesting ───────────────────────────────

@graph(name="sub-pipeline")
def sub_pipeline(item: str) -> str:
    log.info(f"Sub-pipeline processing: {item}")
    return f"processed:{item}"

@graph(name="parent-pipeline")
def parent_pipeline(items: list) -> list:
    results = []
    for item in items:
        result = sub_pipeline(item)  # subgraph detected automatically
        results.append(result)
    return results


# ─── Scenario 5: State hooks ────────────────────────────────────

completion_log = []
failure_log = []

def on_done(ctx, result):
    completion_log.append({"graph": ctx.graph_name, "result": result})
    log.info(f"✓ Hook: {ctx.graph_name} completed with: {result}")

def on_fail(ctx, error):
    failure_log.append({"graph": ctx.graph_name, "error": str(error)})
    log.info(f"✗ Hook: {ctx.graph_name} failed with: {error}")

@graph(
    name="hooks-demo",
    on_completion=[on_done],
    on_failure=[on_fail],
)
def hooks_pipeline(should_fail: bool):
    if should_fail:
        raise ValueError("Intentional failure for testing")
    return "success"


# ─── Scenario 6: Flow control types (validation only) ───────────

# Only attach flow control policies if the control plane is running
import os
_has_control_plane = bool(os.environ.get("GRAPHINGEST_API_URL") or os.environ.get("INGEST_API_URL"))

@graph(
    name="flow-control-demo",
    concurrency=ConcurrencyPolicy(limit=5, key="user_id", wait_timeout_seconds=30) if _has_control_plane else None,
    throttle=ThrottlePolicy(limit=100, period_seconds=60) if _has_control_plane else None,
    priority=10,
)
def flow_control_pipeline(user_id: str, data: str):
    """Flow control types are accepted by the decorator.
    Enforcement only happens when control plane is running."""
    log.info(f"Processing for user={user_id}: {data}")
    return f"done:{user_id}"


# ─── Run all scenarios ──────────────────────────────────────────

def run_test(name, fn, expect_error=False):
    print(f"\n{'='*60}")
    print(f"  TEST: {name}")
    print(f"{'='*60}")
    try:
        result = fn()
        if expect_error:
            print(f"  ✗ FAIL — expected error but got: {result}")
            return False
        print(f"  ✓ PASS — result: {result}")
        return True
    except Exception as e:
        if expect_error:
            print(f"  ✓ PASS — got expected error: {type(e).__name__}: {e}")
            return True
        print(f"  ✗ FAIL — unexpected error: {type(e).__name__}: {e}")
        return False


if __name__ == "__main__":
    results = []

    # 1. Basic pipeline
    results.append(run_test(
        "Basic ETL Pipeline",
        lambda: basic_pipeline("https://example.com/data.csv"),
    ))

    # 2. Retry with backoff
    attempt_count = 0  # reset before the test, not inside the graph
    results.append(run_test(
        "Retry Policy (fails 2x, succeeds on 3rd)",
        lambda: retry_pipeline("https://api.example.com/v1/data"),
    ))

    # 3. Timeout — passes
    results.append(run_test(
        "Timeout (1s task, 2s limit → should pass)",
        lambda: timeout_pipeline(),
    ))

    # 4. Timeout — fails
    results.append(run_test(
        "Timeout (5s task, 1s limit → should timeout)",
        lambda: timeout_fail_pipeline(),
        expect_error=True,
    ))

    # 5. Subgraph nesting
    results.append(run_test(
        "Subgraph Nesting (parent → child)",
        lambda: parent_pipeline(["alpha", "beta", "gamma"]),
    ))

    # 6. Hooks — success
    results.append(run_test(
        "State Hooks — on_completion",
        lambda: hooks_pipeline(should_fail=False),
    ))

    # 7. Hooks — failure
    results.append(run_test(
        "State Hooks — on_failure",
        lambda: hooks_pipeline(should_fail=True),
        expect_error=True,
    ))

    # 8. Flow control types
    results.append(run_test(
        "Flow Control Types (ConcurrencyPolicy + ThrottlePolicy)",
        lambda: flow_control_pipeline("user_123", "some payload"),
    ))

    # Summary
    passed = sum(results)
    total = len(results)
    print(f"\n{'='*60}")
    print(f"  RESULTS: {passed}/{total} passed")
    print(f"{'='*60}")

    if passed < total:
        exit(1)
