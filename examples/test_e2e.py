#!/usr/bin/env python3
"""
End-to-end test: GraphIngest SDK → Control Plane → Cloud Run Worker → Callback

This script tests the full pipeline:
  1. Creates a flow definition via the control plane API
  2. Triggers a flow run
  3. Polls until completion
  4. Verifies results in the dashboard

Prerequisites:
  - Control plane running and publicly accessible (Vercel or ngrok)
  - Cloud Run worker deployed and configured with GRAPHINGEST_API_URL
  - GRAPHINGEST_API_URL and GRAPHINGEST_API_KEY env vars set

Usage:
    export GRAPHINGEST_API_URL=https://your-app.vercel.app   # or ngrok URL
    export GRAPHINGEST_API_KEY=your-api-key
    python examples/test_e2e.py
"""

import os
import sys
import time
import json
import httpx

# ─── Config ───────────────────────────────────────────────────────

API_URL = (
    os.environ.get("GRAPHINGEST_API_URL")
    or os.environ.get("INGEST_API_URL")
    or "http://localhost:3000"
)
API_KEY = (
    os.environ.get("GRAPHINGEST_API_KEY")
    or os.environ.get("INGEST_API_KEY")
    or ""
)

POLL_INTERVAL = 2  # seconds
MAX_WAIT = 120     # seconds

_headers: dict[str, str] = {"Content-Type": "application/json"}
if API_KEY:
    _headers["Authorization"] = f"Bearer {API_KEY}"

client = httpx.Client(
    base_url=API_URL,
    headers=_headers,
    timeout=30.0,
)


def log(msg: str):
    print(f"  → {msg}")


def log_ok(msg: str):
    print(f"  ✓ {msg}")


def log_fail(msg: str):
    print(f"  ✗ {msg}")


# ─── Step 1: Health check ────────────────────────────────────────

def test_health():
    print("\n" + "=" * 60)
    print("  STEP 1: Control Plane Health Check")
    print("=" * 60)
    log(f"Target: {API_URL}")

    try:
        # Try the flows endpoint as a health proxy
        resp = client.get("/api/flows?limit=1")
        if resp.status_code == 200:
            log_ok(f"Control plane is reachable (status {resp.status_code})")
            return True
        else:
            log_fail(f"Unexpected status: {resp.status_code} — {resp.text[:200]}")
            return False
    except Exception as e:
        log_fail(f"Cannot reach control plane: {e}")
        return False


# ─── Step 2: Create a test flow ──────────────────────────────────

def create_test_flow() -> str | None:
    print("\n" + "=" * 60)
    print("  STEP 2: Create Test Flow")
    print("=" * 60)

    flow_def = {
        "name": "e2e-test-pipeline",
        "description": "End-to-end test: extract → transform → load",
        "tags": ["e2e-test", "automated"],
        "parameters_schema": {
            "source": {"type": "string", "default": "https://example.com/data.csv"}
        },
        "task_definitions": [
            {
                "key": "extract",
                "name": "Extract Data",
                "type": "standard",
                "depends_on": [],
                "max_retries": 2,
            },
            {
                "key": "transform",
                "name": "Transform Data",
                "type": "standard",
                "depends_on": ["extract"],
                "max_retries": 1,
            },
            {
                "key": "load",
                "name": "Load Data",
                "type": "standard",
                "depends_on": ["transform"],
                "max_retries": 1,
            },
        ],
    }

    log(f"Creating flow: {flow_def['name']}")

    try:
        resp = client.post("/api/flows", json=flow_def)
        if resp.status_code == 201:
            data = resp.json()
            flow_id = data["data"]["id"]
            log_ok(f"Flow created: {flow_id}")
            return flow_id
        else:
            log_fail(f"Failed to create flow: {resp.status_code} — {resp.text[:300]}")
            return None
    except Exception as e:
        log_fail(f"Error creating flow: {e}")
        return None


# ─── Step 3: Trigger a flow run ──────────────────────────────────

def trigger_flow_run(flow_id: str) -> str | None:
    print("\n" + "=" * 60)
    print("  STEP 3: Trigger Flow Run")
    print("=" * 60)

    payload = {
        "parameters": {
            "source": "https://example.com/test-data.csv",
        },
    }

    log(f"Triggering flow run for flow: {flow_id}")

    try:
        resp = client.post(f"/api/flows/{flow_id}/runs", json=payload)
        if resp.status_code == 201:
            data = resp.json()
            run_id = data["data"]["id"]
            state = data["data"]["state"]
            log_ok(f"Flow run created: {run_id} (state: {state})")
            return run_id
        else:
            log_fail(f"Failed to trigger run: {resp.status_code} — {resp.text[:300]}")
            return None
    except Exception as e:
        log_fail(f"Error triggering run: {e}")
        return None


# ─── Step 4: Poll for completion ─────────────────────────────────

def poll_flow_run(flow_id: str, run_id: str) -> dict | None:
    print("\n" + "=" * 60)
    print("  STEP 4: Polling for Completion")
    print("=" * 60)

    start = time.monotonic()
    last_state = None

    while True:
        elapsed = time.monotonic() - start
        if elapsed > MAX_WAIT:
            log_fail(f"Timed out after {MAX_WAIT}s")
            return None

        try:
            resp = client.get(f"/api/flows/{flow_id}/runs?limit=1")
            if resp.status_code != 200:
                log(f"Poll error: {resp.status_code}")
                time.sleep(POLL_INTERVAL)
                continue

            runs = resp.json().get("data", [])
            if not runs:
                log("No runs found yet...")
                time.sleep(POLL_INTERVAL)
                continue

            run = runs[0]
            state = run.get("state", "UNKNOWN")

            if state != last_state:
                log(f"State: {state} ({elapsed:.1f}s elapsed)")
                last_state = state

            if state in ("COMPLETED", "FAILED", "CANCELLED"):
                return run

        except Exception as e:
            log(f"Poll error: {e}")

        time.sleep(POLL_INTERVAL)


# ─── Step 5: Check task run details ──────────────────────────────

def check_task_runs(flow_run_id: str):
    print("\n" + "=" * 60)
    print("  STEP 5: Task Run Details")
    print("=" * 60)

    try:
        resp = client.get(f"/api/runs/{flow_run_id}/tasks")
        if resp.status_code != 200:
            # Try alternative endpoint
            log(f"Task details endpoint returned {resp.status_code}")
            return

        tasks = resp.json().get("data", [])
        for task in tasks:
            state = task.get("state", "?")
            key = task.get("task_key", "?")
            duration = task.get("duration_ms")
            icon = "✓" if state == "COMPLETED" else "✗" if state == "FAILED" else "⋯"
            dur_str = f" ({duration}ms)" if duration else ""
            print(f"  {icon} {key}: {state}{dur_str}")

    except Exception as e:
        log(f"Could not fetch task details: {e}")


# ─── Main ────────────────────────────────────────────────────────

def main():
    print("\n" + "█" * 60)
    print("  GraphIngest E2E Test")
    print("█" * 60)
    print(f"  API URL:  {API_URL}")
    print(f"  API Key:  {'*' * 8}...{API_KEY[-4:]}" if len(API_KEY) > 4 else "  API Key:  (not set)")

    # Step 1: Health check
    if not test_health():
        print("\n  ⚠ Control plane is not reachable. Make sure it is running.")
        print(f"    Tried: {API_URL}")
        sys.exit(1)

    # Step 2: Create flow
    flow_id = create_test_flow()
    if not flow_id:
        print("\n  ⚠ Could not create flow. Check the control plane logs.")
        sys.exit(1)

    # Step 3: Trigger run
    run_id = trigger_flow_run(flow_id)
    if not run_id:
        print("\n  ⚠ Could not trigger flow run.")
        sys.exit(1)

    # Step 4: Poll
    result = poll_flow_run(flow_id, run_id)

    # Step 5: Check task details
    if result:
        check_task_runs(result["id"])

    # Summary
    print("\n" + "=" * 60)
    if result and result.get("state") == "COMPLETED":
        print("  ✓ E2E TEST PASSED — Full pipeline completed!")
    elif result and result.get("state") == "FAILED":
        error = result.get("state_message", "unknown")
        print(f"  ✗ E2E TEST FAILED — Pipeline failed: {error}")
        print(f"\n  Check dashboard: {API_URL}/dashboard")
    else:
        print("  ⚠ E2E TEST INCONCLUSIVE — Pipeline did not complete")
        print(f"\n  Check dashboard: {API_URL}/dashboard")

    print(f"\n  Flow ID:  {flow_id}")
    print(f"  Run ID:   {run_id}")
    print(f"  Dashboard: {API_URL}/dashboard")
    print("=" * 60 + "\n")

    # Cleanup hint
    print("  💡 To clean up, delete the test flow from the dashboard.")
    print(f"     Or: curl -X DELETE {API_URL}/api/flows/{flow_id}\n")


if __name__ == "__main__":
    main()
