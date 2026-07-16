#!/usr/bin/env python3
"""
Cloud Run Worker Test: Execute the full ETL pipeline directly on Cloud Run.

Tests all 4 registered tasks:
  1. extract  → fetches data from a source
  2. transform → doubles values
  3. load     → writes to destination
  4. process_item → fan-out task (simulates .map())

This bypasses Cloud Tasks and calls the worker directly via authenticated HTTP.

Usage:
    python examples/test_cloud_run.py

Requirements:
    - gcloud CLI authenticated (`gcloud auth login`)
    - Cloud Run worker deployed
"""

import json
import subprocess
import sys
import time
import httpx


# ─── Config ───────────────────────────────────────────────────────

WORKER_URL = None  # auto-detected below


def get_worker_url() -> str:
    """Auto-detect the Cloud Run worker URL."""
    result = subprocess.run(
        ["gcloud", "run", "services", "describe", "graphingest-worker",
         "--region=us-central1", "--format=value(status.url)"],
        capture_output=True, text=True,
    )
    url = result.stdout.strip()
    if not url:
        print("  ✗ Could not detect worker URL. Is it deployed?")
        sys.exit(1)
    return url


def get_identity_token() -> str:
    """Get a GCP identity token for Cloud Run auth."""
    result = subprocess.run(
        ["gcloud", "auth", "print-identity-token"],
        capture_output=True, text=True,
    )
    token = result.stdout.strip()
    if not token:
        print("  ✗ Could not get identity token. Run: gcloud auth login")
        sys.exit(1)
    return token


def execute_task(client: httpx.Client, task_key: str, input_data: dict,
                 task_run_id: str = "", flow_run_id: str = "test-flow",
                 map_index: int | None = None) -> dict:
    """Send a task to the Cloud Run worker and return the response."""
    payload = {
        "taskRunId": task_run_id or f"test-{task_key}-{int(time.time())}",
        "flowRunId": flow_run_id,
        "taskKey": task_key,
        "inputData": input_data,
    }
    if map_index is not None:
        payload["mapIndex"] = map_index

    resp = client.post("/api/execute", json=payload)
    if resp.status_code != 200:
        raise RuntimeError(f"Worker returned {resp.status_code}: {resp.text[:300]}")
    return resp.json()


def run_test(name: str, fn, expect_error: bool = False) -> bool:
    print(f"\n{'=' * 60}")
    print(f"  TEST: {name}")
    print(f"{'=' * 60}")
    try:
        result = fn()
        if expect_error:
            print(f"  ✗ FAIL — expected error but got: {json.dumps(result, indent=2)[:200]}")
            return False
        print(f"  ✓ PASS — {json.dumps(result, indent=2)[:300]}")
        return True
    except Exception as e:
        if expect_error:
            print(f"  ✓ PASS — got expected error: {e}")
            return True
        print(f"  ✗ FAIL — {e}")
        return False


# ─── Main ────────────────────────────────────────────────────────

def main():
    print("\n" + "█" * 60)
    print("  GraphIngest — Cloud Run Worker Test")
    print("█" * 60)

    # Setup
    worker_url = get_worker_url()
    token = get_identity_token()
    print(f"  Worker:  {worker_url}")
    print(f"  Token:   {token[:20]}...")

    client = httpx.Client(
        base_url=worker_url,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        timeout=30.0,
    )

    results = []

    # Test 1: Health check
    results.append(run_test("Health Check", lambda: client.get("/health").json()))

    # Test 2: Extract task
    extract_result = {}

    def test_extract():
        nonlocal extract_result
        resp = execute_task(client, "extract", {"source": "https://example.com/data.csv"})
        assert resp["status"] == "completed", f"Expected completed, got {resp['status']}"
        extract_result = resp["result_data"]
        return resp
    results.append(run_test("Extract Task", test_extract))

    # Test 3: Transform task (chain from extract)
    transform_result = {}

    def test_transform():
        nonlocal transform_result
        resp = execute_task(client, "transform", {"data": extract_result})
        assert resp["status"] == "completed", f"Expected completed, got {resp['status']}"
        transform_result = resp["result_data"]
        # Verify values were doubled
        rows = transform_result.get("transformed_rows", [])
        assert len(rows) > 0, "No transformed rows"
        assert rows[0]["value"] == extract_result["rows"][0]["value"] * 2, "Values not doubled"
        return resp
    results.append(run_test("Transform Task (chained from extract)", test_transform))

    # Test 4: Load task (chain from transform)
    def test_load():
        resp = execute_task(client, "load", {"data": transform_result})
        assert resp["status"] == "completed"
        assert resp["result_data"]["status"] == "success"
        return resp
    results.append(run_test("Load Task (chained from transform)", test_load))

    # Test 5: Process item (simulated fan-out)
    def test_fan_out():
        items = ["item_A", "item_B", "item_C"]
        fan_results = []
        for i, item in enumerate(items):
            resp = execute_task(
                client, "process_item",
                {"item": item},
                map_index=i,
            )
            assert resp["status"] == "completed"
            fan_results.append(resp["result_data"])
        return fan_results
    results.append(run_test("Fan-Out (3 process_item tasks)", test_fan_out))

    # Test 6: Unknown task (should fail gracefully)
    def test_unknown():
        resp = execute_task(client, "nonexistent_task", {"hello": "world"})
        assert resp["status"] == "failed", f"Expected failed, got {resp['status']}"
        assert "not found" in resp.get("error", ""), "Expected 'not found' in error"
        return resp
    results.append(run_test("Unknown Task (graceful failure)", test_unknown))

    # Summary
    passed = sum(results)
    total = len(results)
    print(f"\n{'=' * 60}")
    print(f"  RESULTS: {passed}/{total} passed")
    print(f"{'=' * 60}")

    if extract_result:
        print(f"\n  📊 Pipeline results:")
        print(f"     Extract:   {extract_result.get('count', '?')} rows from {extract_result.get('source', '?')}")
        if transform_result:
            print(f"     Transform: {transform_result.get('count', '?')} rows (values doubled)")
        print(f"     Load:      ✓ success")

    print(f"\n  callback_reported: false (expected — no GRAPHINGEST_API_URL on worker)")
    print(f"  To complete the E2E loop, set GRAPHINGEST_API_URL on the worker.\n")

    client.close()
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
