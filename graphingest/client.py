"""
GraphIngest Orchestrator Python Client

Communicates with the GraphIngest control plane (Next.js API routes)
to dispatch nodes, poll status, and manage flow control.

Authentication
--------------
All public endpoints require a per-user API key generated from the dashboard
(Settings → API Keys). Provide it via:

    >>> client = GraphIngestClient(api_key="sk_live_…")

…or by setting ``GRAPHINGEST_API_KEY`` in your environment.

The legacy ``report_task_completed`` / ``report_task_failed`` methods speak
to the worker-callback endpoint, which is now authenticated with a separate
``WORKER_CALLBACK_SECRET`` known only to the Cloud Run worker. They remain on
the client for in-cluster use cases (e.g. building a custom worker image) but
will fail with 401/403 when called from a regular SDK user.
"""

import os
import json
import hashlib
import logging
from typing import Any, Optional

import httpx

logger = logging.getLogger("graphingest")


class MissingAPIKeyError(RuntimeError):
    pass


class GraphIngestClient:
    """Client for the GraphIngest control plane."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        self.base_url = (
            base_url
            or os.environ.get("GRAPHINGEST_API_URL")
            or os.environ.get("INGEST_API_URL", "")
        ).rstrip("/")
        self.api_key = (
            api_key
            or os.environ.get("GRAPHINGEST_API_KEY")
            or os.environ.get("INGEST_API_KEY", "")
        )
        if not self.base_url:
            raise RuntimeError(
                "GraphIngestClient requires base_url or GRAPHINGEST_API_URL"
            )
        if not self.api_key:
            raise MissingAPIKeyError(
                "GraphIngestClient requires an api_key. Generate one in the "
                "dashboard under Settings → API Keys, then set GRAPHINGEST_API_KEY."
            )

        self._client = httpx.Client(
            base_url=self.base_url,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            timeout=30.0,
        )

    # ── Worker-internal (not for SDK users) ───────────────────────

    def report_task_completed(
        self,
        task_run_id: str,
        flow_run_id: str,
        result_url: Optional[str] = None,
        result_data: Optional[dict] = None,
        duration_ms: Optional[int] = None,
        logs: Optional[list[dict]] = None,
        artifacts: Optional[list[dict]] = None,
    ) -> dict:
        """Report a task as completed to the control plane.

        Internal — used by the Cloud Run worker. Requires
        ``WORKER_CALLBACK_SECRET`` as the bearer token, not the SDK key.
        """
        payload = {
            "task_run_id": task_run_id,
            "flow_run_id": flow_run_id,
            "status": "COMPLETED",
            "result_url": result_url,
            "result_data": result_data,
            "duration_ms": duration_ms,
            "logs": logs or [],
            "artifacts": artifacts or [],
        }
        response = self._client.post("/api/webhook/worker-callback", json=payload)
        response.raise_for_status()
        return response.json()

    def report_task_failed(
        self,
        task_run_id: str,
        flow_run_id: str,
        error_message: str,
        error_traceback: Optional[str] = None,
        logs: Optional[list[dict]] = None,
    ) -> dict:
        """Report a task as failed. Worker-internal — see ``report_task_completed``."""
        payload = {
            "task_run_id": task_run_id,
            "flow_run_id": flow_run_id,
            "status": "FAILED",
            "error_message": error_message,
            "error_traceback": error_traceback,
            "logs": logs or [],
        }
        response = self._client.post("/api/webhook/worker-callback", json=payload)
        response.raise_for_status()
        return response.json()

    # ── User-facing SDK API ───────────────────────────────────────

    def trigger_flow_run(
        self,
        flow_id: str,
        parameters: Optional[dict] = None,
    ) -> dict:
        """Trigger a new flow run."""
        response = self._client.post(
            f"/api/flows/{flow_id}/runs",
            json={"parameters": parameters or {}},
        )
        response.raise_for_status()
        return response.json()

    def dispatch_nodes(
        self,
        graph_run_id: str,
        node_key: str,
        inputs: list[Any],
    ) -> dict:
        """Dispatch one or more node executions within a graph run."""
        response = self._client.post(
            "/api/nodes/dispatch",
            json={
                "graphRunId": graph_run_id,
                "nodeKey": node_key,
                "inputs": inputs,
            },
        )
        response.raise_for_status()
        return response.json()

    def poll_task_runs(self, task_run_ids: list[str]) -> dict:
        """Check status of task runs."""
        response = self._client.post(
            "/api/nodes/status",
            json={"taskRunIds": task_run_ids},
        )
        response.raise_for_status()
        return response.json()

    def acquire_flow_control(
        self,
        graph_name: str,
        graph_run_id: str,
        concurrency_limit: int = 0,
        concurrency_key: str = "",
        throttle_limit: int = 0,
        throttle_period_seconds: int = 0,
        priority: int = 0,
    ) -> dict:
        response = self._client.post(
            "/api/flow-control",
            json={
                "action": "acquire",
                "graphName": graph_name,
                "graphRunId": graph_run_id,
                "concurrencyLimit": concurrency_limit,
                "concurrencyKey": concurrency_key,
                "throttleLimit": throttle_limit,
                "throttlePeriodSeconds": throttle_period_seconds,
                "priority": priority,
            },
        )
        response.raise_for_status()
        return response.json()

    def release_flow_control(self, graph_name: str, graph_run_id: str) -> dict:
        response = self._client.post(
            "/api/flow-control",
            json={
                "action": "release",
                "graphName": graph_name,
                "graphRunId": graph_run_id,
            },
        )
        response.raise_for_status()
        return response.json()

    # ── Deployment ────────────────────────────────────────────────

    def deploy(self, payload: dict) -> dict:
        """Upload a project payload to the GraphIngest control plane.

        ``payload`` is expected to contain ``files`` (a path → source map),
        ``language`` ("python" | "javascript"), and optionally
        ``requirements`` / ``dependencies`` and ``env_vars``. The platform
        builds an execution environment and registers the discovered
        functions for remote execution.
        """
        response = self._client.post(
            "/api/deploy",
            json=payload,
            timeout=300,
        )
        response.raise_for_status()
        return response.json()

    # ── Run telemetry (logs / artifacts) ──────────────────────────

    def send_logs(self, flow_run_id: str, entries: list[dict]) -> dict:
        """Stream a batch of log entries to the dashboard for a running flow.

        ``entries`` is a list of ``{level, message, task_run_id?, worker_id?, metadata?}``.
        The endpoint authenticates with the SDK key and accepts up to 500
        entries per call.
        """
        if not entries:
            return {"data": []}
        response = self._client.post(
            f"/api/runs/{flow_run_id}/logs",
            json=entries,
        )
        response.raise_for_status()
        return response.json()

    def create_artifact(
        self,
        flow_run_id: str,
        key: str,
        *,
        task_run_id: Optional[str] = None,
        type: str = "json",
        description: Optional[str] = None,
        data: Any = None,
        storage_url: Optional[str] = None,
    ) -> dict:
        """Attach an artifact to a flow run.

        ``data`` is stored inline (must serialize ≤ 256 KiB). For larger
        payloads upload to GCS first and pass ``storage_url`` instead.
        """
        response = self._client.post(
            f"/api/runs/{flow_run_id}/artifacts",
            json={
                "task_run_id": task_run_id,
                "key": key,
                "type": type,
                "description": description,
                "data": data,
                "storage_url": storage_url,
            },
        )
        response.raise_for_status()
        return response.json()

    # ── Hashing (must match server canonicalisation) ──────────────

    @staticmethod
    def compute_input_hash(task_key: str, input_data: Any) -> str:
        """Deterministic SHA-256 hash matching the server's canonical form.

        Object keys are sorted, lists preserve order, NaN/Infinity reject.
        Anything not JSON-representable falls back to the str() repr to keep
        the call from raising — but the resulting hash will not match the
        server, so cache hits won't happen.
        """

        def canonicalise(v: Any) -> Any:
            if isinstance(v, dict):
                return {k: canonicalise(v[k]) for k in sorted(v)}
            if isinstance(v, list):
                return [canonicalise(x) for x in v]
            return v

        try:
            payload = json.dumps(
                {"taskKey": task_key, "inputData": canonicalise(input_data)},
                separators=(",", ":"),
                sort_keys=True,
                allow_nan=False,
            )
        except (TypeError, ValueError):
            payload = json.dumps(
                {"taskKey": task_key, "inputData": str(input_data)},
                separators=(",", ":"),
            )
        return hashlib.sha256(payload.encode()).hexdigest()

    def close(self):
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
