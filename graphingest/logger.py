"""
Streaming Run Logger

A logging.Handler that POSTs log entries to the control plane in real-time
via /api/runs/{flow_run_id}/logs, enabling live log streaming in the dashboard.

Logs are buffered and flushed either when the buffer reaches a threshold
or after a configurable interval, whichever comes first.
"""

from __future__ import annotations

import os
import time
import logging
import threading
from typing import Optional

from .client import GraphIngestClient, MissingAPIKeyError

_logger = logging.getLogger("graphingest.logger")


class StreamingLogHandler(logging.Handler):
    """
    A logging handler that streams log records to the GraphIngest control plane.

    Buffers records and flushes them in batches to minimize HTTP overhead.
    Flush happens when buffer_size is reached OR every flush_interval seconds.
    """

    def __init__(
        self,
        flow_run_id: str,
        task_run_id: Optional[str] = None,
        client: Optional[GraphIngestClient] = None,
        buffer_size: int = 20,
        flush_interval: float = 2.0,
    ):
        super().__init__()
        self.flow_run_id = flow_run_id
        self.task_run_id = task_run_id
        # Lazily fall back to a no-op handler when no API key is configured,
        # so logging from inside a graph never crashes the user's process.
        if client is None:
            try:
                client = GraphIngestClient()
            except (MissingAPIKeyError, RuntimeError) as e:
                _logger.warning(
                    "graphingest streaming disabled: %s", e,
                )
                client = None
        self.client = client
        self.buffer_size = buffer_size
        self.flush_interval = flush_interval

        self._buffer: list[dict] = []
        self._lock = threading.Lock()
        self._last_flush = time.monotonic()

        # Background flush thread
        self._stop_event = threading.Event()
        self._flush_thread = threading.Thread(
            target=self._flush_loop, daemon=True, name="graphingest-log-flusher"
        )
        self._flush_thread.start()

    def emit(self, record: logging.LogRecord) -> None:
        entry = {
            "flow_run_id": self.flow_run_id,
            "task_run_id": self.task_run_id,
            "level": record.levelname,
            "message": self.format(record),
            "worker_id": os.environ.get("WORKER_ID", "sdk"),
            "metadata": {
                "logger": record.name,
                "filename": record.filename,
                "lineno": record.lineno,
            },
        }

        with self._lock:
            self._buffer.append(entry)
            if len(self._buffer) >= self.buffer_size:
                self._do_flush()

    def _flush_loop(self) -> None:
        """Periodically flush buffered logs."""
        while not self._stop_event.is_set():
            self._stop_event.wait(timeout=self.flush_interval)
            with self._lock:
                if self._buffer:
                    self._do_flush()

    def _do_flush(self) -> None:
        """Send buffered logs to the control plane. Must be called under _lock."""
        if not self._buffer:
            return

        batch = self._buffer[:]
        self._buffer.clear()
        self._last_flush = time.monotonic()

        if self.client is None:
            return
        try:
            self.client.send_logs(self.flow_run_id, batch)
        except Exception as e:
            _logger.debug(f"Failed to flush logs: {e}")

    def flush(self) -> None:
        """Force flush any remaining buffered logs."""
        with self._lock:
            self._do_flush()

    def close(self) -> None:
        """Stop the background flusher and flush remaining logs."""
        self._stop_event.set()
        self._flush_thread.join(timeout=5.0)
        self.flush()
        super().close()

    @property
    def buffered_records(self) -> list[dict]:
        """Return a copy of currently buffered records (for batch reporting)."""
        with self._lock:
            return self._buffer[:]


def get_run_logger(
    flow_run_id: Optional[str] = None,
    task_run_id: Optional[str] = None,
    name: str = "graphingest.run",
) -> logging.Logger:
    """
    Get a logger that streams to the GraphIngest dashboard in real-time.

    Usage inside a @graph or @node:
        from graphingest import get_run_logger
        logger = get_run_logger()
        logger.info("Processing started")

    Falls back to a standard logger if no flow_run_id is available.
    """
    from .context import GraphRunContext, NodeRunContext

    if flow_run_id is None:
        ctx = GraphRunContext.get()
        if ctx:
            flow_run_id = ctx.graph_run_id

    if task_run_id is None:
        tctx = NodeRunContext.get()
        if tctx:
            task_run_id = tctx.node_run_id

    run_logger = logging.getLogger(name)
    run_logger.setLevel(logging.DEBUG)

    if flow_run_id:
        # Check if we already have a StreamingLogHandler for this run
        for handler in run_logger.handlers:
            if (
                isinstance(handler, StreamingLogHandler)
                and handler.flow_run_id == flow_run_id
            ):
                return run_logger

        handler = StreamingLogHandler(
            flow_run_id=flow_run_id,
            task_run_id=task_run_id,
        )
        handler.setFormatter(logging.Formatter("%(asctime)s [%(name)s] %(message)s"))
        run_logger.addHandler(handler)

    return run_logger
