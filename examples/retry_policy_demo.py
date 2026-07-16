"""
GraphIngest SDK — Retry Policy Demo
=====================================

Shows how exponential backoff + jitter works with the new RetryPolicy.

Prerequisites:
  pip install graphingest
  export GRAPHINGEST_API_URL=http://localhost:3000
  export GRAPHINGEST_API_KEY=your-key
"""

from graphingest import graph, RetryPolicy


# ---------------------------------------------------------------------------
# Example 1: Simple fixed-delay retries (legacy style, still works)
# ---------------------------------------------------------------------------

@graph(name="legacy-retries", retries=3, retry_delay_seconds=5)
def legacy_pipeline(source: str) -> str:
    """Uses the old-style fixed delay: retries 3 times, 5s between each."""
    if source == "fail":
        raise ValueError("Simulated failure")
    return f"done:{source}"


# ---------------------------------------------------------------------------
# Example 2: Exponential backoff (recommended)
# ---------------------------------------------------------------------------

@graph(
    name="backoff-pipeline",
    retry_policy=RetryPolicy(
        max_retries=4,
        delay_seconds=1,       # start at 1s
        backoff_factor=2,      # double each time
        max_delay_seconds=30,  # cap at 30s
        jitter=True,           # ±50% randomization
    ),
)
def backoff_pipeline(source: str) -> str:
    """
    Retry delays will be approximately:
      Attempt 1 fails → wait ~1s   (0.5s - 1.5s with jitter)
      Attempt 2 fails → wait ~2s   (1.0s - 3.0s with jitter)
      Attempt 3 fails → wait ~4s   (2.0s - 6.0s with jitter)
      Attempt 4 fails → wait ~8s   (4.0s - 12.0s with jitter)
      Attempt 5 fails → gives up, calls on_failure hooks
    """
    if source == "fail":
        raise ValueError("Simulated failure")
    return f"done:{source}"


# ---------------------------------------------------------------------------
# Example 3: Aggressive retries for flaky APIs
# ---------------------------------------------------------------------------

@graph(
    name="flaky-api-pipeline",
    retry_policy=RetryPolicy(
        max_retries=6,
        delay_seconds=0.5,        # start fast
        backoff_factor=3,          # triple each time
        max_delay_seconds=60,      # cap at 1 minute
        jitter=True,
    ),
    timeout_seconds=300,  # overall 5-minute timeout
)
def flaky_api_pipeline(endpoint: str) -> dict:
    """
    Retry delays: ~0.5s, ~1.5s, ~4.5s, ~13.5s, ~40.5s, ~60s (capped)
    Great for external APIs that have intermittent failures.
    """
    # your code here
    return {"endpoint": endpoint, "status": "ok"}


if __name__ == "__main__":
    # This will succeed
    print(backoff_pipeline("good-data"))

    # This will retry 4 times with exponential backoff, then fail
    try:
        backoff_pipeline("fail")
    except ValueError as e:
        print(f"Expected failure after retries: {e}")
