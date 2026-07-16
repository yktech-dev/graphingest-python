"""
GraphIngest SDK — Feature Demo
================================

Demonstrates the 3 new SDK capabilities:
  1. Exponential backoff + jitter  (RetryPolicy)
  2. .map() / .arun() fan-out      (parallel dispatch)
  3. Subgraphs                     (nested @graph inside @graph)

Prerequisites:
  pip install graphingest            # or: pip install -e sdk/python
  export GRAPHINGEST_API_URL=http://localhost:3000
  export GRAPHINGEST_API_KEY=your-key
"""

from graphingest import node, graph, deploy, RetryPolicy, GraphRunContext


# ---------------------------------------------------------------------------
# 1. Define some @node functions
# ---------------------------------------------------------------------------

@node(name="extract")
def extract(url: str) -> dict:
    """Simulate fetching data from a URL."""
    print(f"  [extract] Fetching {url}")
    # In real code: requests.get(url).json()
    return {"url": url, "rows": 42}


@node(name="transform")
def transform(data: dict) -> dict:
    """Simulate transforming extracted data."""
    print(f"  [transform] Processing {data['url']}")
    return {**data, "transformed": True}


@node(name="load")
def load(record: dict) -> str:
    """Simulate loading data into a warehouse."""
    print(f"  [load] Loaded {record['url']} ({record['rows']} rows)")
    return f"loaded:{record['url']}"


# ---------------------------------------------------------------------------
# 2. Subgraph: a reusable ETL sub-pipeline
# ---------------------------------------------------------------------------

@graph(
    name="etl-single-source",
    retry_policy=RetryPolicy(
        max_retries=2,
        delay_seconds=1,
        backoff_factor=2,    # delays: ~1s, ~2s
        jitter=True,
    ),
)
def etl_single_source(url: str) -> str:
    """ETL pipeline for a single data source.

    Can be used standalone OR as a subgraph inside a larger pipeline.
    """
    data = extract(url)
    transformed = transform(data)
    result = load(transformed)
    return result


# ---------------------------------------------------------------------------
# 3. Top-level graph with .map(), .arun(), and subgraph calls
# ---------------------------------------------------------------------------

@graph(
    name="multi-source-pipeline",
    retry_policy=RetryPolicy(
        max_retries=3,
        delay_seconds=2,
        backoff_factor=3,       # delays: ~2s, ~6s, ~18s
        max_delay_seconds=30,
        jitter=True,
    ),
    timeout_seconds=600,
)
def multi_source_pipeline(urls: list[str]) -> dict:
    """
    Demonstrates all 3 new features:

    - .map()     → fan-out extract across N parallel invocations
    - .arun()    → fire-and-forget a single node, get a future back
    - subgraph   → call another @graph from within this @graph
    """

    ctx = GraphRunContext.get()
    print(f"Pipeline started: run={ctx.graph_run_id}")

    # ── Feature 1: .map() fan-out ──────────────────────────────────────
    # Dispatches len(urls) parallel invocations of the
    # "extract" node. Blocks until all complete. Returns ordered results.
    print(f"\n→ Fan-out: extracting {len(urls)} sources in parallel...")
    extracted = extract.map(urls)
    print(f"  Got {len(extracted)} results")

    # ── Feature 2: .arun() async dispatch ─────────────────────
    # Fire off a transform in the background while we do other work.
    print("\n→ Dispatching async transform...")
    future = transform.arun(extracted[0])
    # ... do other work here ...
    first_transformed = future.result(timeout=60)  # blocks until done
    print(f"  Async result: {first_transformed}")

    # ── Feature 3: Subgraph ────────────────────────────────────────────
    # Call another @graph from within this @graph.
    # It gets its own graph_run_id with parent_graph_run_id linking back.
    print("\n→ Running subgraph for remaining sources...")
    sub_results = []
    for url in urls[1:]:
        result = etl_single_source(url)
        sub_results.append(result)

    return {
        "total_sources": len(urls),
        "first_result": first_transformed,
        "sub_results": sub_results,
    }


# ---------------------------------------------------------------------------
# Run it
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Push code to platform
    deploy()

    # Execute on managed infrastructure
    result = multi_source_pipeline(
        urls=[
            "https://api.example.com/dataset-a",
            "https://api.example.com/dataset-b",
            "https://api.example.com/dataset-c",
        ]
    )
    print(f"\n✓ Pipeline complete: {result}")
