# GraphIngest Python SDK

Python SDK for the [GraphIngest](https://graphingest.io) workflow orchestration platform — define pipeline nodes and graphs with decorators, deploy with one call.

## Installation

```bash
pip install graphingest

# With AI agent support:
pip install graphingest[react]

# With LangGraph integration:
pip install graphingest[langgraph]
```

## Quick Start

```python
from graphingest import node, graph, deploy, RetryPolicy

@node(name="extract", cache_ttl=3600)
def extract(url: str) -> dict:
    return {"url": url, "rows": 100}

@node(name="transform")
def transform(data: dict) -> dict:
    return {"cleaned": True, **data}

@graph(
    name="etl-pipeline",
    retry_policy=RetryPolicy(max_retries=3, delay_seconds=1, backoff_factor=2, jitter=True),
    timeout_seconds=600,
)
def pipeline(url: str):
    data = extract(url)
    return transform(data)

deploy()
```

## Features

- **`@node` decorator** — Define individual tasks with caching, retries, and timeouts
- **`@graph` decorator** — Compose nodes into pipelines with retry policies and state hooks
- **`.map()` fan-out** — Parallel execution across multiple inputs
- **`.arun()` async dispatch** — Fire-and-forget with `NodeFuture`
- **`deploy()`** — Push code to platform with zero config
- **Input-based caching** — SHA-256 hashing with configurable TTL
- **Incremental retries** — Resume from failure point, not from scratch
- **AI agents** — Built-in ReAct agents and LangGraph integration

## Fan-Out (.map)

```python
@graph(name="parallel-pipeline")
def pipeline(urls: list[str]):
    results = extract.map(urls)  # dispatches N parallel workers
    return results
```

## Async Dispatch (.arun)

```python
@graph(name="async-pipeline")
def pipeline(data: dict):
    future = transform.arun(data)   # returns immediately
    do_other_stuff()
    result = future.result(timeout=120)  # block until ready
    return result
```

## AI Agent Orchestration

```python
from graphingest.react import agent

@agent(name="researcher", tools=[search, scrape], model="standard")
def research(query: str) -> str:
    """You are a research assistant."""
    ...

answer = research.run("What are the latest advances in fusion energy?")
```

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GRAPHINGEST_API_URL` | Yes | Control plane URL |
| `GRAPHINGEST_API_KEY` | Yes | API key |

## Documentation

Full documentation at [graphingest.io/docs](https://graphingest.io/docs)

## License

MIT
