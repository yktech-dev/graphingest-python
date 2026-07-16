"""
GraphIngest ReAct Agent Primitives

Automatic tool-call routing: @node functions become LLM tools.
The SDK generates tool schemas from function signatures, routes
LLM tool calls to node.run() on managed infrastructure, and
feeds results back to the LLM in a ReAct loop.

Supported models:
    Platform-managed (no API key needed, billed to your account):
    - "standard"  — fast and cost-effective (default)
    - "high"      — premium quality for complex reasoning

    Bring Your Own Key (BYOK):
    - OpenAI:    gpt-4o, gpt-4o-mini, o1, o3, ...
    - Anthropic: claude-3.5-sonnet, claude-3-opus, claude-3-haiku, ...
    - Google:    gemini-2.5-flash, gemini-2.5-pro, ...

Usage:
    from graphingest import node, deploy
    from graphingest.react import agent

    @node(name="search")
    def search(query: str) -> list[str]:
        \"""Search the web.\"""
        return google_search(query)

    @agent(name="researcher", model="standard", tools=[search])
    def research(query: str) -> str:
        \"""Research a topic.\"""
        ...

    deploy()
    answer = research.run("What is quantum computing?")

Requires (install the one you need):
    pip install graphingest[react]      # Platform tiers (standard, high) — no extra deps
    pip install graphingest[openai]     # BYOK: OpenAI models
    pip install graphingest[anthropic]  # BYOK: Claude models
    pip install graphingest[google]     # BYOK: Gemini models
"""

from __future__ import annotations

import inspect
import json
import logging
import os
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, get_type_hints

from .task import graph, _node_registry

logger = logging.getLogger("graphingest.react")

# Python type → JSON Schema type mapping
_TYPE_MAP = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


# ---------------------------------------------------------------------------
# tools_from_nodes: auto-generate tool schemas from @node functions
# ---------------------------------------------------------------------------

def tools_from_nodes(nodes: list[Callable]) -> list[dict]:
    """Convert a list of @node-decorated functions into tool schemas.

    Inspects each function's signature, type hints, and docstring to generate
    a tool definition. The returned format is provider-agnostic (OpenAI style)
    and gets converted to provider-specific format internally.

    Args:
        nodes: List of @node-decorated functions.

    Returns:
        List of tool schema dicts (OpenAI format).

    Example:
        schemas = tools_from_nodes([search, scrape, summarize])
    """
    tools = []
    for fn in nodes:
        node_key = getattr(fn, "_node_key", fn.__name__)
        original = getattr(fn, "_original_fn", fn)

        sig = inspect.signature(original)
        try:
            hints = get_type_hints(original)
        except Exception:
            hints = {}

        properties = {}
        required = []
        for param_name, param in sig.parameters.items():
            param_type = hints.get(param_name, str)
            json_type = _TYPE_MAP.get(param_type, "string")

            prop: dict[str, Any] = {"type": json_type}

            if param.default is not inspect.Parameter.empty:
                prop["default"] = param.default
            else:
                required.append(param_name)

            properties[param_name] = prop

        description = ""
        if original.__doc__:
            doc_lines = original.__doc__.strip().split("\n")
            description = doc_lines[0].strip()

            for line in doc_lines[1:]:
                line = line.strip()
                for pname in properties:
                    if line.startswith(f"{pname}:") or line.startswith(f":param {pname}:"):
                        desc = line.split(":", 2)[-1].strip()
                        properties[pname]["description"] = desc

        tool = {
            "type": "function",
            "function": {
                "name": node_key,
                "description": description or f"Execute the {node_key} tool",
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }
        tools.append(tool)

    return tools


# ---------------------------------------------------------------------------
# LLM Provider abstraction
# ---------------------------------------------------------------------------

@dataclass
class ToolCall:
    """Normalized tool call from any LLM provider."""
    id: str
    name: str
    arguments: dict


@dataclass
class LLMResponse:
    """Normalized LLM response from any provider."""
    content: Optional[str]
    tool_calls: list[ToolCall]
    raw: Any = None


class LLMProvider(ABC):
    """Abstract base for LLM providers."""

    @abstractmethod
    def chat(
        self,
        messages: list[dict],
        tools: list[dict],
        temperature: float,
        model: str,
    ) -> LLMResponse:
        ...

    @abstractmethod
    def append_assistant(self, messages: list[dict], response: LLMResponse) -> None:
        """Append the assistant's response to the message history."""
        ...

    @abstractmethod
    def append_tool_result(self, messages: list[dict], tool_call: ToolCall, result: str) -> None:
        """Append a tool result to the message history."""
        ...


class OpenAIProvider(LLMProvider):
    """OpenAI (gpt-*) and OpenAI-compatible endpoints (platform proxy)."""

    def __init__(self, base_url: Optional[str] = None, api_key: Optional[str] = None):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai is required. Install with: pip install openai")
        kwargs: dict[str, Any] = {}
        if base_url:
            kwargs["base_url"] = base_url
        if api_key:
            kwargs["api_key"] = api_key
        self._client = OpenAI(**kwargs)

    def chat(self, messages, tools, temperature, model):
        response = self._client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools if tools else None,
            temperature=temperature,
        )
        choice = response.choices[0]
        msg = choice.message
        tool_calls = []
        if msg.tool_calls:
            for tc in msg.tool_calls:
                tool_calls.append(ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=json.loads(tc.function.arguments),
                ))
        return LLMResponse(content=msg.content, tool_calls=tool_calls, raw=msg)

    def append_assistant(self, messages, response):
        msg: dict[str, Any] = {"role": "assistant"}
        if response.content:
            msg["content"] = response.content
        if response.tool_calls:
            msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
                }
                for tc in response.tool_calls
            ]
        messages.append(msg)

    def append_tool_result(self, messages, tool_call, result):
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": result,
        })


class AnthropicProvider(LLMProvider):
    """Anthropic (claude-*) provider."""

    def __init__(self):
        try:
            import anthropic
        except ImportError:
            raise ImportError("anthropic is required. Install with: pip install anthropic")
        self._client = anthropic.Anthropic()

    def chat(self, messages, tools, temperature, model):
        # Anthropic uses a different tool format
        anthropic_tools = []
        for t in tools:
            fn = t["function"]
            anthropic_tools.append({
                "name": fn["name"],
                "description": fn.get("description", ""),
                "input_schema": fn["parameters"],
            })

        # Anthropic separates system prompt from messages
        system = ""
        filtered_messages = []
        for m in messages:
            if m["role"] == "system":
                system = m["content"]
            else:
                filtered_messages.append(m)

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": filtered_messages,
            "max_tokens": 4096,
            "temperature": temperature,
        }
        if anthropic_tools:
            kwargs["tools"] = anthropic_tools
        if system:
            kwargs["system"] = system

        response = self._client.messages.create(**kwargs)

        # Parse response — Anthropic returns content blocks
        content = ""
        tool_calls = []
        for block in response.content:
            if block.type == "text":
                content += block.text
            elif block.type == "tool_use":
                tool_calls.append(ToolCall(
                    id=block.id,
                    name=block.name,
                    arguments=block.input if isinstance(block.input, dict) else json.loads(block.input),
                ))

        return LLMResponse(content=content or None, tool_calls=tool_calls, raw=response)

    def append_assistant(self, messages, response):
        content_blocks = []
        if response.content:
            content_blocks.append({"type": "text", "text": response.content})
        for tc in response.tool_calls:
            content_blocks.append({
                "type": "tool_use",
                "id": tc.id,
                "name": tc.name,
                "input": tc.arguments,
            })
        messages.append({"role": "assistant", "content": content_blocks})

    def append_tool_result(self, messages, tool_call, result):
        messages.append({
            "role": "user",
            "content": [{
                "type": "tool_result",
                "tool_use_id": tool_call.id,
                "content": result,
            }],
        })


class GeminiProvider(LLMProvider):
    """Google Gemini provider."""

    def __init__(self):
        try:
            from google import genai
        except ImportError:
            raise ImportError(
                "google-genai is required. Install with: pip install google-genai"
            )
        self._genai = genai
        self._client = genai.Client()

    def chat(self, messages, tools, temperature, model):
        from google.genai import types

        # Convert tool schemas to Gemini format
        function_declarations = []
        for t in tools:
            fn = t["function"]
            function_declarations.append(types.FunctionDeclaration(
                name=fn["name"],
                description=fn.get("description", ""),
                parameters=fn["parameters"],
            ))

        gemini_tools = [types.Tool(function_declarations=function_declarations)] if function_declarations else None

        # Build Gemini contents from messages
        system_instruction = None
        contents = []
        for m in messages:
            if m["role"] == "system":
                system_instruction = m["content"]
            elif m["role"] == "user":
                contents.append(types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=m["content"])],
                ))
            elif m["role"] == "assistant":
                parts = []
                if isinstance(m.get("content"), str) and m["content"]:
                    parts.append(types.Part.from_text(text=m["content"]))
                for tc in m.get("_tool_calls", []):
                    parts.append(types.Part.from_function_call(
                        name=tc["name"],
                        args=tc["arguments"],
                    ))
                if parts:
                    contents.append(types.Content(role="model", parts=parts))
            elif m["role"] == "tool_result":
                contents.append(types.Content(
                    role="user",
                    parts=[types.Part.from_function_response(
                        name=m["name"],
                        response={"result": m["content"]},
                    )],
                ))

        config = types.GenerateContentConfig(
            temperature=temperature,
            tools=gemini_tools,
        )
        if system_instruction:
            config.system_instruction = system_instruction

        response = self._client.models.generate_content(
            model=model,
            contents=contents,
            config=config,
        )

        # Parse response
        content = ""
        tool_calls = []
        if response.candidates and response.candidates[0].content:
            for part in response.candidates[0].content.parts:
                if part.text:
                    content += part.text
                elif part.function_call:
                    fc = part.function_call
                    tool_calls.append(ToolCall(
                        id=f"call_{uuid.uuid4().hex[:8]}",
                        name=fc.name,
                        arguments=dict(fc.args) if fc.args else {},
                    ))

        return LLMResponse(content=content or None, tool_calls=tool_calls, raw=response)

    def append_assistant(self, messages, response):
        msg: dict[str, Any] = {"role": "assistant", "content": response.content or ""}
        msg["_tool_calls"] = [{"name": tc.name, "arguments": tc.arguments} for tc in response.tool_calls]
        messages.append(msg)

    def append_tool_result(self, messages, tool_call, result):
        messages.append({
            "role": "tool_result",
            "name": tool_call.name,
            "content": result,
        })


def _get_provider(model: str) -> tuple[LLMProvider, str]:
    """Auto-detect and return the correct LLM provider based on model name.

    Returns:
        (provider_instance, resolved_model_name)
    """
    # Platform-managed tiers — user never sees the underlying model
    _PLATFORM_TIERS = {
        "standard": "gemini-2.5-flash",
        "high": "gemini-2.5-pro",
    }

    if model in _PLATFORM_TIERS:
        platform_url = os.environ.get("GRAPHINGEST_API_URL", "")
        api_key = os.environ.get("GRAPHINGEST_API_KEY", "")
        if not platform_url:
            raise ValueError(
                f"model='{model}' requires GRAPHINGEST_API_URL. "
                "Run deploy() first or set the env var."
            )
        return OpenAIProvider(
            base_url=f"{platform_url}/llm/v1",
            api_key=api_key,
        ), _PLATFORM_TIERS[model]

    # BYOK — user provides their own API key for a specific model
    model_lower = model.lower()

    if model_lower.startswith(("gpt-", "o1", "o3", "o4")):
        return OpenAIProvider(), model

    if model_lower.startswith("claude"):
        return AnthropicProvider(), model

    if model_lower.startswith("gemini"):
        return GeminiProvider(), model

    # Default: try OpenAI (works for OpenAI-compatible providers like Together, Groq, etc.)
    logger.info(f"Unknown model prefix '{model}', defaulting to OpenAI provider")
    return OpenAIProvider(), model


# ---------------------------------------------------------------------------
# react: built-in ReAct loop with automatic tool routing
# ---------------------------------------------------------------------------

@dataclass
class ReactResult:
    """Result of a react() loop execution."""
    answer: str
    tool_calls: list[dict] = field(default_factory=list)
    steps: int = 0
    elapsed_seconds: float = 0.0
    model: str = ""


def react(
    query: str,
    tools: list[Callable],
    *,
    model: str = "standard",
    system_prompt: str = "",
    max_iterations: int = 10,
    temperature: float = 0.0,
    parallel_tool_calls: bool = True,
) -> ReactResult:
    """Run a ReAct loop: LLM reasons → picks tools → tools execute on managed infra → repeat.

    The LLM automatically decides which @node tools to call based on the query.
    Each tool call is routed to node.run() on managed infrastructure with retries,
    caching, and observability.

    Supported models:
        Platform-managed (no API key needed):
        - "standard"  — fast and cost-effective (default)
        - "high"      — premium quality for complex reasoning

        BYOK (bring your own API key):
        - "gpt-4o", "gpt-4o-mini", "o1", "o3-mini"
        - "claude-3.5-sonnet", "claude-3-opus", "claude-3-haiku"
        - "gemini-2.5-flash", "gemini-2.5-pro"

    Args:
        query:                User's question or task.
        tools:                List of @node-decorated functions to make available.
        model:                "standard", "high", or a specific BYOK model name (default: "standard").
        system_prompt:        System prompt prepended to the conversation.
        max_iterations:       Max reasoning loops before stopping (default: 10).
        temperature:          LLM temperature (default: 0.0).
        parallel_tool_calls:  Execute multiple tool calls in parallel via .arun() (default: True).

    Returns:
        ReactResult with the final answer, tool call log, step count, and timing.

    Example:
        result = react("Research fusion energy", tools=[search, scrape])  # uses "standard"
        result = react("Complex analysis", tools=[search], model="high")   # premium tier
        result = react("Summarize this", tools=[search], model="gpt-4o")   # BYOK
    """
    provider, resolved_model = _get_provider(model)
    tool_schemas = tools_from_nodes(tools)

    # Build tool lookup: node_key → @node function
    tool_map: dict[str, Callable] = {}
    for fn in tools:
        node_key = getattr(fn, "_node_key", fn.__name__)
        tool_map[node_key] = fn

    # Build initial messages
    messages: list[dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": query})

    all_tool_calls: list[dict] = []
    start = time.monotonic()

    for step in range(max_iterations):
        # Call LLM via provider
        response = provider.chat(messages, tool_schemas, temperature, resolved_model)

        # Append assistant message to history
        provider.append_assistant(messages, response)

        # If no tool calls → LLM is done
        if not response.tool_calls:
            elapsed = time.monotonic() - start
            return ReactResult(
                answer=response.content or "",
                tool_calls=all_tool_calls,
                steps=step + 1,
                elapsed_seconds=round(elapsed, 2),
                model=model,
            )

        # Execute tool calls
        if parallel_tool_calls and len(response.tool_calls) > 1:
            # Parallel execution via .arun()
            futures = []
            for tc in response.tool_calls:
                tool_fn = tool_map.get(tc.name)
                if tool_fn is None:
                    futures.append((tc, None, f"Unknown tool: {tc.name}"))
                    continue

                logger.info(f"[react] Dispatching tool '{tc.name}' async")
                future = tool_fn.arun(_pack_args(tc.arguments))
                futures.append((tc, future, tc.arguments))

            for tc, future, args_or_error in futures:
                if future is None:
                    tool_result = str(args_or_error)
                else:
                    try:
                        tool_result = str(future.result(timeout=300))
                    except Exception as e:
                        tool_result = f"Error: {e}"

                all_tool_calls.append({"tool": tc.name, "args": args_or_error, "result": tool_result})
                provider.append_tool_result(messages, tc, tool_result)
        else:
            # Sequential execution via .run()
            for tc in response.tool_calls:
                tool_fn = tool_map.get(tc.name)
                if tool_fn is None:
                    tool_result = f"Unknown tool: {tc.name}"
                else:
                    logger.info(f"[react] Calling tool '{tc.name}'")
                    try:
                        tool_result = str(tool_fn.run(_pack_args(tc.arguments)))
                    except Exception as e:
                        tool_result = f"Error: {e}"

                all_tool_calls.append({"tool": tc.name, "args": tc.arguments, "result": tool_result})
                provider.append_tool_result(messages, tc, tool_result)

    # Max iterations reached
    elapsed = time.monotonic() - start
    return ReactResult(
        answer=f"Max iterations ({max_iterations}) reached without a final answer.",
        tool_calls=all_tool_calls,
        steps=max_iterations,
        elapsed_seconds=round(elapsed, 2),
        model=model,
    )


# ---------------------------------------------------------------------------
# @agent decorator: @graph + react() in one line
# ---------------------------------------------------------------------------

def agent(
    name: str,
    *,
    tools: list[Callable],
    model: str = "standard",
    system_prompt: str = "",
    max_iterations: int = 10,
    temperature: float = 0.0,
    parallel_tool_calls: bool = True,
    timeout_seconds: int = 600,
    retry_policy: Any = None,
    tags: Optional[list[str]] = None,
):
    """Decorator that creates an AI agent backed by @node tools.

    Combines @graph with a built-in ReAct loop. The LLM automatically
    decides which @node tools to call. Each tool call runs on managed
    infrastructure.

    Args:
        name:                 Agent name (shown in dashboard).
        tools:                List of @node functions available as tools.
        model:                "standard", "high", or a BYOK model name (default: "standard").
        system_prompt:        System prompt. If empty, uses the function's docstring.
        max_iterations:       Max ReAct loop iterations (default: 10).
        temperature:          LLM temperature (default: 0.0).
        parallel_tool_calls:  Run multiple tool calls in parallel (default: True).
        timeout_seconds:      Graph timeout (default: 600).
        retry_policy:         Optional RetryPolicy for the graph.
        tags:                 Metadata tags.

    Usage:
        @agent(name="researcher", tools=[search, scrape])
        def research(query: str) -> str:
            \"""You are a research assistant. Use search and scrape to answer questions.\"""
            ...

        deploy()
        answer = research.run("What is quantum computing?")
    """
    def decorator(fn: Callable) -> Callable:
        # Use docstring as system prompt if not provided
        prompt = system_prompt or (fn.__doc__ or "").strip()

        # Build the graph function that runs react()
        def _agent_fn(query: str) -> str:
            result = react(
                query=query,
                tools=tools,
                model=model,
                system_prompt=prompt,
                max_iterations=max_iterations,
                temperature=temperature,
                parallel_tool_calls=parallel_tool_calls,
            )
            logger.info(
                f"[agent:{name}] Completed in {result.elapsed_seconds}s "
                f"({result.steps} steps, {len(result.tool_calls)} tool calls)"
            )
            return result.answer

        # Apply @graph decorator
        graph_kwargs: dict[str, Any] = {"name": name}
        if timeout_seconds:
            graph_kwargs["timeout_seconds"] = timeout_seconds
        if retry_policy:
            graph_kwargs["retry_policy"] = retry_policy
        if tags:
            graph_kwargs["tags"] = tags

        wrapped = graph(**graph_kwargs)(_agent_fn)

        # Preserve original function metadata
        wrapped.__doc__ = fn.__doc__
        wrapped.__name__ = fn.__name__

        return wrapped

    return decorator


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _pack_args(args: dict) -> Any:
    """Pack tool call arguments for node.run().

    If the tool has a single argument, unwrap it.
    Otherwise pass the dict.
    """
    if len(args) == 1:
        return list(args.values())[0]
    return args


def _message_to_dict(message: Any) -> dict:
    """Convert an OpenAI ChatCompletionMessage to a plain dict for the messages list."""
    msg: dict[str, Any] = {
        "role": message.role,
    }
    if message.content:
        msg["content"] = message.content
    if message.tool_calls:
        msg["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            }
            for tc in message.tool_calls
        ]
    return msg
