"""
LangChain / Groq chat models and a traced invoke helper.

The rest of the codebase builds LangChain runnables (structured output,
tool binding) and runs them through `traced_invoke`, which times each call
and logs token usage / cost to the trace table (see tracing.py).
"""

import os
import time
from dotenv import load_dotenv
from langchain_groq import ChatGroq

import tracing

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Models — Llama 3.3 70B supports structured outputs and tool calling on Groq.
ROUTER_MODEL = "llama-3.3-70b-versatile"
QUERY_MODEL  = "llama-3.3-70b-versatile"


def get_chat(model: str, temperature: float = 0.2, max_tokens: int = 1024) -> ChatGroq:
    """Build a ChatGroq model. Callers add structured output / tools on top."""
    return ChatGroq(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        api_key=GROQ_API_KEY,
    )


class _Usage:
    """Adapter so LangChain's usage_metadata dict fits tracing.log_call."""
    def __init__(self, prompt_tokens: int, completion_tokens: int, total_tokens: int):
        self.prompt_tokens     = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens      = total_tokens


def _extract_usage(result) -> _Usage:
    """
    Pull token usage from an invoke result. `result` is either an AIMessage
    or the {"raw", "parsed", "parsing_error"} dict from structured output.
    """
    ai = result.get("raw") if isinstance(result, dict) else result
    um = getattr(ai, "usage_metadata", None) or {}
    prompt_tokens     = um.get("input_tokens", 0) or 0
    completion_tokens = um.get("output_tokens", 0) or 0
    total_tokens      = um.get("total_tokens", prompt_tokens + completion_tokens) or (prompt_tokens + completion_tokens)
    return _Usage(prompt_tokens, completion_tokens, total_tokens)


async def traced_invoke(call_type: str, runnable, messages, model: str):
    """
    Run a LangChain runnable asynchronously and log the call.

    `call_type` groups calls in the metrics view ("router", "query", ...);
    `model` is used for cost estimation.
    """
    start = time.perf_counter()
    try:
        result = await runnable.ainvoke(messages)
        latency_ms = (time.perf_counter() - start) * 1000
        tracing.log_call(call_type, model, _extract_usage(result), latency_ms, success=True)
        return result
    except Exception as e:
        latency_ms = (time.perf_counter() - start) * 1000
        tracing.log_call(call_type, model, None, latency_ms, success=False, error=str(e))
        raise
