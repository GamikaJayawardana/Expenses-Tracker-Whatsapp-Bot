"""
Groq client and a thin, traced wrapper around chat completions.

Every call routed through `traced_chat` is timed and logged to the trace
table (see tracing.py), so the rest of the codebase never talks to the raw
client directly.
"""

import os
import time
import asyncio
from dotenv import load_dotenv
from groq import Groq

import tracing

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Models — Llama-4 Scout supports structured outputs and tool calling.
ROUTER_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
QUERY_MODEL  = "meta-llama/llama-4-scout-17b-16e-instruct"

client = Groq(api_key=GROQ_API_KEY)


async def traced_chat(call_type: str, **kwargs):
    """
    Run a chat completion off the event loop and log it.

    `call_type` is a short label (e.g. "router", "query") used to group
    calls in the metrics view. All other kwargs are forwarded to Groq.
    """
    model = kwargs.get("model")
    start = time.perf_counter()
    try:
        response = await asyncio.to_thread(client.chat.completions.create, **kwargs)
        latency_ms = (time.perf_counter() - start) * 1000
        tracing.log_call(call_type, model, getattr(response, "usage", None), latency_ms, success=True)
        return response
    except Exception as e:
        latency_ms = (time.perf_counter() - start) * 1000
        tracing.log_call(call_type, model, None, latency_ms, success=False, error=str(e))
        raise
