"""
Lightweight LLM tracing.

Every model call is written to a local SQLite trace table so we can track
cost, latency, token usage and error rates without pulling in an external
service. Aggregates are exposed through the /metrics endpoint in main.py.
"""

import os
import sqlite3
import threading
from datetime import datetime

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DB_PATH = os.getenv("TRACE_DB_PATH", "traces.db")
_LOCK   = threading.Lock()

# USD per 1,000,000 tokens, keyed by model. Update these to match the current
# Groq price sheet — they are only used to estimate spend.
MODEL_PRICES = {
    "llama-3.3-70b-versatile":                   {"input": 0.59, "output": 0.79},
    "meta-llama/llama-4-scout-17b-16e-instruct": {"input": 0.11, "output": 0.34},
}
_DEFAULT_PRICE = {"input": 0.0, "output": 0.0}


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
def init_db() -> None:
    """Create the trace table if it doesn't exist yet."""
    with _LOCK, sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS llm_traces (
                id                INTEGER PRIMARY KEY AUTOINCREMENT,
                ts                TEXT,
                call_type         TEXT,
                model             TEXT,
                prompt_tokens     INTEGER,
                completion_tokens INTEGER,
                total_tokens      INTEGER,
                cost_usd          REAL,
                latency_ms        REAL,
                success           INTEGER,
                error             TEXT
            )
            """
        )


def _estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    price = MODEL_PRICES.get(model, _DEFAULT_PRICE)
    return (prompt_tokens / 1_000_000) * price["input"] + \
           (completion_tokens / 1_000_000) * price["output"]


# ---------------------------------------------------------------------------
# Writing traces
# ---------------------------------------------------------------------------
def log_call(call_type, model, usage, latency_ms, success=True, error=None) -> None:
    """
    Record a single LLM call. `usage` is the `response.usage` object returned
    by the Groq client (or None on failure).
    """
    prompt_tokens     = getattr(usage, "prompt_tokens", 0) or 0
    completion_tokens = getattr(usage, "completion_tokens", 0) or 0
    total_tokens      = getattr(usage, "total_tokens", prompt_tokens + completion_tokens) or 0
    cost              = _estimate_cost(model, prompt_tokens, completion_tokens)

    try:
        with _LOCK, sqlite3.connect(DB_PATH) as conn:
            conn.execute(
                """
                INSERT INTO llm_traces
                    (ts, call_type, model, prompt_tokens, completion_tokens,
                     total_tokens, cost_usd, latency_ms, success, error)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    datetime.utcnow().isoformat(),
                    call_type,
                    model,
                    prompt_tokens,
                    completion_tokens,
                    total_tokens,
                    cost,
                    round(latency_ms, 2),
                    1 if success else 0,
                    error,
                ),
            )
    except Exception as e:
        # Tracing must never take down a request.
        print(f"[Tracing Error] {e}")


# ---------------------------------------------------------------------------
# Reading aggregates
# ---------------------------------------------------------------------------
def summary() -> dict:
    """Return aggregate cost / latency / reliability stats, overall and by call type."""
    with _LOCK, sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        overall = conn.execute(
            """
            SELECT COUNT(*)                              AS calls,
                   COALESCE(SUM(cost_usd), 0)            AS total_cost_usd,
                   COALESCE(SUM(total_tokens), 0)        AS total_tokens,
                   COALESCE(AVG(latency_ms), 0)          AS avg_latency_ms,
                   COALESCE(SUM(success), 0)             AS successes
            FROM llm_traces
            """
        ).fetchone()

        by_type = conn.execute(
            """
            SELECT call_type,
                   COUNT(*)                       AS calls,
                   COALESCE(SUM(cost_usd), 0)     AS cost_usd,
                   COALESCE(AVG(latency_ms), 0)   AS avg_latency_ms,
                   COALESCE(SUM(success), 0)      AS successes
            FROM llm_traces
            GROUP BY call_type
            ORDER BY calls DESC
            """
        ).fetchall()

    calls = overall["calls"] or 0
    return {
        "calls":            calls,
        "total_cost_usd":   round(overall["total_cost_usd"], 6),
        "total_tokens":     overall["total_tokens"],
        "avg_latency_ms":   round(overall["avg_latency_ms"], 2),
        "success_rate":     round(overall["successes"] / calls, 4) if calls else None,
        "by_call_type": [
            {
                "call_type":      r["call_type"],
                "calls":          r["calls"],
                "cost_usd":       round(r["cost_usd"], 6),
                "avg_latency_ms": round(r["avg_latency_ms"], 2),
                "success_rate":   round(r["successes"] / r["calls"], 4) if r["calls"] else None,
            }
            for r in by_type
        ],
    }
