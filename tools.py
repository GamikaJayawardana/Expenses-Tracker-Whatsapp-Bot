"""
Tool-calling query agent.

Free-text finance questions ("show my spending this month", "what's left on my
credit card") are answered by letting the model call typed tools that read from
the database, then summarising the tool results back to the user. This keeps the
numbers computed in Python and lets the model handle the language.
"""

import json

from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage

import llm
import store

# ---------------------------------------------------------------------------
# Tool schemas (OpenAI / Groq function-calling format)
# ---------------------------------------------------------------------------
QUERY_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_spending",
            "description": "Total expense spending and a per-category breakdown for a period.",
            "parameters": {
                "type": "object",
                "properties": {
                    "period": {
                        "type": "string",
                        "enum": ["this_week", "this_month", "last_month", "all"],
                        "description": "Time window to summarise. Defaults to this_month.",
                    }
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_account_balances",
            "description": "Current net balance of every regular account/wallet.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_credit_info",
            "description": "Limit, spend, remaining and utilisation for each credit card.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_recent_transactions",
            "description": "The user's most recent transactions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "How many transactions to return (1-50). Defaults to 5.",
                    }
                },
            },
        },
    },
]

QUERY_SYSTEM_PROMPT = (
    "You are a friendly personal finance assistant for a Sri Lankan user. "
    "Use the provided tools to look up the numbers you need before answering — "
    "never invent figures. Answer concisely using emojis for clarity.\n"
    "CRITICAL FORMATTING RULES for WhatsApp:\n"
    "  1. Use ONLY single asterisks for bold: *like this* — NEVER use **double asterisks**.\n"
    "  2. Always use 'Rs.' for currency (e.g. Rs. 1,500) — NEVER use the rupee sign, INR, LKR, or $.\n"
    "  3. Keep answers SHORT (max 8 lines). No verbose breakdowns or calculation formulas.\n"
    "  4. Never show math expressions like 'Rs. 200 - Rs. 1,000 = Rs. -800'.\n"
    "  5. If a tool returns no data, say so politely."
)


# ---------------------------------------------------------------------------
# Tool execution
# ---------------------------------------------------------------------------
async def _execute_tool(name: str, args: dict, sender_phone: str) -> dict:
    if name == "get_spending":
        return await store.get_spending(sender_phone, args.get("period", "this_month"))
    if name == "get_account_balances":
        return {"balances": await store.get_account_balances(sender_phone)}
    if name == "get_credit_info":
        return {"credit_cards": await store.get_credit_info(sender_phone)}
    if name == "get_recent_transactions":
        return {"transactions": await store.get_recent_transactions(sender_phone, args.get("limit", 5))}
    return {"error": f"Unknown tool: {name}"}


async def run_query_agent(sender_phone: str, question: str, today: str) -> str:
    """
    Answer a finance question using LangChain tool calls against the user's data.

    One tool-calling round-trip: the model (with tools bound) picks tools, we
    run them, then a plain model call writes the final reply from the results.
    """
    chat_with_tools = llm.get_chat(llm.QUERY_MODEL, temperature=0.2, max_tokens=700).bind_tools(QUERY_TOOLS)

    messages = [
        SystemMessage(content=QUERY_SYSTEM_PROMPT),
        HumanMessage(content=f"Today: {today}\n\nUser question: {question}"),
    ]

    try:
        ai = await llm.traced_invoke("query", chat_with_tools, messages, llm.QUERY_MODEL)
    except Exception:
        return "⚠️ Couldn't fetch an answer right now. Please try again in a moment."

    tool_calls = ai.tool_calls or []

    # No tool needed — the model answered directly.
    if not tool_calls:
        return ai.content or "I couldn't generate an answer. Please try again."

    # Record the assistant turn (with its tool calls), then run each tool.
    messages.append(ai)
    for tc in tool_calls:
        try:
            result = await _execute_tool(tc["name"], tc.get("args") or {}, sender_phone)
        except Exception as e:
            # A failing tool becomes an error payload so the model can still reply.
            print(f"[Tool Error] {tc['name']}: {e}")
            result = {"error": "data lookup failed"}
        messages.append(ToolMessage(
            content=json.dumps(result, default=str),
            tool_call_id=tc["id"],
        ))

    try:
        final = await llm.traced_invoke(
            "query_final",
            llm.get_chat(llm.QUERY_MODEL, temperature=0.3, max_tokens=512),
            messages,
            llm.QUERY_MODEL,
        )
        return final.content or "I couldn't generate an answer. Please try again."
    except Exception:
        return "⚠️ Couldn't fetch an answer right now. Please try again in a moment."
