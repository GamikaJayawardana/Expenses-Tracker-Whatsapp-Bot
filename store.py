"""
MongoDB access layer.

Holds the transactions collection and every read helper the bot needs:
running balances, credit-card status, monthly/period spending and recent
history. Both the message worker and the query tools read through here.
"""

import os
import secrets
import string
from datetime import datetime, timedelta
from typing import Optional
import pytz
from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient

load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI")
SL_TIMEZONE = pytz.timezone("Asia/Colombo")

mongo_client = AsyncIOMotorClient(MONGODB_URI)
db           = mongo_client.expense_tracker
transactions_collection = db.transactions


def generate_tx_id() -> str:
    """Generate a 5-character alphanumeric transaction ID."""
    alphabet = string.ascii_uppercase + string.digits
    return "".join(secrets.choice(alphabet) for _ in range(5))


# ---------------------------------------------------------------------------
# Balances and credit cards
# ---------------------------------------------------------------------------
async def get_account_balances(sender_phone: str) -> dict[str, float]:
    """
    Return {account: net_balance} for regular accounts only. Credit-card
    accounts (those with an init_limit record) are excluded — their spending
    is tracked via get_credit_info() instead.
    """
    limit_cursor = transactions_collection.find(
        {"user_phone": sender_phone, "type": "init_limit"}
    )
    limit_docs  = await limit_cursor.to_list(length=None)
    cc_accounts = {d["account"] for d in limit_docs}

    cursor = transactions_collection.find({"user_phone": sender_phone}).sort("created_at", 1)
    docs   = await cursor.to_list(length=None)

    balances: dict[str, float] = {}

    for t in docs:
        tt  = t.get("type", "")
        amt = float(t.get("amount", 0))
        acc = t.get("account", "")

        if tt == "init_balance":
            if acc not in cc_accounts:
                balances[acc] = amt
        elif tt == "income":
            if acc not in cc_accounts:
                balances[acc] = balances.get(acc, 0) + amt
        elif tt == "expense":
            if acc not in cc_accounts:
                balances[acc] = balances.get(acc, 0) - amt
        elif tt == "transfer":
            src = t.get("from_account", "")
            dst = t.get("to_account", "")
            if src and src not in cc_accounts:
                balances[src] = balances.get(src, 0) - amt
            if dst and dst not in cc_accounts:
                balances[dst] = balances.get(dst, 0) + amt

    return balances


async def get_credit_info(sender_phone: str) -> list[dict]:
    """
    Return a list of dicts for each credit-card account:
    {account, limit, spent, remaining, used_pct}.
    """
    limit_cursor = transactions_collection.find(
        {"user_phone": sender_phone, "type": "init_limit"}
    )
    limit_docs = await limit_cursor.to_list(length=None)

    result = []
    for ld in limit_docs:
        acc   = ld["account"]
        limit = float(ld["amount"])

        cursor = transactions_collection.find({
            "user_phone": sender_phone,
            "$or": [
                {"account": acc},
                {"to_account": acc},
                {"from_account": acc},
            ],
        })
        tx_docs = await cursor.to_list(length=None)

        total_spent = 0.0
        for d in tx_docs:
            tt  = d.get("type", "")
            amt = float(d.get("amount", 0))
            if tt == "expense" and d.get("account") == acc:
                total_spent += amt
            elif tt == "income" and d.get("account") == acc:
                total_spent -= amt
            elif tt == "transfer" and d.get("from_account") == acc:
                total_spent += amt  # cash advance
            elif tt == "transfer" and d.get("to_account") == acc:
                total_spent -= amt  # bill payment

        remaining = limit - total_spent
        used_pct  = (total_spent / limit * 100) if limit > 0 else 0

        result.append({
            "account":   acc,
            "limit":     limit,
            "spent":     total_spent,
            "remaining": remaining,
            "used_pct":  used_pct,
        })

    return result


# ---------------------------------------------------------------------------
# Spending queries (used by the tool-calling query agent)
# ---------------------------------------------------------------------------
def _period_start(period: str) -> Optional[str]:
    """Return an inclusive YYYY-MM-DD lower bound for the given period, or None for 'all'."""
    now = datetime.now(SL_TIMEZONE)
    if period == "this_month":
        return now.strftime("%Y-%m-01")
    if period == "last_month":
        first_this = now.replace(day=1)
        last_month_end = first_this - timedelta(days=1)
        return last_month_end.strftime("%Y-%m-01")
    if period == "this_week":
        monday = now - timedelta(days=now.weekday())
        return monday.strftime("%Y-%m-%d")
    return None


def _period_end(period: str) -> Optional[str]:
    """Return an inclusive upper bound only where a period has a hard end (last_month)."""
    if period == "last_month":
        now = datetime.now(SL_TIMEZONE)
        first_this = now.replace(day=1)
        last_month_end = first_this - timedelta(days=1)
        return last_month_end.strftime("%Y-%m-%d")
    return None


async def get_spending(sender_phone: str, period: str = "this_month") -> dict:
    """
    Summarise expense spending for a period: this_week | this_month |
    last_month | all. Returns total, transaction count and a per-category
    breakdown.
    """
    query: dict = {"user_phone": sender_phone, "type": "expense"}
    start = _period_start(period)
    end   = _period_end(period)
    if start and end:
        query["date"] = {"$gte": start, "$lte": end}
    elif start:
        query["date"] = {"$gte": start}

    cursor = transactions_collection.find(query)
    docs   = await cursor.to_list(length=None)

    total = 0.0
    by_category: dict[str, float] = {}
    for d in docs:
        amt = float(d.get("amount", 0))
        cat = d.get("category", "General")
        total += amt
        by_category[cat] = by_category.get(cat, 0) + amt

    return {
        "period":      period,
        "total":       total,
        "count":       len(docs),
        "by_category": dict(sorted(by_category.items(), key=lambda x: x[1], reverse=True)),
    }


async def get_recent_transactions(sender_phone: str, limit: int = 5) -> list[dict]:
    """Return the most recent transactions as plain dicts."""
    limit  = max(1, min(int(limit), 50))
    cursor = transactions_collection.find(
        {"user_phone": sender_phone}
    ).sort("created_at", -1).limit(limit)
    docs = await cursor.to_list(length=limit)

    out = []
    for t in docs:
        out.append({
            "id":           t.get("tx_id", "INIT"),
            "date":         t.get("date", ""),
            "type":         t.get("type", ""),
            "amount":       float(t.get("amount", 0)),
            "category":     t.get("category"),
            "account":      t.get("account"),
            "from_account": t.get("from_account"),
            "to_account":   t.get("to_account"),
        })
    return out
