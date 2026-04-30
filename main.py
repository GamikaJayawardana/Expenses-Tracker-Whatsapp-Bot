import os
import re
import json
import httpx
import asyncio
import secrets
import string
from datetime import datetime
from typing import Optional
import pytz
from fastapi import FastAPI, Request, Query, HTTPException, BackgroundTasks
from fastapi.responses import PlainTextResponse
from dotenv import load_dotenv
from pydantic import BaseModel
from groq import Groq
from motor.motor_asyncio import AsyncIOMotorClient

# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------
load_dotenv()

app = FastAPI(title="Groq Finance Bot - Master Edition")

# --- Configuration ---
VERIFY_TOKEN    = os.getenv("WHATSAPP_VERIFY_TOKEN")
ACCESS_TOKEN    = os.getenv("WHATSAPP_ACCESS_TOKEN")
PHONE_NUMBER_ID = os.getenv("WHATSAPP_PHONE_ID")
MONGODB_URI     = os.getenv("MONGODB_URI")
GROQ_API_KEY    = os.getenv("GROQ_API_KEY")
SL_TIMEZONE     = pytz.timezone('Asia/Colombo')

# Groq model — supports structured outputs (best-effort mode via json_schema)
GROQ_ROUTER_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"  # structured-output capable
GROQ_QUERY_MODEL  = "meta-llama/llama-4-scout-17b-16e-instruct"  # used for free-text answers

# --- Initialize Clients ---
groq_client  = Groq(api_key=GROQ_API_KEY)
mongo_client = AsyncIOMotorClient(MONGODB_URI)
db           = mongo_client.expense_tracker
transactions_collection = db.transactions

# ---------------------------------------------------------------------------
# Data Schemas (Pydantic → used to build the JSON Schema for Groq)
# ---------------------------------------------------------------------------

class ActionItem(BaseModel):
    """A single user intent extracted from the message."""
    intent: str          # LOG | TRANSFER | INITIALIZE | CREDIT_LIMIT | QUERY | DELETE | UPDATE | DELETE_ALL | HELP
    type: Optional[str]  = None   # income | expense | balance | limit
    amount: Optional[float] = None
    category: Optional[str] = None
    date: Optional[str]    = None   # YYYY-MM-DD
    account: Optional[str] = None
    from_account: Optional[str] = None
    to_account: Optional[str]   = None
    transaction_id: Optional[str] = None  # e.g. "A3K9F" (without #)
    note: Optional[str] = None
    query_limit: Optional[int] = None     # For "last N transactions" — set to N

class MultiTransactionData(BaseModel):
    actions: list[ActionItem]

# ---------------------------------------------------------------------------
# System Prompt — carefully engineered for Groq / Llama-4
# ---------------------------------------------------------------------------

ROUTER_SYSTEM_PROMPT = """\
You are FinanceBot, an intelligent financial transaction router for a WhatsApp expense tracker.

## YOUR ONLY JOB
Parse the user's natural-language message into a JSON list of structured actions.
Do NOT answer conversationally. Return ONLY valid JSON that matches the provided schema.

## AVAILABLE INTENTS
| Intent        | When to use                                                         |
|---------------|---------------------------------------------------------------------|
| LOG           | User records an income or expense                                   |
| TRANSFER      | Money moved between two accounts/wallets (e.g. BOC → Wallet)       |
| INITIALIZE    | User sets an opening balance for an account/wallet                  |
| CREDIT_LIMIT  | User sets or updates a credit card spending limit                   |
| QUERY         | User asks a question about their finances (summary, balance, etc.)  |
| UPDATE        | User corrects or edits a previously logged transaction by ID        |
| DELETE        | User deletes a single transaction by ID                             |
| DELETE_ALL    | User wants to wipe ALL their data                                   |
| HELP          | User asks what the bot can do, or sends a greeting                  |

## FIELD RULES
- `intent`          : One of the intents above, UPPER_CASE.
- `type`            : "income" or "expense" for LOG; "balance" for INITIALIZE; "limit" for CREDIT_LIMIT.
- `amount`          : Numeric. Convert shorthand: 1k→1000, 1.5k→1500, 1m→1000000.
- `category`        : Infer from context if not stated (Food, Transport, Bills, Salary, Entertainment, etc.).
- `date`            : YYYY-MM-DD. Resolve relative dates using today's date provided in the user message.
  - "yesterday" → subtract 1 day from today
  - "last Monday" → calculate the most recent Monday
  - If no date mentioned, use today's date.
- `account`         : The account/wallet name. 
  - IMPORTANT: If the user does NOT mention an account/wallet name, ALWAYS default to "Cash" for both income and expense LOG entries. Never leave account null.
  - CRITICAL FOR CREDIT CARDS: When setting a CREDIT_LIMIT or logging to a credit card, the account name MUST ALWAYS end with the words "Credit Card" (e.g., "BOC Credit Card", "Visa Credit Card"). If the user says "Set my BOC credit limit", output account as "BOC Credit Card", NOT just "BOC". This prevents collisions with regular bank accounts.
- `from_account`    : Source account for TRANSFER.
- `to_account`      : Destination account for TRANSFER.
- `transaction_id`  : Extract from patterns like "#ABC12" → "ABC12" (strip the #, uppercase).
- `note`            : Any additional note or description the user provides.
- `query_limit`     : For QUERY only. If the user asks for "last N transactions" (e.g. "last 5", "show 10 transactions", "recent 20"), set this to the integer N. Otherwise leave null.

## HANDLING MULTI-INTENT MESSAGES
A single message can contain multiple actions. Return ALL of them as separate items in the `actions` array.

## EXAMPLE MAPPINGS
User: "Got paid 50k salary into BOC today and spent 500 on lunch from wallet"
→ [ {intent:"LOG", type:"income", amount:50000, category:"Salary", account:"BOC", date:"<today>"},
    {intent:"LOG", type:"expense", amount:500, category:"Food", account:"Wallet", date:"<today>"} ]

User: "Move 10k from BOC to my Wallet"
→ [ {intent:"TRANSFER", amount:10000, from_account:"BOC", to_account:"Wallet"} ]

User: "Paid my 5k Visa credit card bill from BOC"
→ [ {intent:"TRANSFER", amount:5000, from_account:"BOC", to_account:"Visa Credit Card"} ]

User: "Got 2000 refunded to my credit card"
→ [ {intent:"LOG", type:"income", amount:2000, category:"Refund", account:"Credit Card", date:"<today>"} ]

User: "Lent 5k to John from wallet"
→ [ {intent:"LOG", type:"expense", amount:5000, category:"Debt: John", account:"Wallet", date:"<today>"} ]

User: "John paid back 5k to BOC"
→ [ {intent:"LOG", type:"income", amount:5000, category:"Debt: John", account:"BOC", date:"<today>"} ]

User: "Set my Commercial Bank credit card limit to 200k"
→ [ {intent:"CREDIT_LIMIT", amount:200000, account:"Commercial Bank Credit Card", type:"limit"} ]

User: "Update #A3K9F amount to 600"
→ [ {intent:"UPDATE", transaction_id:"A3K9F", amount:600} ]

User: "Delete #XY99Z"
→ [ {intent:"DELETE", transaction_id:"XY99Z"} ]

User: "how much did I spend this month?"
→ [ {intent:"QUERY"} ]

User: "show my last 5 transactions" or "last 10 records"
→ [ {intent:"QUERY", query_limit:5} ]   (use the exact number the user said)

User: "hi" or "help"
→ [ {intent:"HELP"} ]

## UNKNOWN / INVALID INPUT
If the message has no financial meaning (e.g. random text, jokes), return:
[ {intent:"HELP"} ]
"""

# ---------------------------------------------------------------------------
# Helper: WhatsApp message sender
# ---------------------------------------------------------------------------

async def send_whatsapp_message(to_phone_number: str, text: str):
    """Fire a text message back to the user via WhatsApp Cloud API."""
    url = f"https://graph.facebook.com/v18.0/{PHONE_NUMBER_ID}/messages"
    headers = {
        "Authorization": f"Bearer {ACCESS_TOKEN}",
        "Content-Type":  "application/json",
    }
    data = {
        "messaging_product": "whatsapp",
        "to":   to_phone_number,
        "type": "text",
        "text": {"body": text},
    }
    async with httpx.AsyncClient() as client:
        response = await client.post(url, headers=headers, json=data)
        if response.status_code != 200:
            print(f"[WhatsApp Error] {response.status_code}: {response.text}")

# ---------------------------------------------------------------------------
# Helper: short unique ID
# ---------------------------------------------------------------------------

def generate_tx_id() -> str:
    """Generates a 5-character alphanumeric transaction ID."""
    alphabet = string.ascii_uppercase + string.digits
    return ''.join(secrets.choice(alphabet) for _ in range(5))

# ---------------------------------------------------------------------------
# Helper: call Groq router with structured output
# ---------------------------------------------------------------------------

async def call_groq_router(message_text: str, today: str) -> list[dict]:
    """
    Calls Groq with the router system prompt and returns a list of action dicts.
    Uses json_schema response_format (best-effort) with retry logic.
    """
    max_retries = 3
    schema = MultiTransactionData.model_json_schema()

    for attempt in range(max_retries):
        try:
            response = await asyncio.to_thread(
                groq_client.chat.completions.create,
                model=GROQ_ROUTER_MODEL,
                messages=[
                    {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
                    {"role": "user",   "content": f"Today's date is {today}. User message: \"{message_text}\""},
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name":   "multi_transaction_data",
                        "schema": schema,
                    },
                },
                temperature=0.1,   # Low temp → more deterministic parsing
                max_tokens=1024,
            )
            raw = response.choices[0].message.content or "{}"
            parsed = json.loads(raw)
            return parsed.get("actions", [])

        except Exception as e:
            err_str = str(e)
            if ("429" in err_str or "503" in err_str or "502" in err_str) and attempt < max_retries - 1:
                wait = 2 ** attempt  # exponential back-off: 1s, 2s
                print(f"[Groq] Rate-limited/busy, retrying in {wait}s (attempt {attempt + 1})")
                await asyncio.sleep(wait)
            else:
                print(f"[Groq Router Error] {e}")
                raise

    return []

# ---------------------------------------------------------------------------
# Helper: call Groq for free-text query answers
# ---------------------------------------------------------------------------

async def call_groq_query(history_str: str, question: str, today: str) -> str:
    """
    Answers a user's finance question in natural language using their history.
    """
    system = (
        "You are a friendly personal finance assistant for a Sri Lankan user. "
        "The user's transaction history is provided below (pipe-separated). "
        "Answer their question concisely using emojis for clarity. "
        "Calculate totals yourself — show ONLY the final result, NOT the calculation steps. "
        "If the history is empty, say so politely. "
        "CRITICAL FORMATTING RULES for WhatsApp:\n"
        "  1. Use ONLY single asterisks for bold: *like this* — NEVER use **double asterisks**.\n"
        "  2. Always use 'Rs.' for currency (e.g. Rs. 1,500) — NEVER use ₹, INR, LKR, or $.\n"
        "  3. Keep answers SHORT (max 8 lines). No verbose breakdowns or calculation formulas.\n"
        "  4. Never show math expressions like 'Rs. 200 - Rs. 1,000 = Rs. -800'."
    )
    user_content = (
        f"Today: {today}\n\n"
        f"Transaction history (ID | Date | Type | Amount | Category | Account):\n"
        f"{history_str}\n\n"
        f"User question: {question}"
    )

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = await asyncio.to_thread(
                groq_client.chat.completions.create,
                model=GROQ_QUERY_MODEL,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user_content},
                ],
                temperature=0.3,
                max_tokens=512,
            )
            return response.choices[0].message.content or "I couldn't generate an answer. Please try again."
        except Exception:
            if attempt < max_retries - 1:
                await asyncio.sleep(1)
    return "⚠️ Couldn't fetch an answer right now. Please try again in a moment."

# ---------------------------------------------------------------------------
# Core: compute net balances and credit limits per account
# ---------------------------------------------------------------------------

# Keywords that trigger a Python-computed summary (bypass Groq)
SUMMARY_RE = re.compile(
    r'\b(summ(?:a(?:r(?:y|ies)?)?|e?ry)?|balance|overview|total|how much.*have|account.*status)\b',
    re.IGNORECASE
)

async def get_account_balances(sender_phone: str) -> dict[str, float]:
    """
    Returns {account: net_balance} for regular accounts only.
    Credit-card accounts (those with an init_limit record) are excluded here;
    their spending is tracked via get_credit_info() instead.
    """
    # Find all accounts that have a credit limit (treat them separately)
    limit_cursor = transactions_collection.find(
        {"user_phone": sender_phone, "type": "init_limit"}
    )
    limit_docs   = await limit_cursor.to_list(length=None)
    cc_accounts  = {d["account"] for d in limit_docs}

    cursor = transactions_collection.find({"user_phone": sender_phone}).sort("created_at", 1)
    docs   = await cursor.to_list(length=None)

    balances: dict[str, float] = {}

    for t in docs:
        tt  = t.get("type", "")
        amt = float(t.get("amount", 0))
        acc = t.get("account", "")

        if tt == "init_balance":
            if acc not in cc_accounts:   # only for regular accounts
                balances[acc] = amt
        elif tt == "income":
            if acc not in cc_accounts:
                balances[acc] = balances.get(acc, 0) + amt
        elif tt == "expense":
            if acc not in cc_accounts:   # CC expenses tracked via get_credit_info()
                balances[acc] = balances.get(acc, 0) - amt
        elif tt == "transfer":
            src = t.get("from_account", "")
            dst = t.get("to_account", "")
            if src and src not in cc_accounts:
                balances[src] = balances.get(src, 0) - amt
            if dst and dst not in cc_accounts:
                balances[dst] = balances.get(dst, 0) + amt
        # init_limit handled separately

    return balances


async def get_credit_info(sender_phone: str) -> list[dict]:
    """
    Returns a list of dicts for each credit-card account:
    {account, limit, spent, remaining, used_pct}
    """
    limit_cursor = transactions_collection.find(
        {"user_phone": sender_phone, "type": "init_limit"}
    )
    limit_docs = await limit_cursor.to_list(length=None)

    result = []
    for ld in limit_docs:
        acc   = ld["account"]
        limit = float(ld["amount"])

        # Fetch all transactions involving this credit card account
        cursor = transactions_collection.find({
            "user_phone": sender_phone,
            "$or": [
                {"account": acc},
                {"to_account": acc},
                {"from_account": acc}
            ]
        })
        tx_docs = await cursor.to_list(length=None)

        total_spent = 0.0
        for d in tx_docs:
            tt = d.get("type", "")
            amt = float(d.get("amount", 0))
            if tt == "expense" and d.get("account") == acc:
                total_spent += amt
            elif tt == "income" and d.get("account") == acc:
                total_spent -= amt
            elif tt == "transfer" and d.get("from_account") == acc:
                total_spent += amt  # Cash advance
            elif tt == "transfer" and d.get("to_account") == acc:
                total_spent -= amt  # Bill payment

        remaining   = limit - total_spent
        used_pct    = (total_spent / limit * 100) if limit > 0 else 0

        result.append({
            "account":   acc,
            "limit":     limit,
            "spent":     total_spent,
            "remaining": remaining,
            "used_pct":  used_pct,
        })

    return result

# ---------------------------------------------------------------------------
# Core: Background worker — processes each WhatsApp message
# ---------------------------------------------------------------------------

async def process_user_message(sender_phone: str, message_text: str):
    print(f"\n[Worker] From={sender_phone} | Msg='{message_text}'")
    today = datetime.now(SL_TIMEZONE).strftime('%Y-%m-%d')

    try:
        # ── 1. BRAIN: Parse the message into structured actions ─────────────
        try:
            actions = await call_groq_router(message_text, today)
        except Exception:
            await send_whatsapp_message(
                sender_phone,
                "⏳ My AI brain is temporarily overloaded. Please try again in a few seconds!"
            )
            return

        if not actions:
            await send_whatsapp_message(sender_phone, "🤔 I didn't quite understand that. Try *help* to see what I can do.")
            return

        # ── 2. EXECUTION: Handle each action ────────────────────────────────
        for action in actions:
            intent = action.get("intent", "").upper()
            raw_id = action.get("transaction_id") or ""
            tx_id  = raw_id.upper().replace("#", "").strip() if raw_id else None

            # ── LOG ──────────────────────────────────────────────────────────
            if intent == "LOG" and action.get("amount"):
                new_id  = generate_tx_id()
                tx_type = (action.get("type") or "expense").lower()
                acc     = action.get("account") or "Cash"
                cat     = action.get("category") or "General"
                doc = {
                    "user_phone": sender_phone,
                    "tx_id":      new_id,
                    "type":       tx_type,
                    "amount":     action["amount"],
                    "category":   cat,
                    "account":    acc,
                    "date":       action.get("date") or today,
                    "note":       action.get("note") or "",
                    "created_at": datetime.now(SL_TIMEZONE),
                }
                await transactions_collection.insert_one(doc)

                # Fetch updated balance to show in the confirmation
                balances = await get_account_balances(sender_phone)
                rem_str = ""
                
                if acc in balances:
                    rem_str = f"✅ Remaining: Rs. {balances[acc]:,.0f}\n"
                else:
                    # Might be a credit card
                    c_info = await get_credit_info(sender_phone)
                    cc_data = next((c for c in c_info if c["account"] == acc), None)
                    if cc_data:
                        rem_str = f"✅ Remaining Limit: Rs. {cc_data['remaining']:,.0f}\n"

                icon = "🔴" if tx_type == "expense" else "🟢"
                await send_whatsapp_message(
                    sender_phone,
                    f"{icon} *{tx_type.capitalize()} Logged*\n"
                    f"💰 Rs. {action['amount']:,.0f}\n"
                    f"📂 {cat}  |  🏦 {acc}\n"
                    f"{rem_str}"
                    f"📅 {doc['date']}\n"
                    f"🆔 `#{new_id}`"
                )

            # ── TRANSFER ─────────────────────────────────────────────────────
            elif intent == "TRANSFER" and action.get("amount"):
                from_acc = action.get("from_account") or "Unknown"
                to_acc   = action.get("to_account")   or "Unknown"

                # Validate: source account must exist
                balances = await get_account_balances(sender_phone)
                src_bal  = balances.get(from_acc)
                if src_bal is None:
                    await send_whatsapp_message(
                        sender_phone,
                        f"❌ Account *{from_acc}* not found. Initialize it first with:\n"
                        f"`I have <amount> in {from_acc}`"
                    )
                    continue

                transfer_amt = float(action["amount"])
                if src_bal < transfer_amt:
                    await send_whatsapp_message(
                        sender_phone,
                        f"⚠️ *Insufficient balance* in *{from_acc}*.\n"
                        f"Available: Rs. {src_bal:,.0f} | Requested: Rs. {transfer_amt:,.0f}\n"
                        f"Do you want to proceed anyway? Reply `confirm transfer #{{}}`"
                    )
                    # Still log it — the user can decide
                    # Fall through to log so they have a record

                new_id = generate_tx_id()
                doc = {
                    "user_phone":   sender_phone,
                    "tx_id":        new_id,
                    "type":         "transfer",
                    "amount":       transfer_amt,
                    "from_account": from_acc,
                    "to_account":   to_acc,
                    "date":         action.get("date") or today,
                    "note":         action.get("note") or "",
                    "created_at":   datetime.now(SL_TIMEZONE),
                }
                await transactions_collection.insert_one(doc)

                # Check if this is a credit card payment
                limit_check = await transactions_collection.find_one({"user_phone": sender_phone, "type": "init_limit", "account": to_acc})
                is_cc_payment = limit_check is not None

                # New balances after transfer
                new_balances = await get_account_balances(sender_phone)
                new_src_bal  = new_balances.get(from_acc, 0)
                
                if is_cc_payment:
                    c_info = await get_credit_info(sender_phone)
                    cc_data = next((c for c in c_info if c["account"] == to_acc), None)
                    if cc_data:
                        await send_whatsapp_message(
                            sender_phone,
                            f"💳 *Credit Card Bill Paid*\n"
                            f"💸 Rs. {transfer_amt:,.0f}\n"
                            f"📤 *{from_acc}* → Rs. {new_src_bal:,.0f}\n"
                            f"✅ *{to_acc}* Remaining Limit: Rs. {cc_data['remaining']:,.0f}\n"
                            f"🆔 `#{new_id}`"
                        )
                else:
                    new_dst_bal  = new_balances.get(to_acc,   0)
                    await send_whatsapp_message(
                        sender_phone,
                        f"🔄 *Transfer Done*\n"
                        f"💸 Rs. {transfer_amt:,.0f}\n"
                        f"📤 *{from_acc}* → Rs. {new_src_bal:,.0f}\n"
                        f"📥 *{to_acc}* → Rs. {new_dst_bal:,.0f}\n"
                        f"🆔 `#{new_id}`"
                    )

            # ── INITIALIZE (set opening balance) ─────────────────────────────
            elif intent == "INITIALIZE" and action.get("amount"):
                new_id     = generate_tx_id()
                acc        = action.get("account") or "Wallet"
                type_label = (action.get("type") or "balance").lower()
                doc = {
                    "user_phone": sender_phone,
                    "tx_id":      new_id,
                    "type":       f"init_{type_label}",
                    "amount":     action["amount"],
                    "account":    acc,
                    "date":       today,
                    "note":       action.get("note") or "",
                    "created_at": datetime.now(SL_TIMEZONE),
                }
                await transactions_collection.insert_one(doc)
                await send_whatsapp_message(
                    sender_phone,
                    f"⚙️ *{acc}* initialized!\n"
                    f"💰 Opening balance: Rs. {action['amount']:,.0f}\n"
                    f"🆔 `#{new_id}`"
                )

            # ── CREDIT_LIMIT ──────────────────────────────────────────────────
            elif intent == "CREDIT_LIMIT" and action.get("amount"):
                acc   = action.get("account") or "Credit Card"
                limit = float(action["amount"])

                # Upsert a "credit_limit" document for this account
                await transactions_collection.update_one(
                    {"user_phone": sender_phone, "type": "init_limit", "account": acc},
                    {"$set": {
                        "user_phone": sender_phone,
                        "type":       "init_limit",
                        "amount":     limit,
                        "account":    acc,
                        "date":       today,
                        "note":       action.get("note") or "",
                        "created_at": datetime.now(SL_TIMEZONE),
                    }},
                    upsert=True,
                )

                # How much spent on this card already?
                spent_cursor = transactions_collection.find({
                    "user_phone": sender_phone,
                    "type":       "expense",
                    "account":    acc,
                })
                spent_docs = await spent_cursor.to_list(length=None)
                total_spent = sum(float(d.get("amount", 0)) for d in spent_docs)
                remaining   = limit - total_spent
                used_pct    = (total_spent / limit * 100) if limit > 0 else 0

                status_emoji = "🟢" if used_pct < 50 else ("🟡" if used_pct < 80 else "🔴")
                await send_whatsapp_message(
                    sender_phone,
                    f"💳 *Credit Limit Set*\n"
                    f"🏦 Card: *{acc}*\n"
                    f"📊 Limit: Rs. {limit:,.0f}\n"
                    f"💸 Spent: Rs. {total_spent:,.0f} ({used_pct:.1f}%)\n"
                    f"✅ Remaining: Rs. {remaining:,.0f}\n"
                    f"{status_emoji} Status: {'OK' if used_pct < 80 else 'Near Limit!'}"
                )

            # ── UPDATE ────────────────────────────────────────────────────────
            elif intent == "UPDATE" and tx_id:
                update_fields = {
                    k: v for k, v in action.items()
                    if v is not None and k not in ("intent", "transaction_id")
                }
                res = await transactions_collection.update_one(
                    {"user_phone": sender_phone, "tx_id": tx_id},
                    {"$set": update_fields}
                )
                if res.modified_count:
                    await send_whatsapp_message(sender_phone, f"📝 *Updated* `#{tx_id}` successfully.")
                else:
                    await send_whatsapp_message(sender_phone, f"❌ No record found for `#{tx_id}`.")

            # ── DELETE ────────────────────────────────────────────────────────
            elif intent == "DELETE" and tx_id:
                res = await transactions_collection.delete_one(
                    {"user_phone": sender_phone, "tx_id": tx_id}
                )
                if res.deleted_count:
                    await send_whatsapp_message(sender_phone, f"🗑️ *Deleted* `#{tx_id}`.")
                else:
                    await send_whatsapp_message(sender_phone, f"❌ No record found for `#{tx_id}`.")

            # ── DELETE_ALL ────────────────────────────────────────────────────
            elif intent == "DELETE_ALL":
                await transactions_collection.delete_many({"user_phone": sender_phone})
                await send_whatsapp_message(sender_phone, "🧹 *All your data has been wiped.* Fresh start! 🚀")

            # ── QUERY ─────────────────────────────────────────────────────────
            elif intent == "QUERY":
                q_limit     = action.get("query_limit")  # None or int
                fetch_limit = min(int(q_limit), 50) if q_limit else 100

                cursor = transactions_collection.find(
                    {"user_phone": sender_phone}
                ).sort("created_at", -1).limit(fetch_limit)
                data = await cursor.to_list(length=fetch_limit)

                # ── Fast-path: "last N transactions" — return a formatted table ──
                if q_limit and data:
                    header = f"📋 *Last {min(int(q_limit), len(data))} Transactions*\n━━━━━━━━━━━━━━━━━━━━━━"
                    lines  = [header]
                    type_icons = {
                        "income":   "🟢",
                        "expense":  "🔴",
                        "transfer": "🔄",
                    }
                    for i, t in enumerate(data, 1):
                        tid = t.get("tx_id", "INIT")
                        tt  = t.get("type", "")
                        amt = float(t.get("amount", 0))
                        dt  = t.get("date", "")

                        if tt == "transfer":
                            lines.append(
                                f"{i}. 🔄 *Transfer* Rs. {amt:,.0f}\n"
                                f"   📤 {t.get('from_account')} → 📥 {t.get('to_account')}\n"
                                f"   📅 {dt}  🆔 `#{tid}`"
                            )
                        elif tt.startswith("init_"):
                            label = tt.split("_")[1].capitalize()
                            lines.append(
                                f"{i}. ⚙️ *Init {label}* {t.get('account')} = Rs. {amt:,.0f}\n"
                                f"   📅 {dt}  🆔 `#{tid}`"
                            )
                        else:
                            icon = type_icons.get(tt, "⚪")
                            cat  = t.get("category", "—")
                            acc  = t.get("account", "—")
                            lines.append(
                                f"{i}. {icon} *{tt.capitalize()}* Rs. {amt:,.0f}\n"
                                f"   📂 {cat}  🏦 {acc}\n"
                                f"   📅 {dt}  🆔 `#{tid}`"
                            )

                    await send_whatsapp_message(sender_phone, "\n\n".join(lines))

                elif SUMMARY_RE.search(message_text):
                    # ── Summary fast-path: compute everything in Python ──────
                    balances    = await get_account_balances(sender_phone)
                    credit_info = await get_credit_info(sender_phone)

                    # Compute monthly stats
                    current_month = datetime.now(SL_TIMEZONE).strftime('%Y-%m')
                    monthly_cursor = transactions_collection.find({
                        "user_phone": sender_phone,
                        "type": "expense",
                        "date": {"$regex": f"^{current_month}"}
                    })
                    monthly_docs = await monthly_cursor.to_list(length=None)
                    
                    monthly_total = 0.0
                    cat_map = {}
                    for d in monthly_docs:
                        amt = float(d.get("amount", 0))
                        cat = d.get("category", "General")
                        monthly_total += amt
                        cat_map[cat] = cat_map.get(cat, 0) + amt
                        
                    top_cats = sorted(cat_map.items(), key=lambda x: x[1], reverse=True)[:3]

                    lines = ["📊 *Account Summary*", "━━━━━━━━━━━━━━━━━━━━━━"]
                    
                    if monthly_total > 0:
                        lines.append(f"📅 *This Month's Spending:* Rs. {monthly_total:,.0f}")
                        if top_cats:
                            lines.append("🔥 *Top Categories:*")
                            for cat, amt in top_cats:
                                lines.append(f"   • {cat}: Rs. {amt:,.0f}")
                        lines.append("━━━━━━━━━━━━━━━━━━━━━━")

                    total_regular = 0.0

                    # Regular accounts
                    if balances:
                        lines.append("\n🏦 *Accounts & Wallets*")
                        for acc, bal in sorted(balances.items()):
                            total_regular += bal
                            if bal < 0:
                                lines.append(f"  ⚠️ {acc}: Rs. {bal:,.0f}  _(overdrawn)_")
                            else:
                                lines.append(f"  ✅ {acc}: Rs. {bal:,.0f}")
                        lines.append(f"\n💰 *Total Balance: Rs. {total_regular:,.0f}*")
                    else:
                        lines.append("\nNo accounts set up yet. Try: _I have 5000 in BOC_")

                    # Credit cards
                    if credit_info:
                        lines.append("\n💳 *Credit Cards*")
                        for cc in credit_info:
                            pct   = cc['used_pct']
                            bar   = "🟢" if pct < 50 else ("🟡" if pct < 80 else "🔴")
                            lines.append(
                                f"  {bar} {cc['account']}\n"
                                f"     Spent Rs. {cc['spent']:,.0f} / Limit Rs. {cc['limit']:,.0f}\n"
                                f"     Remaining: Rs. {cc['remaining']:,.0f}  ({pct:.0f}% used)"
                            )

                    await send_whatsapp_message(sender_phone, "\n".join(lines))

                else:
                    # ── Standard path: pass history to Groq for analysis ─────
                    rows = []
                    for t in data:
                        tid = t.get("tx_id", "INIT")
                        tt  = t.get("type", "")
                        if tt == "transfer":
                            rows.append(
                                f"#{tid} | {t['date']} | TRANSFER | Rs.{t.get('amount',0):,.0f} "
                                f"| {t.get('from_account')}→{t.get('to_account')}"
                            )
                        elif tt.startswith("init_"):
                            label = tt.split("_")[1].upper()
                            rows.append(
                                f"#{tid} | {t['date']} | INIT-{label} | {t.get('account')} "
                                f"= Rs.{t.get('amount',0):,.0f}"
                            )
                        else:
                            rows.append(
                                f"#{tid} | {t['date']} | {tt} | Rs.{t.get('amount',0):,.0f} "
                                f"| {t.get('category','—')} | {t.get('account','—')}"
                            )

                    history_str = "\n".join(rows) if rows else "No records found."
                    answer = await call_groq_query(history_str, message_text, today)
                    await send_whatsapp_message(sender_phone, answer)

            # ── HELP ──────────────────────────────────────────────────────────
            elif intent == "HELP":
                help_msg = (
                    "👋 *Hi! I'm your AI Finance Bot* 🤖\n"
                    "Powered by Groq ⚡ | Ultra-fast AI\n"
                    "━━━━━━━━━━━━━━━━━━━━━━\n\n"

                    "🏦 *SETUP — Initialize Accounts*\n"
                    "  • `I have 50k in BOC`\n"
                    "  • `My wallet has Rs. 2000`\n"
                    "  • `Opening balance: 100k in Commercial Bank`\n\n"

                    "💸 *LOG — Record Transactions*\n"
                    "  • `Spent 500 on lunch from Wallet`\n"
                    "  • `Got paid 75k salary into BOC today`\n"
                    "  • `Bought groceries for 3500 yesterday from cash`\n\n"

                    "🔄 *TRANSFER — Between Accounts*\n"
                    "  • `Move 10k from BOC to Wallet`\n"
                    "  • `Transfer 5000 from Wallet to Savings`\n"
                    "  • `Send 20k from Commercial Bank to cash`\n\n"

                    "💳 *CREDIT LIMIT — Set Card Limit*\n"
                    "  • `Set my BOC credit card limit to 200k`\n"
                    "  • `Commercial Bank Visa limit is 150000`\n"
                    "  • `Update my Sampath credit limit to 100k`\n\n"

                    "📊 *QUERY — Ask Anything*\n"
                    "  • `How much did I spend this week?`\n"
                    "  • `What's my BOC balance?`\n"
                    "  • `Show my top spending categories`\n"
                    "  • `How much is left on my credit card?`\n\n"

                    "✏️ *EDIT — Fix Mistakes*\n"
                    "  • `Update #ABC12 amount to 600`\n"
                    "  • `Change #XY9KL account to BOC`\n"
                    "  • `Delete #A1B2C`\n\n"

                    "🗑️ *CLEAR — Wipe All Data*\n"
                    "  • `Delete all my records`\n"
                    "  • `Clear everything`\n\n"

                    "━━━━━━━━━━━━━━━━━━━━━━\n"
                    "💡 *Pro Tips:*\n"
                    "  • I understand shorthand: 1k, 1.5k, 1m\n"
                    "  • I handle multiple actions at once!\n"
                    "    _\"Got 75k salary and spent 2k on bills\"_\n"
                    "  • Every transaction gets a unique `#ID`\n"
                    "  • Dates: today, yesterday, last Monday\n"
                )
                await send_whatsapp_message(sender_phone, help_msg)

            else:
                # Unrecognized intent — nudge user
                await send_whatsapp_message(
                    sender_phone,
                    f"🤔 I'm not sure what `{intent}` means. Try *help* to see what I can do."
                )

    except Exception as e:
        print(f"[Fatal Worker Error] {e}")
        import traceback
        traceback.print_exc()
        await send_whatsapp_message(sender_phone, "⚠️ Something went wrong on my end. Please try again!")

# ---------------------------------------------------------------------------
# FastAPI Webhook Routes
# ---------------------------------------------------------------------------

@app.get("/webhook")
async def verify(
    mode:      str = Query(None, alias="hub.mode"),
    token:     str = Query(None, alias="hub.verify_token"),
    challenge: int = Query(None, alias="hub.challenge"),
):
    if mode == "subscribe" and token == VERIFY_TOKEN:
        return PlainTextResponse(str(challenge))
    raise HTTPException(status_code=403, detail="Verification failed")

@app.post("/webhook")
async def handle_hook(request: Request, bg: BackgroundTasks):
    try:
        raw = await request.json()
        val = raw["entry"][0]["changes"][0]["value"]
        if "messages" in val:
            msg_obj = val["messages"][0]
            if msg_obj.get("type") == "text":
                bg.add_task(
                    process_user_message,
                    msg_obj["from"],
                    msg_obj["text"]["body"],
                )
    except Exception as e:
        print(f"[Webhook Parse Error] {e}")
    return {"status": "success"}

@app.get("/health")
async def health():
    return {"status": "ok", "engine": "Groq", "model": GROQ_ROUTER_MODEL}