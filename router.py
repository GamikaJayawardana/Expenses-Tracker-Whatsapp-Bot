"""
Intent router.

Turns a free-form WhatsApp message into a validated list of structured
actions. The model is asked for JSON matching the action schema; the response
is then validated against Pydantic. If validation (or JSON parsing) fails, the
error is fed back to the model and the call is retried before we fall back to a
lenient salvage pass.
"""

import json
import asyncio
from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, field_validator, ValidationError

import llm

MAX_RETRIES = 3
_TRANSIENT_MARKERS = ("429", "502", "503")


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class Intent(str, Enum):
    LOG          = "LOG"
    TRANSFER     = "TRANSFER"
    INITIALIZE   = "INITIALIZE"
    CREDIT_LIMIT = "CREDIT_LIMIT"
    QUERY        = "QUERY"
    UPDATE       = "UPDATE"
    DELETE       = "DELETE"
    DELETE_ALL   = "DELETE_ALL"
    HELP         = "HELP"


class ActionItem(BaseModel):
    """A single user intent extracted from the message."""
    model_config = ConfigDict(extra="ignore", use_enum_values=True)

    intent: Intent
    type: Optional[str] = None            # income | expense | balance | limit
    amount: Optional[float] = None
    category: Optional[str] = None
    date: Optional[str] = None            # YYYY-MM-DD
    account: Optional[str] = None
    from_account: Optional[str] = None
    to_account: Optional[str] = None
    transaction_id: Optional[str] = None  # e.g. "A3K9F" (without #)
    note: Optional[str] = None
    query_limit: Optional[int] = None     # for "last N transactions" — set to N

    @field_validator("intent", mode="before")
    @classmethod
    def _normalise_intent(cls, v):
        if isinstance(v, str):
            return v.strip().upper()
        return v

    @field_validator("type", mode="before")
    @classmethod
    def _normalise_type(cls, v):
        if isinstance(v, str):
            return v.strip().lower()
        return v

    @field_validator("transaction_id", mode="before")
    @classmethod
    def _clean_tx_id(cls, v):
        if isinstance(v, str):
            cleaned = v.replace("#", "").strip().upper()
            return cleaned or None
        return v


class MultiTransactionData(BaseModel):
    actions: list[ActionItem]


# ---------------------------------------------------------------------------
# System prompt
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
| TRANSFER      | Money moved between two accounts/wallets (e.g. BOC -> Wallet)       |
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
- `amount`          : Numeric. Convert shorthand: 1k->1000, 1.5k->1500, 1m->1000000.
- `category`        : Infer from context if not stated (Food, Transport, Bills, Salary, Entertainment, etc.).
- `date`            : YYYY-MM-DD. Resolve relative dates using today's date provided in the user message.
  - "yesterday" -> subtract 1 day from today
  - "last Monday" -> calculate the most recent Monday
  - If no date mentioned, use today's date.
- `account`         : The account/wallet name.
  - IMPORTANT: If the user does NOT mention an account/wallet name, ALWAYS default to "Cash" for both income and expense LOG entries. Never leave account null.
  - CRITICAL FOR CREDIT CARDS: When setting a CREDIT_LIMIT or logging to a credit card, the account name MUST ALWAYS end with the words "Credit Card" (e.g., "BOC Credit Card", "Visa Credit Card"). If the user says "Set my BOC credit limit", output account as "BOC Credit Card", NOT just "BOC". This prevents collisions with regular bank accounts.
- `from_account`    : Source account for TRANSFER.
- `to_account`      : Destination account for TRANSFER.
- `transaction_id`  : Extract from patterns like "#ABC12" -> "ABC12" (strip the #, uppercase).
- `note`            : Any additional note or description the user provides.
- `query_limit`     : For QUERY only. If the user asks for "last N transactions" (e.g. "last 5", "show 10 transactions", "recent 20"), set this to the integer N. Otherwise leave null.

## HANDLING MULTI-INTENT MESSAGES
A single message can contain multiple actions. Return ALL of them as separate items in the `actions` array.

## EXAMPLE MAPPINGS
User: "Got paid 50k salary into BOC today and spent 500 on lunch from wallet"
-> [ {intent:"LOG", type:"income", amount:50000, category:"Salary", account:"BOC", date:"<today>"},
     {intent:"LOG", type:"expense", amount:500, category:"Food", account:"Wallet", date:"<today>"} ]

User: "Move 10k from BOC to my Wallet"
-> [ {intent:"TRANSFER", amount:10000, from_account:"BOC", to_account:"Wallet"} ]

User: "Paid my 5k Visa credit card bill from BOC"
-> [ {intent:"TRANSFER", amount:5000, from_account:"BOC", to_account:"Visa Credit Card"} ]

User: "Got 2000 refunded to my credit card"
-> [ {intent:"LOG", type:"income", amount:2000, category:"Refund", account:"Credit Card", date:"<today>"} ]

User: "Lent 5k to John from wallet"
-> [ {intent:"LOG", type:"expense", amount:5000, category:"Debt: John", account:"Wallet", date:"<today>"} ]

User: "John paid back 5k to BOC"
-> [ {intent:"LOG", type:"income", amount:5000, category:"Debt: John", account:"BOC", date:"<today>"} ]

User: "Set my Commercial Bank credit card limit to 200k"
-> [ {intent:"CREDIT_LIMIT", amount:200000, account:"Commercial Bank Credit Card", type:"limit"} ]

User: "Update #A3K9F amount to 600"
-> [ {intent:"UPDATE", transaction_id:"A3K9F", amount:600} ]

User: "Delete #XY99Z"
-> [ {intent:"DELETE", transaction_id:"XY99Z"} ]

User: "how much did I spend this month?"
-> [ {intent:"QUERY"} ]

User: "show my last 5 transactions" or "last 10 records"
-> [ {intent:"QUERY", query_limit:5} ]   (use the exact number the user said)

User: "hi" or "help"
-> [ {intent:"HELP"} ]

## UNKNOWN / INVALID INPUT
If the message has no financial meaning (e.g. random text, jokes), return:
[ {intent:"HELP"} ]
"""


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------
def _is_transient(err: str) -> bool:
    return any(marker in err for marker in _TRANSIENT_MARKERS)


def _salvage(raw: str) -> list[dict]:
    """Best-effort recovery: keep whichever action items validate individually."""
    try:
        blob = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return []
    items = blob.get("actions", []) if isinstance(blob, dict) else []
    good = []
    for item in items:
        try:
            good.append(ActionItem.model_validate(item).model_dump())
        except ValidationError:
            continue
    return good


async def parse_message(message_text: str, today: str) -> list[dict]:
    """
    Parse a message into validated action dicts.

    Retries on transient Groq errors (rate limit / 5xx) with back-off, and on
    JSON/validation failures by feeding the error back to the model. Falls back
    to a lenient salvage pass if every attempt still fails validation.
    """
    schema   = MultiTransactionData.model_json_schema()
    messages = [
        {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
        {"role": "user",   "content": f'Today\'s date is {today}. User message: "{message_text}"'},
    ]
    last_raw = "{}"

    for attempt in range(MAX_RETRIES):
        try:
            response = await llm.traced_chat(
                "router",
                model=llm.ROUTER_MODEL,
                messages=messages,
                response_format={
                    "type": "json_schema",
                    "json_schema": {"name": "multi_transaction_data", "schema": schema},
                },
                temperature=0.1,
                max_tokens=1024,
            )
        except Exception as e:
            if _is_transient(str(e)) and attempt < MAX_RETRIES - 1:
                wait = 2 ** attempt
                print(f"[Router] Transient error, retrying in {wait}s (attempt {attempt + 1})")
                await asyncio.sleep(wait)
                continue
            print(f"[Router Error] {e}")
            raise

        last_raw = response.choices[0].message.content or "{}"
        try:
            data = MultiTransactionData.model_validate_json(last_raw)
            return [a.model_dump() for a in data.actions]
        except (json.JSONDecodeError, ValidationError) as ve:
            print(f"[Router] Validation failed (attempt {attempt + 1}): {ve}")
            if attempt < MAX_RETRIES - 1:
                messages.append({"role": "assistant", "content": last_raw})
                messages.append({
                    "role": "user",
                    "content": (
                        "Your previous reply did not match the required schema:\n"
                        f"{ve}\n"
                        "Return ONLY corrected JSON that matches the schema. No prose."
                    ),
                })
                continue

    # Every attempt failed validation — keep whatever is individually valid.
    return _salvage(last_raw)
