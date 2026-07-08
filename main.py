import os
import re
import httpx
from datetime import datetime
from fastapi import FastAPI, Request, Query, HTTPException, BackgroundTasks
from fastapi.responses import PlainTextResponse
from dotenv import load_dotenv

import tracing
import llm
from router import parse_message
from tools import run_query_agent
from store import (
    SL_TIMEZONE,
    transactions_collection,
    generate_tx_id,
    get_account_balances,
    get_credit_info,
)

# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------
load_dotenv()
tracing.init_db()

app = FastAPI(title="Groq Finance Bot - Master Edition")

# --- Configuration ---
VERIFY_TOKEN    = os.getenv("WHATSAPP_VERIFY_TOKEN")
ACCESS_TOKEN    = os.getenv("WHATSAPP_ACCESS_TOKEN")
PHONE_NUMBER_ID = os.getenv("WHATSAPP_PHONE_ID")

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

# Keywords that trigger a Python-computed summary (bypass the LLM entirely)
SUMMARY_RE = re.compile(
    r'\b(summ(?:a(?:r(?:y|ies)?)?|e?ry)?|balance|overview|total|how much.*have|account.*status)\b',
    re.IGNORECASE
)

# ---------------------------------------------------------------------------
# Core: Background worker — processes each WhatsApp message
# ---------------------------------------------------------------------------

async def process_user_message(sender_phone: str, message_text: str):
    print(f"\n[Worker] From={sender_phone} | Msg='{message_text}'")
    today = datetime.now(SL_TIMEZONE).strftime('%Y-%m-%d')

    try:
        # ── 1. BRAIN: Parse the message into structured, validated actions ──
        try:
            actions = await parse_message(message_text, today)
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
                    # ── Standard path: tool-calling query agent ─────────────
                    answer = await run_query_agent(sender_phone, message_text, today)
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
    return {"status": "ok", "engine": "Groq", "model": llm.ROUTER_MODEL}

@app.get("/metrics")
async def metrics():
    """Aggregate cost, latency and reliability across all logged LLM calls."""
    return tracing.summary()
