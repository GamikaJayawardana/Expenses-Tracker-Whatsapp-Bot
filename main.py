import os
import json
import httpx
import asyncio
import secrets
import string
from datetime import datetime
import pytz
from fastapi import FastAPI, Request, Query, HTTPException, BackgroundTasks
from fastapi.responses import PlainTextResponse
from dotenv import load_dotenv
from pydantic import BaseModel
from groq import AsyncGroq
from motor.motor_asyncio import AsyncIOMotorClient

# Load configuration
load_dotenv()

app = FastAPI(title="Groq Finance Bot: Final Master Edition")

# --- Configuration & Constants ---
SL_TIMEZONE = pytz.timezone('Asia/Colombo')
MONGODB_URI = os.getenv("MONGODB_URI")
WH_TOKEN = os.getenv("WHATSAPP_VERIFY_TOKEN")
ACC_TOKEN = os.getenv("WHATSAPP_ACCESS_TOKEN")
PHONE_ID = os.getenv("WHATSAPP_PHONE_ID")

# --- System State ---
processed_msg_ids = set()

# --- Clients ---
groq_client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))
mongo_client = AsyncIOMotorClient(MONGODB_URI)
db = mongo_client.expense_tracker
transactions_collection = db.transactions

# --- Data Models ---
class ActionItem(BaseModel):
    intent: str # LOG, TRANSFER, INITIALIZE, QUERY, DELETE, UPDATE, DELETE_ALL, HELP
    type: str | None          
    amount: float | None
    category: str | None
    date: str | None
    account: str | None       
    from_account: str | None  
    to_account: str | None    
    transaction_id: str | None 

class MultiTransactionData(BaseModel):
    actions: list[ActionItem]

# --- Core Utilities ---
def generate_tx_id():
    return ''.join(secrets.choice(string.ascii_uppercase + string.digits) for _ in range(5))

async def send_whatsapp(to, text):
    url = f"https://graph.facebook.com/v18.0/{PHONE_ID}/messages"
    headers = {"Authorization": f"Bearer {ACC_TOKEN}", "Content-Type": "application/json"}
    payload = {"messaging_product": "whatsapp", "to": to, "type": "text", "text": {"body": text}}
    async with httpx.AsyncClient() as client:
        res = await client.post(url, headers=headers, json=payload)
        if res.status_code != 200:
            print(f"❌ WhatsApp API Send Error: {res.text}")

# --- AI Routing Engine ---
async def get_actions_from_ai(message_text, today):
    system_prompt = (
        f"Today is {today}. You are a financial router.\n"
        "Analyze the message and return a JSON list of actions.\n"
        "INTENTS:\n"
        "- 'DELETE_ALL': Use when user wants to wipe, clear, reset, or delete EVERYTHING/ALL.\n"
        "- 'DELETE': Use only for a specific #ID.\n"
        "- 'QUERY': Use for balance, summary, totals, or 'what I have'.\n"
        "- 'UPDATE': Use for corrections to existing #IDs.\n"
        "- 'LOG', 'TRANSFER', 'INITIALIZE', 'HELP'.\n"
        "Return valid JSON with an 'actions' list."
    )
    
    try:
        completion = await groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": message_text}],
            response_format={"type": "json_object"}
        )
        return json.loads(completion.choices[0].message.content).get("actions", [])
    except Exception as e:
        print(f"❌ Router AI Error: {e}")
        return []

# --- Business Logic Worker ---
async def process_worker(sender, text):
    try:
        today = datetime.now(SL_TIMEZONE).strftime('%Y-%m-%d')
        actions = await get_actions_from_ai(text, today)

        if not actions:
            if any(w in text.lower() for w in ["hi", "hello", "help"]):
                actions = [{"intent": "HELP"}]
            else:
                return

        for action in actions:
            intent = action.get("intent")
            raw_id = action.get("transaction_id") or ""
            tid = raw_id.upper().replace("#", "").strip()

            # 1. DELETE ALL (HIGH PRIORITY)
            if intent == "DELETE_ALL":
                res = await transactions_collection.delete_many({"user_phone": sender})
                await send_whatsapp(sender, f"🧹 *Wipe Complete:* All {res.deleted_count} records deleted. Your account is fresh.")
                continue

            # 2. INITIALIZE (Balance Set)
            elif intent == "INITIALIZE" and action.get("amount"):
                new_id = generate_tx_id()
                acc = action.get("account") or "Wallet"
                doc = {
                    "user_phone": sender, "tx_id": new_id, "type": f"init_{action.get('type','balance')}",
                    "amount": action["amount"], "account": acc, "date": today, "created_at": datetime.now(SL_TIMEZONE)
                }
                await transactions_collection.insert_one(doc)
                await send_whatsapp(sender, f"⚙️ *{acc}* set to Rs. {action['amount']}.\nID: `#{new_id}`")

            # 3. LOGGING (Expenses)
            elif intent == "LOG" and action.get("amount"):
                new_id = generate_tx_id()
                acc = action.get("account") or "Wallet"
                doc = {
                    "user_phone": sender, "tx_id": new_id, "type": action.get("type") or "expense",
                    "amount": action["amount"], "category": action.get("category") or "General",
                    "account": acc, "date": today, "created_at": datetime.now(SL_TIMEZONE)
                }
                await transactions_collection.insert_one(doc)
                await send_whatsapp(sender, f"🛍️ *Logged Rs. {action['amount']}* ({acc})\nID: `#{new_id}`")

            # 4. TRANSFER
            elif intent == "TRANSFER" and action.get("amount"):
                new_id = generate_tx_id()
                doc = {
                    "user_phone": sender, "tx_id": new_id, "type": "transfer",
                    "amount": action["amount"], "from_account": action.get("from_account"),
                    "to_account": action.get("to_account"), "date": today, "created_at": datetime.now(SL_TIMEZONE)
                }
                await transactions_collection.insert_one(doc)
                await send_whatsapp(sender, f"🔄 *Transfer:* Rs. {action['amount']}\n{doc['from_account']} ➡️ {doc['to_account']}\nID: `#{new_id}`")

            # 5. UPDATE & DELETE
            elif intent == "UPDATE" and tid:
                fields = {k: v for k, v in action.items() if v is not None and k not in ["intent", "transaction_id"]}
                res = await transactions_collection.update_one({"user_phone": sender, "tx_id": tid}, {"$set": fields})
                await send_whatsapp(sender, f"📝 Updated `#{tid}`." if res.modified_count else f"❌ ID `#{tid}` not found.")

            elif intent == "DELETE" and tid:
                res = await transactions_collection.delete_one({"user_phone": sender, "tx_id": tid})
                await send_whatsapp(sender, f"🗑️ Deleted `#{tid}`." if res.deleted_count else f"❌ ID `#{tid}` not found.")

            # 6. QUERY (Summary Math)
            elif intent == "QUERY":
                cursor = transactions_collection.find({"user_phone": sender}).sort("created_at", -1).limit(60)
                records = await cursor.to_list(length=60)
                history = []
                for t in records:
                    tt = t.get("type")
                    if "init" in str(tt): history.append(f"START | {t['date']} | {t['account']} balance Rs.{t['amount']}")
                    elif tt == "transfer": history.append(f"XFR | {t['date']} | Rs.{t['amount']} from {t.get('from_account')} to {t.get('to_account')}")
                    else: history.append(f"{str(tt).upper()} | {t['date']} | Rs.{t['amount']} | Acc: {t.get('account')}")
                
                hist_str = "\n".join(history) if history else "No history."
                q_prompt = (
                    f"You are a Sri Lankan Financial Assistant. Today: {today}.\n"
                    f"History:\n{hist_str}\n\nUser Question: {text}\n"
                    "Instructions: Calculate current balances by starting from 'START' values. List each account and total clearly."
                )
                analysis = await groq_client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role": "user", "content": q_prompt}]
                )
                await send_whatsapp(sender, analysis.choices[0].message.content)

            # 7. HELP
            elif intent == "HELP":
                await send_whatsapp(sender, "📘 *Finance Bot*\n- 'I have 5k in Wallet'\n- 'Spent 500 for lunch'\n- 'Transfer 1k from BOC to Wallet'\n- 'Delete all data'")

    except Exception as e:
        print(f"❌ Worker Error: {e}")

# --- API Endpoints ---
@app.get("/webhook")
async def verify(mode: str = Query(None, alias="hub.mode"), token: str = Query(None, alias="hub.verify_token"), challenge: int = Query(None, alias="hub.challenge")):
    if mode == "subscribe" and token == WH_TOKEN: return PlainTextResponse(str(challenge))
    raise HTTPException(status_code=403)

@app.post("/webhook")
async def handle(request: Request, bg: BackgroundTasks):
    try:
        raw = await request.json()
        value = raw['entry'][0]['changes'][0]['value']
        
        if 'statuses' in value: return {"status": "success"}

        if 'messages' in value:
            msg = value['messages'][0]
            mid = msg.get('id')
            
            # Prevent processing the same message during restarts
            if mid in processed_msg_ids: return {"status": "success"}
            processed_msg_ids.add(mid)
            if len(processed_msg_ids) > 100: processed_msg_ids.pop()

            print(f"📥 INCOMING: {msg.get('text', {}).get('body')}")
            bg.add_task(process_worker, msg['from'], msg.get('text', {}).get('body'))
            
    except: pass
    return {"status": "success"}