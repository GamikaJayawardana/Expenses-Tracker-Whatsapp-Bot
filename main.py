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
from google import genai
from google.genai import types
from motor.motor_asyncio import AsyncIOMotorClient

# Load environment variables
load_dotenv()

app = FastAPI(title="Gemini Finance Bot - Master Edition")

# --- Configuration ---
VERIFY_TOKEN = os.getenv("WHATSAPP_VERIFY_TOKEN")
ACCESS_TOKEN = os.getenv("WHATSAPP_ACCESS_TOKEN")
PHONE_NUMBER_ID = os.getenv("WHATSAPP_PHONE_ID")
MONGODB_URI = os.getenv("MONGODB_URI")
SL_TIMEZONE = pytz.timezone('Asia/Colombo')

# --- Initialize Clients ---
gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
mongo_client = AsyncIOMotorClient(MONGODB_URI)
db = mongo_client.expense_tracker
transactions_collection = db.transactions

# --- Data Schemas ---

class ActionItem(BaseModel):
    intent: str  # LOG, TRANSFER, INITIALIZE, QUERY, DELETE, UPDATE, DELETE_ALL, HELP
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

# --- Helper Functions ---

def generate_tx_id():
    """Generates a short 5-character unique ID."""
    return ''.join(secrets.choice(string.ascii_uppercase + string.digits) for _ in range(5))

async def send_whatsapp_message(to_phone_number: str, text: str):
    """Helper to fire text back to WhatsApp."""
    url = f"https://graph.facebook.com/v18.0/{PHONE_NUMBER_ID}/messages"
    headers = {"Authorization": f"Bearer {ACCESS_TOKEN}", "Content-Type": "application/json"}
    data = {"messaging_product": "whatsapp", "to": to_phone_number, "type": "text", "text": {"body": text}}
    async with httpx.AsyncClient() as client:
        response = await client.post(url, headers=headers, json=data)
        if response.status_code != 200:
            print(f"WhatsApp Error: {response.text}")

# --- Background Worker Logic ---

async def process_user_message(sender_phone: str, message_text: str):
    print(f"\n[Worker] Processing: '{message_text}' from {sender_phone}")
    
    try:
        today = datetime.now(SL_TIMEZONE).strftime('%Y-%m-%d')
        max_retries = 3
        actions = []
        
        # 1. BRAIN: Extract actions with Retry Logic for 503/High Demand
        for attempt in range(max_retries):
            try:
                response = gemini_client.models.generate_content(
                    model='gemini-3-flash-preview',
                    contents=f"Today's date is {today}. Analyze: '{message_text}'",
                    config=types.GenerateContentConfig(
                        system_instruction="""You are a multi-action financial router. 
                        Break the message into a list of actions.
                        Intents: LOG, TRANSFER, INITIALIZE, QUERY, DELETE, UPDATE, DELETE_ALL, HELP.
                        For 'INITIALIZE', type is 'balance' or 'limit'.
                        Extract IDs like #ABC12 as 'ABC12' (remove #). 
                        If the user is correcting a previous typo (like 'it is wallet not 8n wallet'), set intent to UPDATE and find the relevant ID.""",
                        response_mime_type="application/json",
                        response_schema=MultiTransactionData,
                    ),
                )
                actions = json.loads(response.text).get("actions", [])
                break # Success
            except Exception as e:
                if ("503" in str(e) or "429" in str(e)) and attempt < max_retries - 1:
                    print(f"Gemini busy... retrying in 2s (Attempt {attempt+1})")
                    await asyncio.sleep(2)
                else:
                    await send_whatsapp_message(sender_phone, "⏳ My AI brain is a bit overwhelmed right now. Please try again in a few seconds!")
                    return

        # 2. EXECUTION: Process each identified action
        for action in actions:
            intent = action.get("intent")
            raw_id = action.get("transaction_id", "")
            tx_id = raw_id.upper().replace("#", "") if raw_id else None

            # ACTION: LOG
            if intent == "LOG" and action.get("amount"):
                new_id = generate_tx_id()
                acc = action.get("account") or "Cash"
                doc = {
                    "user_phone": sender_phone, "tx_id": new_id, "type": action["type"],
                    "amount": action["amount"], "category": action["category"],
                    "account": acc, "date": action["date"] or today, "created_at": datetime.now(SL_TIMEZONE)
                }
                await transactions_collection.insert_one(doc)
                icon = "🔴" if action["type"] == "expense" else "🟢"
                await send_whatsapp_message(sender_phone, f"{icon} *Logged Rs. {action['amount']}*\nAcc: {acc}\nID: `#{new_id}`")

            # ACTION: TRANSFER
            elif intent == "TRANSFER" and action.get("amount"):
                new_id = generate_tx_id()
                doc = {
                    "user_phone": sender_phone, "tx_id": new_id, "type": "transfer",
                    "amount": action["amount"], "from_account": action["from_account"],
                    "to_account": action["to_account"], "date": action["date"] or today, "created_at": datetime.now(SL_TIMEZONE)
                }
                await transactions_collection.insert_one(doc)
                await send_whatsapp_message(sender_phone, f"🔄 *Transfer Rs. {action['amount']}*\n{action['from_account']} ➡️ {action['to_account']}\nID: `#{new_id}`")

            # ACTION: INITIALIZE
            elif intent == "INITIALIZE" and action.get("amount"):
                new_id = generate_tx_id()
                acc = action.get("account") or "Wallet"
                type_label = action.get("type") or "balance"
                doc = {
                    "user_phone": sender_phone, "tx_id": new_id, "type": f"init_{type_label}",
                    "amount": action["amount"], "account": acc, "date": today, "created_at": datetime.now(SL_TIMEZONE)
                }
                await transactions_collection.insert_one(doc)
                await send_whatsapp_message(sender_phone, f"⚙️ *{acc}* {type_label} initialized to Rs. {action['amount']}.\nID: `#{new_id}`")

            # ACTION: UPDATE
            elif intent == "UPDATE" and tx_id:
                update_fields = {k: v for k, v in action.items() if v is not None and k not in ["intent", "transaction_id"]}
                res = await transactions_collection.update_one(
                    {"user_phone": sender_phone, "tx_id": tx_id}, {"$set": update_fields}
                )
                if res.modified_count:
                    await send_whatsapp_message(sender_phone, f"📝 Updated record `#{tx_id}`.")
                else:
                    await send_whatsapp_message(sender_phone, f"❌ ID `#{tx_id}` not found.")

            # ACTION: DELETE
            elif intent == "DELETE" and tx_id:
                res = await transactions_collection.delete_one({"user_phone": sender_phone, "tx_id": tx_id})
                if res.deleted_count:
                    await send_whatsapp_message(sender_phone, f"🗑️ Deleted record `#{tx_id}`.")
                else:
                    await send_whatsapp_message(sender_phone, f"❌ ID `#{tx_id}` not found.")

            # ACTION: QUERY
            elif intent == "QUERY":
                cursor = transactions_collection.find({"user_phone": sender_phone}).sort("created_at", -1).limit(50)
                data = await cursor.to_list(length=50)
                history = []
                for t in data:
                    tid, tt = t.get('tx_id', 'INIT'), t.get("type")
                    if tt == "transfer": history.append(f"#{tid} | {t['date']} | XFR | Rs.{t['amount']} | {t.get('from_account')}->{t.get('to_account')}")
                    elif "init_" in str(tt): history.append(f"START | {t['date']} | {t['account']} {tt.split('_')[1]} = Rs.{t['amount']}")
                    else: history.append(f"#{tid} | {t['date']} | {tt} | Rs.{t['amount']} | {t.get('category')} | Acc:{t.get('account')}")
                
                history_str = "\n".join(history) if history else "No records found."
                q_prompt = f"Today: {today}. History (ID|Date|Type|Amt|Info):\n{history_str}\n\nUser Question: {message_text}. Answer briefly with bold/emojis."
                
                # Nested retry for the query response
                for q_attempt in range(max_retries):
                    try:
                        q_res = gemini_client.models.generate_content(model='gemini-3-flash-preview', contents=q_prompt)
                        await send_whatsapp_message(sender_phone, q_res.text)
                        break
                    except:
                        await asyncio.sleep(1)

            # ACTION: DELETE_ALL
            elif intent == "DELETE_ALL":
                await transactions_collection.delete_many({"user_phone": sender_phone})
                await send_whatsapp_message(sender_phone, "🧹 *Wiped:* All your data has been deleted.")

            # ACTION: HELP
            elif intent == "HELP":
                help_msg = (
                    "👋 *Hi! I'm your Finance Bot.*\n\n"
                    "• *Setup:* 'I have 5k in BOC'\n"
                    "• *Log:* 'Spent 500 on Food from Wallet'\n"
                    "• *XFR:* 'Move 1k from BOC to Wallet'\n"
                    "• *Edit:* 'Update #ID amount to 600' or 'Delete #ID'\n"
                    "• *Ask:* 'How much did I spend today?'\n\n"
                    "I handle multiple things at once! Try: 'I got 2k and spent 500 for lunch'."
                )
                await send_whatsapp_message(sender_phone, help_msg)

    except Exception as e:
        print(f"Fatal Worker Error: {e}")
        await send_whatsapp_message(sender_phone, "⚠️ I hit a snag. Please try your message again.")

# --- FastAPI Webhook Routes ---

@app.get("/webhook")
async def verify(mode: str = Query(None, alias="hub.mode"), token: str = Query(None, alias="hub.verify_token"), challenge: int = Query(None, alias="hub.challenge")):
    if mode == "subscribe" and token == VERIFY_TOKEN: return PlainTextResponse(str(challenge))
    raise HTTPException(status_code=403)

@app.post("/webhook")
async def handle_hook(request: Request, bg: BackgroundTasks):
    try:
        raw = await request.json()
        val = raw['entry'][0]['changes'][0]['value']
        if 'messages' in val:
            msg_obj = val['messages'][0]
            bg.add_task(process_user_message, msg_obj['from'], msg_obj['text']['body'])
    except: pass
    return {"status": "success"}