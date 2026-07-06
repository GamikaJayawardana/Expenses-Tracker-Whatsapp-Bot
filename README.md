# Expenses Tracker WhatsApp Bot

A personal finance tracker that runs entirely inside WhatsApp. Instead of opening a spreadsheet or a budgeting app, you text the bot in plain English and it logs, transfers, edits, or summarizes your finances for you.

The bot parses free-form messages with an LLM-based router (Groq, Llama 3.3 70B) into structured actions, then executes them against a MongoDB-backed ledger — handling multiple transactions in a single message, running balances, credit card limits, and full CRUD on past entries.

## Features

* **Natural language input** — no fixed commands or menus; describe transactions the way you'd say them out loud.
* **Multi-action parsing** — a single message like *"Spent 500 on lunch and got my 50k salary"* is split into separate, correctly-typed transactions.
* **Running balances** — balances and credit limits are computed on demand from the full transaction history, including transfers between accounts.
* **Credit card tracking** — set a limit per card and get spend/remaining/utilization on every transaction.
* **Edit and delete by ID** — every transaction gets a short unique ID so you can correct or remove it later (`Update #DM5VD amount to 4000`).
* **Resilient webhook handling** — background task processing and retry/back-off logic for WhatsApp Cloud API and Groq rate limits.

## Tech Stack

* **Backend:** Python 3, FastAPI, Uvicorn
* **Database:** MongoDB (Motor async driver)
* **LLM:** Groq API (Llama 3.3 70B) for intent parsing and free-text query answers
* **Messaging:** Meta WhatsApp Cloud API

## Setup

### 1. Prerequisites
* Python 3.10+
* A MongoDB cluster (the free Atlas tier works)
* A [Groq](https://console.groq.com/) API key
* A Meta Developer account with a WhatsApp Business app configured

### 2. Clone the repository
```bash
git clone https://github.com/GamikaJayawardana/Expenses-Tracker-Whatsapp-Bot.git
cd Expenses-Tracker-Whatsapp-Bot
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure environment variables
Create a `.env` file in the project root:
```env
WHATSAPP_VERIFY_TOKEN=your_custom_verify_token
WHATSAPP_ACCESS_TOKEN=your_long_lived_meta_token
WHATSAPP_PHONE_ID=your_whatsapp_phone_number_id
MONGODB_URI=mongodb+srv://<username>:<password>@cluster.mongodb.net/
GROQ_API_KEY=gsk_your_groq_api_key
```

### 5. Run the server
```bash
uvicorn main:app --reload --port 8000
```
Expose port 8000 with a tool like ngrok to point the Meta webhook dashboard at it.

## Usage

Once the server is running and the webhook is verified, message the bot on WhatsApp:

**Set up accounts**
> "I have 45000 in ComBank, 12000 in BOC, and 3000 in my Wallet."

**Log income and expenses**
> "Spent 150 for the bus from Wallet."
> "Paid CEB bill 4500 from ComBank and did a Dialog reload for 500 from BOC."
> "Got a freelance payment of 15000 to ComBank."

**Transfer between accounts**
> "I withdrew 10000 from ComBank and put it to Wallet."

**Edit or delete a transaction**
> "Update #DM5VD amount to 4000."
> "Delete #TRXM4."

**Query your data**
> "What is my balance now?"
> "Give me a full summary of what I have."

**Reset**
> "Clear all my data" — wipes all stored transactions for that user.

## License
Licensed under the MIT License — see [LICENSE](LICENSE) for details.
