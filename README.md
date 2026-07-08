# Expenses Tracker WhatsApp Bot

A personal finance tracker that runs entirely inside WhatsApp. Instead of opening a spreadsheet or a budgeting app, you text the bot in plain English and it logs, transfers, edits, or summarizes your finances for you.

The bot parses free-form messages with an LLM-based router (Groq, Llama-4 Scout) into structured actions, then executes them against a MongoDB-backed ledger — handling multiple transactions in a single message, running balances, credit card limits, and full CRUD on past entries.

Router output is validated against Pydantic schemas (with an automatic retry when the model returns something off-schema), free-text questions are answered through function/tool calling, and every LLM call is logged to a local trace table for cost, latency and reliability tracking.

## Features

* **Natural language input** — no fixed commands or menus; describe transactions the way you'd say them out loud.
* **Multi-action parsing** — a single message like *"Spent 500 on lunch and got my 50k salary"* is split into separate, correctly-typed transactions.
* **Validated structured output** — every parse is checked against a Pydantic schema; malformed responses trigger a corrective retry, then a lenient salvage pass, so bad JSON never reaches the ledger.
* **Tool-calling query agent** — questions like *"show my spending this month"* let the model call typed tools (`get_spending`, `get_account_balances`, `get_credit_info`, `get_recent_transactions`) that read from the DB, keeping the numbers computed in Python.
* **Running balances** — balances and credit limits are computed on demand from the full transaction history, including transfers between accounts.
* **Credit card tracking** — set a limit per card and get spend/remaining/utilization on every transaction.
* **Edit and delete by ID** — every transaction gets a short unique ID so you can correct or remove it later (`Update #DM5VD amount to 4000`).
* **LLM observability** — cost, latency, token usage and success rate are logged per call and exposed at `GET /metrics`.
* **Extraction eval harness** — a 100-message labelled set with a runner that reports intent/type/amount/category accuracy.
* **Resilient webhook handling** — background task processing and retry/back-off logic for WhatsApp Cloud API and Groq rate limits.

## Tech Stack

* **Backend:** Python 3, FastAPI, Uvicorn
* **Database:** MongoDB (Motor async driver)
* **LLM:** Groq API (Llama-4 Scout) for intent parsing, structured output, and tool-calling query answers
* **Validation:** Pydantic v2 schemas for the router output
* **Observability:** SQLite trace table (no external dependency)
* **Messaging:** Meta WhatsApp Cloud API

## Project Structure

| File          | Responsibility                                                        |
|---------------|-----------------------------------------------------------------------|
| `main.py`     | FastAPI app, WhatsApp webhook, and the message-processing worker      |
| `router.py`   | Pydantic action schemas + the validated, retrying intent parser       |
| `tools.py`    | Tool-calling query agent and its tool definitions                     |
| `store.py`    | MongoDB access — balances, credit info, spending and history queries  |
| `llm.py`      | Groq client and a traced chat-completion wrapper                      |
| `tracing.py`  | SQLite trace table for cost / latency / reliability                   |
| `evals/`      | Labelled dataset and the extraction-accuracy runner                   |

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

## Evaluation

`evals/dataset.jsonl` holds 100 labelled messages with the expected intent, type, amount and category. The runner sends each through the router and reports extraction accuracy (amount is matched exactly; category uses a small alias table).

```bash
python evals/run_eval.py             # full 100-message set
python evals/run_eval.py --limit 20  # quick subset
python evals/run_eval.py --delay 0.5 # throttle to stay under rate limits
```

The report breaks accuracy down by field and prints the cost/latency for the run from the trace table.

## Observability

Every LLM call is timed and written to a local SQLite trace table (`traces.db` by default; override with `TRACE_DB_PATH`). Aggregate metrics are available while the server is running:

```bash
curl http://localhost:8000/metrics
```

```json
{
  "calls": 42,
  "total_cost_usd": 0.0084,
  "avg_latency_ms": 910.4,
  "success_rate": 1.0,
  "by_call_type": [ ... ]
}
```

Per-token prices live in `MODEL_PRICES` in `tracing.py` — update them to match the current Groq price sheet.

## License
Licensed under the MIT License — see [LICENSE](LICENSE) for details.
