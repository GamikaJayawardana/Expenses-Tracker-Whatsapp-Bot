# AI-Powered WhatsApp Finance Bot 🤖💸

An intelligent, natural-language personal finance assistant built directly into WhatsApp. Instead of manually categorizing expenses in spreadsheets or clicking through apps, you can simply text this bot as if it were a human accountant.

Powered by a hybrid LLM architecture (Groq's Llama 3.3 70B & Google Gemini), the bot can extract multiple financial intents from a single message, handle complex math, and maintain state via MongoDB.

## ✨ Features

* **Natural Language Routing:** Understands complex sentences. No need for strict commands or menus.
* **Multi-Action Processing:** Handles multiple transactions in one text (e.g., *"Spent 500 on lunch and got my 50k salary"*).
* **Smart Accountant Logic:** Calculates running balances accurately from your initialized starting points.
* **Resilient Webhook Handling:** Built-in duplicate detection and auto-retries to gracefully handle Meta API floods.
* **Self-Healing LLM Fallbacks:** Primary logic runs on Groq for ultra-fast inference, with silent fallbacks to Gemini if rate limits are hit.

## 🛠️ Tech Stack

* **Backend Framework:** Python 3, FastAPI, Uvicorn
* **Database:** MongoDB (Motor Asyncio)
* **AI & LLMs:** Groq API (Llama 3.3 70B), Google Gemini API
* **Integration:** Meta WhatsApp Cloud API

## 🚀 Quick Setup

### 1. Prerequisites
* Python 3.10+
* A [MongoDB](https://www.mongodb.com/) cluster (Atlas free tier works perfectly)
* A [Groq](https://console.groq.com/) API Key
* A Meta Developer Account with a WhatsApp Business App set up

### 2. Clone the Repository
```bash
git clone [https://github.com/gamikajayawardana/expenses-tracker-whatsapp-bot.git](https://github.com/gamikajayawardana/expenses-tracker-whatsapp-bot.git)
cd expenses-tracker-whatsapp-bot
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Environment Variables
Create a `.env` file in the root directory and add your credentials:
```env
WHATSAPP_VERIFY_TOKEN=your_custom_verify_token
WHATSAPP_ACCESS_TOKEN=your_long_lived_meta_token
WHATSAPP_PHONE_ID=your_whatsapp_phone_number_id
MONGODB_URI=mongodb+srv://<username>:<password>@cluster.mongodb.net/
GROQ_API_KEY=gsk_your_groq_api_key
```

### 5. Run the Server
```bash
uvicorn main:app --reload --port 8000
```
*(Note: You will need to expose your local port 8000 using a tool like Ngrok to connect it to the Meta Webhook dashboard).*

---

## 📱 How to Use It (Example Prompts)

Once the bot is running, just text it on WhatsApp! Here are the core commands the AI understands:

**1. Setup & Initialize**
> "I have 45000 in ComBank, 12000 in BOC, and 3000 in my Wallet."

**2. Log Daily Expenses & Income**
> "Spent 150 for the bus from Wallet."
> "Paid CEB bill 4500 from ComBank and did a Dialog reload for 500 from BOC."
> "Got a freelance payment of 15000 to ComBank."

**3. Transfers between accounts**
> "I withdrew 10000 from ComBank and put it to Wallet."

**4. Edit or Delete past mistakes**
> "Update #DM5VD amount to 4000."
> "Delete #TRXM4."

**5. Query Summaries**
> "What is my balance now?"
> "Give me a full summary of what I have."

**6. Hard Reset**
> "Clear all my data" (Wipes your user data from the database).

---

## 🤝 Contributing
Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/gamikajayawardana/expenses-tracker-whatsapp-bot/issues).

## 📝 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
