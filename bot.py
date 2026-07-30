import os
import json
import threading
import logging
from datetime import datetime, timezone

import requests
from openai import OpenAI
from flask import Flask
try:
    from telegram import Update
    from telegram.ext import ApplicationBuilder, ContextTypes, MessageHandler, filters
except ModuleNotFoundError as e:
    print("Required package 'python-telegram-bot' is not installed.\nInstall dependencies with: python -m pip install -r requirements.txt")
    raise

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_BASE_URL = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
LOG_PUBLIC_URL = os.getenv("LOG_PUBLIC_URL", "none")
LOCAL_LOG_PATH = os.getenv("LOCAL_LOG_PATH", "run.jsonl")
PORT = int(os.getenv("PORT", "10000"))  # Render sets $PORT for web services

# --- Gist-based log upload ---
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GIST_ID = os.getenv("GIST_ID")
GIST_FILENAME = os.getenv("GIST_FILENAME", "run.jsonl")

if not TELEGRAM_BOT_TOKEN:
    logger.error("TELEGRAM_BOT_TOKEN not set. Exiting.")

if not GROQ_API_KEY:
    logger.warning("GROQ_API_KEY not set. Model calls will fail if attempted.")

# --- Groq client (OpenAI-compatible, free developer tier) ---
client = OpenAI(api_key=GROQ_API_KEY, base_url=GROQ_BASE_URL) if GROQ_API_KEY else None

PROMPT_INSTRUCTIONS = (
    "You are a careful data-analyst agent. The user will send a plain-text message asking a data-analysis question. "
    "You must reply with exactly one JSON object and nothing else. The object MUST have two keys: \"answer\" and \"log_url\". "
    "The value of \"answer\" should be shaped exactly as the user's message requests (for example, if the user asks 'Reply with ONLY this JSON object: {\"answer\": {\"state\": \"<state name>\"}, \"log_url\": \"<url>\"}', then \"answer\" must be an object with key \"state\", etc.). "
    "The value of \"log_url\" must be the public wget-able URL where the agent's run log will be available. Use the provided LOG_PUBLIC_URL value. Do not include any explanatory text or extra fields. "
    "If you need to fetch data from a URL mentioned in the user's message, do so. If the user provided inline CSV or data, parse it. Keep outputs concise and strictly follow the requested JSON shape."
)


def call_openai(system_prompt: str, user_prompt: str, max_tokens: int = 1000):
    """Calls a model via Groq (OpenAI-compatible), returns (model_name, text)."""
    model_name = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    resp = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
        max_tokens=max_tokens,
    )
    text = resp.choices[0].message.content.strip()
    return model_name, text


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    text = update.message.text.strip() if update.message and update.message.text else ""
    logger.info("Received message from %s (%s): %s", user.username, user.id, text)

    timestamp = datetime.now(timezone.utc).isoformat()

    system_prompt = PROMPT_INSTRUCTIONS + f"\nLOG_PUBLIC_URL={LOG_PUBLIC_URL}\n"
    user_prompt = (
        "Here is the user's message. Follow my earlier instructions and reply with exactly one JSON object and nothing else.\n"
        "Message:\n" + text + "\n\nRespond with only the JSON object."
    )

    model_response_text = None
    model_used = None
    try:
        if client is None:
            raise RuntimeError("GROQ_API_KEY not configured")
        model_used, model_response_text = call_openai(system_prompt, user_prompt)
        logger.info("Model reply: %s", model_response_text)
    except Exception as e:
        logger.exception("Model call failed: %s", e)
        fallback = {"answer": {"error": "model_call_failed", "message": str(e)}, "log_url": LOG_PUBLIC_URL}
        model_response_text = json.dumps(fallback, ensure_ascii=False)

    # Ensure the model output is a single JSON object
    parsed = None
    try:
        parsed = json.loads(model_response_text)
        if not isinstance(parsed, dict):
            raise ValueError("Top-level JSON is not an object")
    except Exception as e:
        logger.warning("Model output not valid JSON object: %s", e)
        try:
            repair_prompt = (
                "The previous model output was not a valid single JSON object. Extract or produce the exact single JSON object required (with keys 'answer' and 'log_url') and nothing else. "
                "User message was:\n" + text + "\nPrevious model output:\n" + model_response_text
            )
            _, model_response_text = call_openai(PROMPT_INSTRUCTIONS, repair_prompt, max_tokens=800)
            parsed = json.loads(model_response_text)
            if not isinstance(parsed, dict):
                raise ValueError("Repaired output not an object")
        except Exception as e2:
            logger.exception("Repair failed: %s", e2)
            parsed = {"answer": {"error": "could_not_produce_valid_json"}, "log_url": LOG_PUBLIC_URL}
            model_response_text = json.dumps(parsed, ensure_ascii=False)

    # Ensure log_url in parsed is set to LOG_PUBLIC_URL
    try:
        parsed["log_url"] = LOG_PUBLIC_URL
    except Exception:
        parsed = {"answer": {"error": "invalid_parsed_structure"}, "log_url": LOG_PUBLIC_URL}
        model_response_text = json.dumps(parsed, ensure_ascii=False)

    # Write run log locally (append JSONL)
    run_entry = {
        "timestamp": timestamp,
        "user_id": user.id,
        "username": user.username,
        "message": text,
        "model": model_used,
        "response_text": model_response_text,
        "parsed_response": parsed,
    }
    try:
        with open(LOCAL_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(run_entry, ensure_ascii=False) + "\n")
    except Exception:
        logger.exception("Failed to write local log")

    # Upload the full local log to the GitHub Gist (overwrites the gist file
    # with the current full contents of LOCAL_LOG_PATH each time).
    if GITHUB_TOKEN and GIST_ID:
        try:
            with open(LOCAL_LOG_PATH, "r", encoding="utf-8") as f:
                full_log_content = f.read()

            gist_resp = requests.patch(
                f"https://api.github.com/gists/{GIST_ID}",
                headers={
                    "Authorization": f"Bearer {GITHUB_TOKEN}",
                    "Accept": "application/vnd.github+json",
                },
                json={
                    "files": {
                        GIST_FILENAME: {
                            "content": full_log_content
                        }
                    }
                },
                timeout=15,
            )
            logger.info("Gist update status=%s", gist_resp.status_code)
            if gist_resp.status_code >= 400:
                logger.error("Gist update failed: %s", gist_resp.text)
        except Exception:
            logger.exception("Log upload to gist failed")
    else:
        logger.warning("GITHUB_TOKEN or GIST_ID not set; skipping gist log upload")

    # Send exactly the JSON object as the reply text
    reply_text = json.dumps(parsed, ensure_ascii=False)
    await update.message.reply_text(reply_text)


# --- Flask keep-alive server -------------------------------------------------
# Render's Web Service type waits for something to bind $PORT. The Telegram
# bot itself never opens a port (it's a polling process), so without this,
# Render considers the deploy failed / keeps restarting the container.
# If you deploy this as a Render "Background Worker" instead, this part is
# not required, but it's harmless to leave in.
flask_app = Flask(__name__)


@flask_app.route("/")
def health():
    return "Bot is running", 200


def run_flask():
    flask_app.run(host="0.0.0.0", port=PORT)


def main():
    if not TELEGRAM_BOT_TOKEN:
        print("TELEGRAM_BOT_TOKEN environment variable required.")
        return

    # Start Flask in a background thread so Render sees an open port
    threading.Thread(target=run_flask, daemon=True).start()

    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
    handler = MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message)
    app.add_handler(handler)

    print(f"Bot starting. Flask health check on port {PORT}. Listening for Telegram messages...")
    app.run_polling()


if __name__ == "__main__":
    main()