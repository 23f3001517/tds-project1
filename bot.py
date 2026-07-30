import os
import re
import json
import threading
import logging
from collections import defaultdict, deque
from datetime import datetime, timezone
from dotenv import load_dotenv
import requests
from google import genai
from google.genai import types
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
load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
LOG_PUBLIC_URL = os.getenv("LOG_PUBLIC_URL", "none")
LOCAL_LOG_PATH = os.getenv("LOCAL_LOG_PATH", "run.jsonl")
PORT = int(os.getenv("PORT", "10000"))  # Render sets $PORT for web services

# --- Gist-based log upload ---
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GIST_ID = os.getenv("GIST_ID")
GIST_FILENAME = os.getenv("GIST_FILENAME", "run.jsonl")

# Use a model that's actually good at reasoning + code, not just fast.
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

# How many previous messages (per chat) to keep for multi-turn questions.
HISTORY_TURNS = int(os.getenv("HISTORY_TURNS", "6"))
# Max characters of fetched URL content to include per URL.
URL_FETCH_CHAR_LIMIT = int(os.getenv("URL_FETCH_CHAR_LIMIT", "20000"))

if not TELEGRAM_BOT_TOKEN:
    logger.error("TELEGRAM_BOT_TOKEN not set. Exiting.")

if not GEMINI_API_KEY:
    logger.warning("GEMINI_API_KEY not set. Gemini calls will fail if attempted.")

# --- Gemini client ---
client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

# Per-chat rolling conversation history (in-memory; resets on restart).
# This is what makes multi-turn question sequences work: we keep the
# last few messages from the same chat and pass them as context, but we
# only ever answer the LAST message.
CHAT_HISTORY = defaultdict(lambda: deque(maxlen=HISTORY_TURNS))

URL_RE = re.compile(r"https?://[^\s)>\]\"']+")

PROMPT_INSTRUCTIONS = (
    "You are a careful, precise data-analyst agent. You will be given a data-analysis question, "
    "optionally some earlier messages from the same conversation for context, and optionally the raw "
    "content fetched from any URL mentioned in the question. "
    "You have a Python code-execution tool available: USE IT for every non-trivial calculation "
    "(sums, averages, filtering, sorting, comparisons across many rows, percentages, etc.). "
    "Never compute or estimate numbers 'in your head' in the text response — write and run Python code "
    "to get exact values, then report the result. "
    "After you finish any code execution, your VERY LAST line of output must be the exact marker "
    "===FINAL_JSON=== followed immediately on the next line by exactly one JSON object and nothing else "
    "after it (no markdown fences, no explanation, no extra text after the JSON). It is fine if earlier "
    "output contains code and code-execution results; only the content after ===FINAL_JSON=== will be read "
    "as your answer. "
    "The JSON object MUST have exactly two keys: \"answer\" and \"log_url\". "
    "The value of \"answer\" must be shaped EXACTLY as the user's message requests (for example, if the user asks "
    "'Reply with ONLY this JSON object: {\"answer\": {\"state\": \"<state name>\"}, \"log_url\": \"<url>\"}', then "
    "\"answer\" must be an object with key \"state\", etc.). Match key names, nesting, and value types exactly. "
    "The value of \"log_url\" must be the LOG_PUBLIC_URL value given to you below. "
    "If data was fetched from a URL for you, treat it as the authoritative source and parse it with code rather "
    "than guessing its contents. If the question embeds inline data (CSV, table, list) in the message text itself, "
    "parse that with code too. Keep the final answer concise and strictly matching the requested JSON shape."
)


def fetch_url_content(url: str) -> str:
    """Fetch a URL and return truncated text content for the model to parse."""
    try:
        resp = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0 (data-analyst-bot)"})
        resp.raise_for_status()
        content = resp.text
        if len(content) > URL_FETCH_CHAR_LIMIT:
            content = content[:URL_FETCH_CHAR_LIMIT] + "\n...[truncated]"
        return content
    except Exception as e:
        logger.warning("Failed to fetch URL %s: %s", url, e)
        return f"[Could not fetch this URL: {e}]"


def build_context_block(text: str) -> str:
    """Fetch any URLs found in the message and format them as extra context."""
    urls = URL_RE.findall(text)
    if not urls:
        return ""
    blocks = []
    for url in dict.fromkeys(urls):  # de-dupe, preserve order
        content = fetch_url_content(url)
        blocks.append(f"--- Content fetched from {url} ---\n{content}\n--- end of {url} ---")
    return "\n\n".join(blocks)


def call_gemini(system_prompt: str, user_prompt: str, max_tokens: int = 4000):
    """Calls Gemini via google-genai SDK with the code-execution tool enabled."""
    resp = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=0.0,
            max_output_tokens=max_tokens,
            tools=[types.Tool(code_execution=types.ToolCodeExecution())],
        ),
    )
    text = (resp.text or "").strip()
    return GEMINI_MODEL, text


def _strip_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*\n?", "", s)
        s = re.sub(r"```$", "", s).strip()
    return s


def _find_balanced_json_candidates(s: str):
    """Yield every top-level {...} substring in s (there can be several, e.g.
    printed dicts from executed code plus the real final answer)."""
    candidates = []
    depth = 0
    start = None
    for i, ch in enumerate(s):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    candidates.append(s[start:i + 1])
    return candidates


def extract_json_object(raw_text: str):
    """Best-effort extraction of the single required JSON object from model output.

    Gemini's code-execution tool often returns several parts (executed code,
    its printed output, narration, and the final answer) concatenated into
    one text blob. Naively grabbing the first '{' and last '}' can capture
    the wrong span. We prefer content after the ===FINAL_JSON=== sentinel;
    otherwise we scan every balanced {...} candidate and pick the one that
    actually looks like our answer object (has 'answer' and 'log_url' keys).
    """
    text = raw_text.strip()

    # 1) Prefer whatever follows our sentinel marker, if present.
    marker = "===FINAL_JSON==="
    if marker in text:
        after = text.split(marker)[-1]
        after = _strip_fences(after)
        try:
            obj = json.loads(after)
            if isinstance(obj, dict) and "answer" in obj:
                return obj
        except Exception:
            pass  # fall through to general scan

    cleaned = _strip_fences(text)

    # 2) Try a direct parse of the whole (fenced-stripped) text.
    try:
        obj = json.loads(cleaned)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # 3) Scan every balanced-brace candidate and prefer one with our required keys.
    candidates = _find_balanced_json_candidates(cleaned)
    best = None
    for cand in candidates:
        try:
            obj = json.loads(cand)
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue
        if "answer" in obj and "log_url" in obj:
            return obj
        if "answer" in obj and best is None:
            best = obj
    if best is not None:
        return best
    if candidates:
        # Last resort: return the last candidate that at least parses as an object.
        for cand in reversed(candidates):
            try:
                obj = json.loads(cand)
                if isinstance(obj, dict):
                    return obj
            except Exception:
                continue

    raise ValueError("No JSON object found in model output")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    chat_id = update.effective_chat.id if update.effective_chat else user.id
    text = update.message.text.strip() if update.message and update.message.text else ""
    logger.info("Received message from %s (%s) in chat %s: %s", user.username, user.id, chat_id, text)

    timestamp = datetime.now(timezone.utc).isoformat()

    # Pull prior messages from this chat for multi-turn context (excluding current one).
    prior_messages = list(CHAT_HISTORY[chat_id])

    fetched_context = build_context_block(text)

    system_prompt = PROMPT_INSTRUCTIONS + f"\nLOG_PUBLIC_URL={LOG_PUBLIC_URL}\n"

    prompt_parts = []
    if prior_messages:
        history_str = "\n".join(f"- {m}" for m in prior_messages)
        prompt_parts.append(
            "Earlier messages in this same conversation (for context only; answer the LAST/current "
            "message below, not these):\n" + history_str
        )
    if fetched_context:
        prompt_parts.append("Data fetched from URL(s) referenced in the current message:\n" + fetched_context)
    prompt_parts.append(
        "Current message to answer (this is the question you must respond to):\n" + text +
        "\n\nRespond with only the required JSON object."
    )
    user_prompt = "\n\n".join(prompt_parts)

    # Record this message in history for future turns in the same chat.
    CHAT_HISTORY[chat_id].append(text)

    model_response_text = None
    model_used = None
    try:
        if client is None:
            raise RuntimeError("GEMINI_API_KEY not configured")
        model_used, model_response_text = call_gemini(system_prompt, user_prompt)
        logger.info("Model reply: %s", model_response_text)
    except Exception as e:
        logger.exception("Gemini call failed: %s", e)
        fallback = {"answer": {"error": "gemini_call_failed", "message": str(e)}, "log_url": LOG_PUBLIC_URL}
        model_response_text = json.dumps(fallback, ensure_ascii=False)

    # Ensure the model output is a single JSON object
    parsed = None
    try:
        parsed = extract_json_object(model_response_text)
        if not isinstance(parsed, dict):
            raise ValueError("Top-level JSON is not an object")
    except Exception as e:
        logger.warning("Model output not valid JSON object: %s", e)
        try:
            repair_prompt = (
                "The previous model output was not a valid single JSON object. Extract or produce the exact "
                "single JSON object required (with keys 'answer' and 'log_url') and nothing else, no markdown "
                "fences. Use code execution again if you need to recompute anything precisely. "
                "End your output with the line ===FINAL_JSON=== followed immediately by only the JSON object.\n"
                "Current message was:\n" + text + "\nPrevious model output:\n" + model_response_text
            )
            _, model_response_text = call_gemini(system_prompt, repair_prompt, max_tokens=2000)
            parsed = extract_json_object(model_response_text)
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
        "chat_id": chat_id,
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

    # Upload the full local log to the GitHub Gist in the background so a slow
    # GitHub API call never delays the Telegram reply.
    def upload_log_to_gist():
        if not (GITHUB_TOKEN and GIST_ID):
            logger.warning("GITHUB_TOKEN or GIST_ID not set; skipping gist log upload")
            return
        try:
            with open(LOCAL_LOG_PATH, "r", encoding="utf-8") as f:
                full_log_content = f.read()

            gist_resp = requests.patch(
                f"https://api.github.com/gists/{GIST_ID}",
                headers={
                    "Authorization": f"Bearer {GITHUB_TOKEN}",
                    "Accept": "application/vnd.github+json",
                },
                json={"files": {GIST_FILENAME: {"content": full_log_content}}},
                timeout=15,
            )
            logger.info("Gist update status=%s", gist_resp.status_code)
            if gist_resp.status_code >= 400:
                logger.error("Gist update failed: %s", gist_resp.text)
        except Exception:
            logger.exception("Log upload to gist failed")

    threading.Thread(target=upload_log_to_gist, daemon=True).start()

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
