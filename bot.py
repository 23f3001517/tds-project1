import json
import os
import sys
import io
import threading
from fastapi import FastAPI
from fastapi.responses import FileResponse
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, MessageHandler, ContextTypes, filters
from openai import OpenAI

# Load environment variables
load_dotenv()

BOT_TOKEN = os.getenv("BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("BASE_URL")

# Initialize OpenAI Client
client = OpenAI(api_key=OPENAI_API_KEY)

app = FastAPI()

# Ensure logs directory exists
os.makedirs("logs", exist_ok=True)
LOG_FILE = "logs/run.jsonl"

# In-memory memory for multi-turn conversations
chat_histories = {}

def log_step(data: dict):
    """Appends a single JSON object to the JSONL log file."""
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(data) + "\n")

def execute_python_code(code: str) -> str:
    """Executes Python code safely-ish and returns the printed output."""
    old_stdout = sys.stdout
    redirected_output = sys.stdout = io.StringIO()
    try:
        # Provide common data science libraries to the LLM's execution environment
        exec_globals = {
            "pd": __import__("pandas"), 
            "np": __import__("numpy"), 
            "requests": __import__("requests")
        }
        exec(code, exec_globals)
        output = redirected_output.getvalue()
        if not output.strip():
            output = "Code executed successfully, but nothing was printed. Use print() to output results."
    except Exception as e:
        output = f"Error during execution: {e}"
    finally:
        sys.stdout = old_stdout
    return output

# Define the tool for OpenAI
tools = [
    {
        "type": "function",
        "function": {
            "name": "execute_python_code",
            "description": "Run Python code to download datasets (CSV/JSON), analyze data, and perform calculations. You have access to pandas (pd), numpy (np), and requests. Always use print() to output the result so you can read it.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "The Python code to execute."
                    }
                },
                "required": ["code"]
            }
        }
    }
]

@app.get("/")
def home():
    return {"message": "Telegram Data Analyst Agent Running"}

@app.get("/run.jsonl")
def logs():
    return FileResponse(LOG_FILE)

async def reply(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.message.chat_id
    user_message = update.message.text
    
    log_step({"event": "received_message", "chat_id": chat_id, "message": user_message})

    # Initialize chat history for multi-turn grader tasks
    if chat_id not in chat_histories:
        chat_histories[chat_id] = [
            {
                "role": "system", 
                "content": (
                    "You are an expert Data-Analyst Agent. You can run Python code to analyze public datasets. "
                    "When asked a question, write and execute Python code to find the exact answer. "
                    "The user will specify a JSON shape for the final answer. "
                    "Your final reply MUST be ONLY a valid JSON object. Do NOT wrap it in markdown block quotes (like ```json). "
                    "Ensure your JSON has an 'answer' key that matches the requested shape."
                )
            }
        ]
        
    chat_histories[chat_id].append({"role": "user", "content": user_message})
    
    messages = chat_histories[chat_id].copy()

    try:
        # Agentic Loop: Allow the LLM to call tools multiple times if needed
        while True:
            response = client.chat.completions.create(
                model="gpt-4o", # Upgraded to gpt-4o for better coding capabilities
                messages=messages,
                tools=tools,
                temperature=0,
            )
            
            message = response.choices[0].message
            messages.append(message)
            
            # If the LLM decides to run Python code
            if message.tool_calls:
                for tool_call in message.tool_calls:
                    if tool_call.function.name == "execute_python_code":
                        args = json.loads(tool_call.function.arguments)
                        code = args.get("code")
                        
                        log_step({"event": "tool_call", "code_executed": code})
                        
                        # Execute the code locally
                        result = execute_python_code(code)
                        
                        log_step({"event": "tool_result", "output": result})
                        
                        # Feed the output back to the LLM
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": tool_call.function.name,
                            "content": result
                        })
                # Continue the while loop so the LLM can evaluate the code output
            else:
                # The LLM has reached its final answer and did not call a tool
                break
                
        llm_content = message.content.strip()
        log_step({"event": "final_llm_output", "content": llm_content})

        # Parse output to enforce grading JSON structure
        if llm_content.startswith("```json"):
            llm_content = llm_content[7:-3].strip()
        elif llm_content.startswith("```"):
            llm_content = llm_content[3:-3].strip()
            
        try:
            response_json = json.loads(llm_content)
            answer_payload = response_json.get("answer", response_json)

            final_reply = {
                "answer": answer_payload,
                "log_url": f"{BASE_URL.rstrip('/')}/run.jsonl"
            }
        except json.JSONDecodeError:
            final_reply = {
                "answer": {"error": "Failed to parse final JSON", "raw": llm_content},
                "log_url": f"{BASE_URL.rstrip('/')}/run.jsonl"
            }

        # Save assistant's final response to history for multi-turn
        chat_histories[chat_id].append({"role": "assistant", "content": json.dumps(final_reply)})

        final_reply_str = json.dumps(final_reply)
        log_step({"event": "sent_reply", "payload": final_reply})

        await update.message.reply_text(final_reply_str)
        
    except Exception as e:
        error_msg = str(e)
        log_step({"event": "system_error", "error": error_msg})
        await update.message.reply_text(json.dumps({
            "answer": {"error": "Internal Agent Error"},
            "log_url": f"{BASE_URL.rstrip('/')}/run.jsonl"
        }))

def telegram_bot():
    telegram_app = Application.builder().token(BOT_TOKEN).build()
    telegram_app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, reply))
    print("Telegram Agent Started...")
    telegram_app.run_polling()

@app.on_event("startup")
async def startup():
    # Clear old logs on startup
    if os.path.exists(LOG_FILE):
        open(LOG_FILE, 'w').close() 
    threading.Thread(target=telegram_bot, daemon=True).start()