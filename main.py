# main.py
# AutoStream Conversational Agent (Gemini 2.5 Flash)
# Run: python main.py
# Requires: .env with GEMINI_API_KEY and rag_knowledge.json in same folder

import os
import json
import re
from collections import deque
from dotenv import load_dotenv

# Try to import Gemini client via LangChain wrapper; fail gracefully if unavailable
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    LCG_AVAILABLE = True
except Exception:
    LCG_AVAILABLE = False

# ---------------- env & llm ----------------
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

llm = None
if LCG_AVAILABLE and GEMINI_API_KEY:
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.0,
            google_api_key=GEMINI_API_KEY
        )
    except Exception:
        llm = None

# ---------------- load RAG ----------------
KB_PATH = "rag_knowledge.json"
if not os.path.exists(KB_PATH):
    raise FileNotFoundError(f"{KB_PATH} not found. Add the local JSON knowledge base.")

with open(KB_PATH, "r") as f:
    KB = json.load(f)

# ---------------- mock tool ----------------
def mock_lead_capture(name: str, email: str, platform: str):
    # This is the required mock tool. It must be called exactly once after collecting all fields.
    print(f"\n✅ Lead captured successfully: {name}, {email}, {platform}\n")

# ---------------- agent state ----------------
class AgentState:
    def __init__(self, history_turns: int = 6):
        self.memory = deque(maxlen=history_turns)   # store last turns if needed
        self.lead_data = {}                          # collects 'name','email','platform'
        self.awaiting_field = None                   # None or one of 'name','email','platform'

    def save_turn(self, user_text: str, agent_text: str):
        self.memory.append({"user": user_text, "agent": agent_text})

state = AgentState()

# ---------------- intent detection (HIGH-INTENT FIRST) ----------------
def detect_intent(text: str) -> str:
    t = text.lower()
    # high-intent patterns (priority)
    high_intent_tokens = [
        "sign up", "signup", "i want to sign", "i want to try", "i want to sign up",
        "i'll take", "i will take", "i want to subscribe", "subscribe", "buy", "get started",
        "get pro", "start trial", "pro plan", "i want the pro", "i want pro"
    ]
    if any(tok in t for tok in high_intent_tokens):
        return "high_intent"

    # pricing / product inquiry
    pricing_tokens = ["price", "pricing", "cost", "plan", "features", "basic plan", "pro plan"]
    if any(tok in t for tok in pricing_tokens):
        return "pricing"

    # greeting
    if any(g in t for g in ["hi", "hello", "hey", "good morning", "good evening"]):
        return "greeting"

    return "other"

# ---------------- simple RAG retrieval ----------------
def rag_pricing_answer() -> str:
    basic = KB.get("pricing", {}).get("basic", {})
    pro = KB.get("pricing", {}).get("pro", {})
    basic_text = f"Basic Plan: {basic.get('price')} , {basic.get('videos')} , {basic.get('resolution')}"
    pro_text = f"Pro Plan: {pro.get('price')} , {pro.get('videos')} , {pro.get('resolution')}"
    features = pro.get("features", [])
    if features:
        pro_text += f" (Features: {', '.join(features)})"
    return f"{basic_text}\n{pro_text}"

def rag_policy_answer() -> str:
    pol = KB.get("policies", {})
    return f"Refund policy: {pol.get('refund')}. Support: {pol.get('support')}."

# ---------------- small validators ----------------
EMAIL_RE = re.compile(r"[^@]+@[^@]+\.[^@]+")

def is_valid_email(s: str) -> bool:
    return bool(EMAIL_RE.fullmatch(s.strip()))

def choose_next_field(collected: dict):
    for f in ("name", "email", "platform"):
        if f not in collected:
            return f
    return None

# ---------------- LLM fallback ----------------
def llm_reply(prompt: str) -> str:
    if not llm:
        return "I can help with pricing or sign-up. Ask 'Tell me about pricing' or 'I want to sign up for Pro'."
    try:
        # safe call - depending on wrapper this might return object or string; handle both
        resp = llm(prompt)
        if isinstance(resp, str):
            return resp
        # langchain-like response object
        return getattr(resp, "content", str(resp))
    except Exception:
        return "Sorry — temporary LLM error. I can still help with pricing or sign-up."

# ---------------- main handler ----------------
def handle_user_input(user_text: str):
    user_text = user_text.strip()
    # If we're currently collecting lead details, prioritize that flow
    if state.awaiting_field:
        field = state.awaiting_field

        # Validate & store
        if field == "email":
            if not is_valid_email(user_text):
                agent_text = "That doesn't look like a valid email. Please provide a valid email address."
                state.save_turn(user_text, agent_text)
                print(agent_text)
                return
            state.lead_data["email"] = user_text.strip()
        else:
            state.lead_data[field] = user_text.strip()

        # Choose next
        next_field = choose_next_field(state.lead_data)
        if next_field:
            state.awaiting_field = next_field
            prompt_map = {
                "name": "Great — what's your full name?",
                "email": "Thanks. What's your email?",
                "platform": "Which creator platform do you use? (YouTube, Instagram, etc.)"
            }
            agent_text = prompt_map[next_field]
            state.save_turn(user_text, agent_text)
            print(agent_text)
            return

        # All fields collected -> call tool exactly once
        name = state.lead_data.get("name")
        email = state.lead_data.get("email")
        platform = state.lead_data.get("platform")
        # Clear awaiting flag BEFORE calling tool to avoid re-entrancy
        state.awaiting_field = None
        # Acknowledge then call tool
        agent_text = f"Thanks {name}! Submitting your details now..."
        state.save_turn(user_text, agent_text)
        print(agent_text)
        mock_lead_capture(name, email, platform)
        final_ack = "Done — our team will reach out soon. Would you like a setup guide?"
        state.save_turn("", final_ack)
        print(final_ack)
        state.lead_data = {}
        return

    # No active lead flow -> detect intent
    intent = detect_intent(user_text)

    if intent == "greeting":
        agent_text = "👋 Hi! I can tell you about pricing or help you sign up for Pro."
        state.save_turn(user_text, agent_text)
        print(agent_text)
        return

    if intent == "pricing":
        # If user explicitly mentions refund/support, give policy
        if any(tok in user_text.lower() for tok in ["refund", "support", "policy"]):
            agent_text = rag_policy_answer()
        else:
            agent_text = rag_pricing_answer()
        state.save_turn(user_text, agent_text)
        print(agent_text)
        return

    if intent == "high_intent":
        # Start qualification flow
        state.lead_data = {}
        state.awaiting_field = "name"
        agent_text = "Awesome — I can help get you started. What's your full name?"
        state.save_turn(user_text, agent_text)
        print(agent_text)
        return

    # fallback: try LLM for freeform replies
    agent_text = llm_reply(user_text)
    state.save_turn(user_text, agent_text)
    print(agent_text)

# ---------------- CLI demo ----------------
def main():
    print("🚀 AutoStream AI Agent (Gemini 2.5 Flash) — type 'exit' to quit")
    while True:
        try:
            user = input("User: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break
        if not user:
            continue
        if user.lower() in ("exit", "quit"):
            print("Bye.")
            break
        handle_user_input(user)

if __name__ == "__main__":
    main()
