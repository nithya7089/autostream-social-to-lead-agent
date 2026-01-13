# AutoStream – Social-to-Lead Agentic Workflow

## Overview
This project implements a conversational AI agent for a fictional SaaS product called **AutoStream**, which provides automated video editing tools for content creators. The agent understands user intent, answers product questions using a local RAG knowledge base, detects high-intent leads, and triggers a backend lead-capture tool only after collecting all required user details.

---

## Features
- Intent classification: greeting, pricing inquiry, high-intent lead  
- RAG-based responses using a local JSON knowledge base  
- Multi-turn state management across 5–6 conversation turns  
- Safe tool execution for lead capture (mock API)  
- Powered by **Gemini 2.5 Flash** using LangChain  

---

## Tech Stack
- **Language:** Python 3.9+  
- **LLM:** Gemini 2.5 Flash  
- **Framework:** LangChain  
- **State Management:** In-memory agent state  
- **RAG Store:** Local JSON file  

---

## How to Run Locally

### 1. Clone and set up environment
```bash
git clone https://github.com/nithya7089/autostream-social-to-lead-agent
cd autostream-social-to-lead-agent
python3 -m venv venv
source venv/bin/activate

###2. Install dependencies
pip install -r requirements.txt

###3. Add environment variables
Create a .env file:

env

GEMINI_API_KEY=your_gemini_api_key_here

4. Run the agent
bash

python main.py
Sample Conversation Flow

User: Hi
User: Tell me about your pricing
User: I want to sign up for the Pro plan
User: Nithya
User: nithya@gmail.com
User: YouTube
Lead captured successfully: Nithya, nithya@gmail.com, YouTube
