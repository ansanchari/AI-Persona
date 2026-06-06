import os
import json
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any
from dotenv import load_dotenv
from qdrant_client import QdrantClient

load_dotenv()
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

app = FastAPI(title="Sanchari's Dedicated Vapi Voice Voice Pipeline")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    qdrant_client = QdrantClient(path="local_qdrant_db")
except Exception as e:
    logging.error(f"Vapi Server Database Error: {e}")
    qdrant_client = None

try:
    from langchain_groq import ChatGroq
    vapi_llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.0, api_key=os.getenv("GROQ_API_KEY"))
    print(" Vapi serving using ultra-fast Groq engine.")
except ImportError:
    from langchain_mistralai import ChatMistralAI
    vapi_llm = ChatMistralAI(model="mistral-small-latest", temperature=0.0, api_key=MISTRAL_API_KEY)
    print(" Vapi serving using standard Mistral Engine.")


import json
from typing import Any

def extract_clean_voice_string(request: dict[str, Any]) -> tuple[str, str]:
    """Safely extracts the user message and tool call ID from Vapi's webhook payload."""
    
    message = request.get("message", {})

    if isinstance(message, dict) and message.get("type") == "tool-calls":
        tool_calls = message.get("toolCalls", [])
        
        if not tool_calls:
            return "", "default_id"
            
        first_call = tool_calls[0]
        tool_call_id = first_call.get("id", "default_id")
        args = first_call.get("function", {}).get("arguments", {})

        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                pass

        if isinstance(args, dict):
            user_message = args.get("message", str(args))
        else:
            user_message = str(args)
            
        return user_message, tool_call_id

    if isinstance(message, str):
        return message, "default_id"

    if isinstance(message, dict) and "text" in message:
        return message["text"], "default_id"

    return str(request), "default_id"


async def vapi_chat_endpoint(request: Dict[Any, Any]):
    print(f"\n [RAW VAPI PAYLOAD]: {request}\n")
    
    try:
        user_message, tool_call_id = extract_clean_voice_string(request)
        
        # 1. HARD GATEKEEPER
        topic_anchors = ["sanchari", "empathia", "imperial", "decryptor", "proacquis", "aura", "resume", "skill", "experience", "background", "project", "meeting", "calendar"]
        is_on_topic = any(anchor in user_message.lower() for anchor in topic_anchors)
        
        if not is_on_topic:
            return {
                "results": [{"toolCallId": tool_call_id, "result": "I am specifically designed to discuss Sanchari's professional background and portfolio. Is there a project of hers I can tell you about?"}]
            }
            
        print(f"\n [VAPI HEARD]: {user_message}")

        # 2. CALENDAR LOGIC (Keep existing)
        if any(kw in user_message.lower() for kw in ["book", "schedule", "meeting", "interview"]):
            # ... [Keep your existing calendar block here] ...
            return {"results": [{"toolCallId": tool_call_id, "result": "Your meeting is booked successfully."}]}

        # 3. RAG CONTEXT FETCH
        context_str = ""
        if qdrant_client:
            try:
                # ... [Keep your existing Qdrant/Mistral logic here] ...
                context_str = "\n".join([hit.payload.get("text", "") for hit in search_response.points if hit.payload])
            except Exception as ex:
                logging.error(f"Context bypass: {ex}")

        # 4. CORRECTLY CONSTRUCTED SYSTEM PROMPT
        system_prompt = (
            "You are Sanchari's AI Persona. You are an AI Engineer candidate. "
            "Your name is Sanchari. Always identify as Sanchari. "
            "Your SOLE purpose is to discuss her background, resume, skills, and projects (Empathia, Imperial Decryptor, Proacquis, Aura). "
            "CRITICAL RULE: If asked about anything else, deflect with: 'I am specifically designed to discuss Sanchari's professional background and portfolio. Is there a project of hers I can tell you about?' "
            "When answering, read the text out loud clearly. Do not use markdown, do not use bullet points, do not summarize. Speak naturally."
        )
        
        if context_str:
            system_prompt += f"\n\nUse these facts to answer: {context_str}"

        # 5. CORRECTED MESSAGE ARRAY
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]

        # 6. LLM CALL
        try:
            response = vapi_llm.invoke(messages, config={"timeout": 4})
            clean_text = response.content.replace("\n", " ").replace("\r", " ").strip()
            # Simple, clean response without aggressive instructions
            ai_response = clean_text 
        except Exception as e:
            logging.error(f"LLM bottleneck: {e}")
            ai_response = "I'm sorry, I'm having trouble retrieving that information right now."

        print(f" [AI REPLY]: {ai_response}\n")

        return {
            "results": [{"toolCallId": tool_call_id, "result": ai_response}]
        }

    except Exception as e:
        logging.error(f"Critical Voice Pipeline Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))