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


@app.post("/vapi-chat")
async def vapi_chat_endpoint(request: Dict[Any, Any]):
    print(f"\n👀 [RAW VAPI PAYLOAD]: {request}\n")
    try:
        user_message, tool_call_id = extract_clean_voice_string(request)

        topic_anchors = ["sanchari", "empathia", "imperial", "decryptor", "proacquis", "aura", "resume", "skill", "experience", "background", "project", "meeting", "calendar"]
        is_on_topic = any(anchor in user_message.lower() for anchor in topic_anchors)
    
        if not is_on_topic:
            return {
                "results": [{"toolCallId": tool_call_id, "result": "I am specifically designed to discuss Sanchari's professional background and portfolio. Is there a project of hers I can tell you about?"}]
            }
        print(f"\n [VAPI HEARD]: {user_message}")
        
        if any(kw in user_message.lower() for kw in ["book", "schedule", "meeting", "interview"]):
            import os, datetime
            
            ai_response = f"Tool operation successful. PLEASE JUST READ THE TEXT OUT ALOUD to the user exactly as written, and do not use any more tools for this turn: Your meeting is booked successfully."
            
            try:
                from google.oauth2.credentials import Credentials
                from googleapiclient.discovery import build
                
                if os.path.exists('token.json'):
                    creds = Credentials.from_authorized_user_file('token.json')
                    service = build('calendar', 'v3', credentials=creds)
                    
                    tomorrow = datetime.datetime.now() + datetime.timedelta(days=1)
                    start_time = tomorrow.replace(hour=10, minute=0, second=0, microsecond=0).isoformat()
                    end_time = tomorrow.replace(hour=11, minute=0, second=0, microsecond=0).isoformat()
                    
                    event = {
                        'summary': "Screening Interview with Sanchari",
                        'start': {'dateTime': start_time, 'timeZone': 'Asia/Kolkata'},
                        'end': {'dateTime': end_time, 'timeZone': 'Asia/Kolkata'}
                    }
                    service.events().insert(calendarId='primary', body=event).execute()
                    print(" [CALENDAR]: Google Calendar Event Created!")
                else:
                    print(" [CALENDAR]: Skipped API call. 'token.json' not found in directory.")
            
            except Exception as e:
                print(f" [CALENDAR ERROR]: {e}")
            
            return {"results": [{"toolCallId": tool_call_id, "result": ai_response}]}

        context_str = ""
        if qdrant_client:
            try:
                import requests
                emb_res = requests.post(
                    "https://api.mistral.ai/v1/embeddings",
                    headers={"Authorization": f"Bearer {MISTRAL_API_KEY}"},
                    json={"model": "mistral-embed", "input": [user_message]},
                    timeout=3
                )
                query_vector = emb_res.json()["data"][0]["embedding"]
                
                search_response = qdrant_client.query_points(
                    collection_name="my_persona_data",
                    query=query_vector,
                    limit=6
                )
                context_str = "\n".join([hit.payload.get("text", "") for hit in search_response.points if hit.payload])
            except Exception as ex:
                logging.error(f"Fallback: context bypass due to velocity limits: {ex}")

        system_prompt = (
            "You are an AI representative for Sanchari. Your SOLE purpose is to discuss her professional background, resume, skills, and portfolio projects (like Empathia, Imperial Decryptor, and Proacquis)."
            "CRITICAL RULE: If the user asks ANY question that is not directly related to Sanchari, her qualifications, or scheduling a meeting with her, you are STRICTLY FORBIDDEN from answering it. You must gracefully deflect by replying EXACTLY with: 'I am specifically designed to discuss Sanchari's professional background and portfolio. Is there a project of hers I can tell you about?"
            "CONVERSATION EXAMPLES:"

            "User: How do I write a Python script to scrape a website?"
            "Assistant: I am specifically designed to discuss Sanchari's professional background and portfolio. Is there a project of hers I can tell you about?"

            "User: What was the focus of the project submission?"
            "Assistant: The Scalability & Optimization narrative was chosen for the project submission to highlight how the architecture handles growth and system weaknesses."

            "User: What is the capital of France?"
            "Assistant: I am specifically designed to discuss Sanchari's professional background and portfolio. Is there a project of hers I can tell you about?"

            "User: Did you publish the final version of the project?"
            "Assistant: We successfully built the project architecture, but the publishing portion was not completed in reality."

            "User: Help me debug this Next.js error."
            "Assistant: I am specifically designed to discuss Sanchari's professional background and portfolio. Is there a project of hers I can tell you about?"
        )
        if context_str:
            system_prompt += f"\n\nAnswer using these facts:\n{context_str}"

        messages = [
            {"role": "system", "content": "You are Sanchari's AI Persona. You strictly discuss Sanchari's background... [rest of your prompt]"},
            {"role": "user", "content": user_message}
        ]

        try:
            response = vapi_llm.invoke(messages, config={"timeout": 4})
            clean_text = response.content.replace("\n", " ").replace("\r", " ").strip()
            ai_response = f"Tool operation successful. PLEASE JUST READ THE TEXT OUT ALOUD to the user exactly as written, and do not use any more tools for this turn, do not add anything more, do not even try to think, just read out the response out aloud: {clean_text}"
        except Exception as e:
            logging.error(f"LLM bottleneck fallback hit: {e}")
            ai_response = "I caught that, but my data stream timed out. Could you try rephrasing your question?"

        print(f" [AI REPLY]: {ai_response}\n")

        return {
            "results": [
                {
                    "toolCallId": tool_call_id,
                    "result": ai_response
                }
            ]
        }

    except Exception as e:
        logging.error(f"Critical Voice Pipeline Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))