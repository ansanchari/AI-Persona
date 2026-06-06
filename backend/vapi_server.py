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


def extract_clean_voice_string(request: Dict[Any, Any]) -> tuple[str, str]:
    user_message = ""
    tool_call_id = "default_id"
    
    if request.get("message", {}).get("type") == "tool-calls":
        tool_calls = request["message"].get("toolCalls", [])
        if tool_calls:
            tool_call_id = tool_calls[0].get("id", "default_id")
            arguments = tool_calls[0].get("function", {}).get("arguments", {})
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    pass
            if isinstance(arguments, dict):
                user_message = arguments.get("message", str(arguments))
            else:
                user_message = str(arguments)
    elif "message" in request and isinstance(request["message"], str):
        user_message = request["message"]
    elif "message" in request and "text" in request["message"]:
        user_message = request["message"]["text"]
    else:
        user_message = str(request)
        
    return user_message, tool_call_id


@app.post("/vapi-chat")
async def vapi_chat_endpoint(request: Dict[Any, Any]):
    try:
        user_message, tool_call_id = extract_clean_voice_string(request)

        print(f"\n [VAPI HEARD]: {user_message}")
        
        if any(kw in user_message.lower() for kw in ["book", "schedule", "meeting", "interview"]):
            import os, datetime
            
            ai_response = "SPEECH_PAYLOAD: Your meeting is booked successfully."
            
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
            "You are the conversational voice persona of Sanchari, an AI Engineer candidate. "
            "Speak clearly, directly, and complete your answer in 1 to 3 short sentences. "
            "Never use markdown symbols, bold text asterisks, or bulleted lists as your output is read aloud."
        )
        if context_str:
            system_prompt += f"\n\nAnswer using these facts:\n{context_str}"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]

        try:
            response = vapi_llm.invoke(messages, config={"timeout": 4})
            clean_text = response.content.replace("\n", " ").replace("\r", " ").strip()
            ai_response = f"SPEECH_PAYLOAD: {clean_text}"
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