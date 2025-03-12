import time
from datetime import datetime
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field, validator
from typing import List, Tuple, Optional
import os

from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI
from sentence_transformers import SentenceTransformer

from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text
from sqlalchemy.orm import sessionmaker, declarative_base, Session

# ----------------------------------
# Database Setup using SQLAlchemy
# ----------------------------------
DATABASE_URL = "sqlite:///./chat_history.db"  # SQLite file-based database

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()

class ChatMessage(Base):
    __tablename__ = "chat_messages"
    id = Column(Integer, primary_key=True, index=True)
    chat_id = Column(String, index=True, nullable=False)
    speaker = Column(String, nullable=False)  # "User" or "Assistant"
    message = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

Base.metadata.create_all(bind=engine)

# Dependency to get a DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key="OPENAI_API_KEY")

# PINECONE_API_KEY = "pcsk_44Uird_RSvvs6T6jZwYnmUF3ySv6PshNLY2zfvJEtTjY9nHwTzbWtaCqp6dYaJt3ieK4Jm"
# PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
# pc = Pinecone(api_key=PINECONE_API_KEY)


INDEX_NAME = "aban-rag-ft2-faq-v1-1"



# Wait until the index is ready.
while not pc.describe_index(INDEX_NAME).status.get('ready', False):
    time.sleep(1)
index = pc.Index(INDEX_NAME)

# Initialize the SentenceTransformer with your fine-tuned model.
ft_model = SentenceTransformer("/home/shahriar/Work/AbanTether/raggpt/pincone_RAG/embedded_fintune/my_finetuned_faq_model_faq_2_pair")
print("Fine-tuned model dimension:", ft_model.get_sentence_embedding_dimension())


def get_embedding(text: str) -> list:
    """Generates an embedding for the given text using the fine-tuned model."""
    emb = ft_model.encode(text)
    return emb.tolist()

def query_index(query: str) -> str:
    """
    Queries the Pinecone index with the query's embedding and returns
    concatenated context from the top matching chunks.
    """
    query_emb = get_embedding(query)
    results = index.query(
        namespace="faq-namespace",
        vector=query_emb,
        top_k=3,
        include_values=False,
        include_metadata=True
    )
    print("Query Results:", results)
    retrieved_texts = [match["metadata"]["text"] for match in results.get("matches", [])]
    return "\n\n".join(retrieved_texts)

def generate_response(query: str, context: str, conversation_history: str) -> str:
    """
    Builds a prompt with the retrieved context and generates a conversational answer
    using GPT-4.5-preview.
    """
    prompt = (
        "You are a friendly, helpful, and natural conversational AI assistant. Work in exchange company named Abanprime and in persian is آبان پرایم. "
        "You should answer questions in a supportive way based on the following context extracted from your indexed documents. "
        "Engage in a natural conversation, and if the user mentions topics such as buying Tether (USDT), "
        "ask follow-up clarifying questions like 'How much USDT would you like to buy?' and "
        "'How will you pay? In which currency?'.\n\n"
        "Context from the vector database:\n"
        f"{context}\n\n"
        "Conversation so far:\n"
        f"{conversation_history}\n"
        f"User: {query}\n"
        "Assistant:"
    )

    chat_response = client.chat.completions.create(
        model="gpt-4.5-preview",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4
    )
    return chat_response.choices[0].message.content

def load_conversation_history(chat_id: str, db: Session) -> List[Tuple[str, str]]:
    """Retrieve conversation history for a given chat_id from the database."""
    messages = db.query(ChatMessage).filter(ChatMessage.chat_id == chat_id).order_by(ChatMessage.created_at).all()
    return [(msg.speaker, msg.message) for msg in messages]

def save_message(chat_id: str, speaker: str, message: str, db: Session) -> None:
    """Save a chat message to the database."""
    new_msg = ChatMessage(chat_id=chat_id, speaker=speaker, message=message)
    db.add(new_msg)
    db.commit()

# ----------------------------------
# FastAPI App Definition
# ----------------------------------

app = FastAPI()

# -----------------------------
# Pydantic Models with Validation
# -----------------------------
class WhatsAppMessage(BaseModel):
    chat_id: str = Field(..., min_length=1, description="Unique identifier for the chat session (e.g., mobile number or user ID)")
    message: str = Field(..., min_length=1, description="The incoming message text")

    @validator("chat_id")
    def chat_id_not_empty(cls, v):
        if not v.strip():
            raise ValueError("chat_id must not be empty or whitespace")
        return v

    @validator("message")
    def message_not_empty(cls, v):
        if not v.strip():
            raise ValueError("message must not be empty or whitespace")
        return v

class WhatsAppResponse(BaseModel):
    chat_id: str
    message: str
    conversation_history: List[Tuple[str, str]]  # Full conversation history

# -----------------------------
# WhatsApp Webhook Endpoint with Database Integration
# -----------------------------
@app.post("/whatsapp", response_model=WhatsAppResponse)
async def whatsapp_webhook(whatsapp_msg: WhatsAppMessage, db: Session = Depends(get_db)):
    try:
        # Load conversation history from the database.
        conversation_history = load_conversation_history(whatsapp_msg.chat_id, db)
        conversation_history_str = "\n".join([f"{speaker}: {msg}" for speaker, msg in conversation_history])
        
        # Query your vector index for context based on the incoming message.
        context = query_index(whatsapp_msg.message)
        # Generate the chatbot's response.
        answer = generate_response(whatsapp_msg.message, context, conversation_history_str)
        
        # Save the new messages to the database.
        save_message(whatsapp_msg.chat_id, "User", whatsapp_msg.message, db)
        save_message(whatsapp_msg.chat_id, "Assistant", answer, db)
        
        # Reload conversation history to return full updated history.
        updated_history = load_conversation_history(whatsapp_msg.chat_id, db)
        
        return WhatsAppResponse(
            chat_id=whatsapp_msg.chat_id,
            message=answer,
            conversation_history=updated_history
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# -----------------------------
# Standard Chat Endpoint for Testing (Using In-Memory History)
# -----------------------------
class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, description="The query text for the chat")
    conversation_history: List[Tuple[str, str]] = Field(default_factory=list, description="List of tuples representing the conversation history")

    @validator("query")
    def query_not_empty(cls, v):
        if not v.strip():
            raise ValueError("query must not be empty or whitespace")
        return v

class ChatResponse(BaseModel):
    response: str

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        context = query_index(request.query)
        conversation_history_str = "\n".join([f"{speaker}: {message}" for speaker, message in request.conversation_history])
        answer = generate_response(request.query, context, conversation_history_str)
        return ChatResponse(response=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ----------------------------------
# Main Entry Point
# ----------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
