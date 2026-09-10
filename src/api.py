# region: Setup and Imports
# Combines every component built so far (classifier, priority rules, RAG) into one API.

from pathlib import Path
import sys
from contextlib import asynccontextmanager
from datetime import datetime
import joblib
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT / "src"))

from business_decision import assign_priority
from response_generation import retrieve_similar_responses, generate_draft_response
from database import SessionLocal, Ticket, init_db

OUTPUTS_PATH = Path(r"C:\Projects\SupportTicketOutputs")
MODEL_DIR = OUTPUTS_PATH / "distilbert_intent_model"
# endregion


# region: Model Loading (runs once, at startup)
# Loaded into a shared dict so every request reuses the same in-memory models.

ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading models, please wait...")
    ml_models["tokenizer"] = AutoTokenizer.from_pretrained(MODEL_DIR)
    ml_models["classifier"] = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    ml_models["label_encoder"] = joblib.load(OUTPUTS_PATH / "label_encoder.joblib")
    init_db()
    print("Models loaded. API ready.")
    yield
    ml_models.clear()

app = FastAPI(title="Customer Support Ticket Intelligence API", lifespan=lifespan)
# endregion


# region: Request/Response Schemas
# Pydantic models define and validate the expected input/output shapes.

class TicketRequest(BaseModel):
    instruction: str
    category: str = "UNKNOWN"

class TicketResponse(BaseModel):
    id: int
    instruction: str
    intent: str
    priority: str
    draft_response: str
    created_at: datetime
# endregion


# region: Core Prediction Logic
# Runs the full pipeline: classify -> prioritize -> retrieve+generate -> save.

def classify_intent(text: str) -> str:
    inputs = ml_models["tokenizer"](text, return_tensors="pt", truncation=True, padding="max_length", max_length=32)
    with torch.no_grad():
        outputs = ml_models["classifier"](**inputs)
    predicted_id = torch.argmax(outputs.logits, dim=1).item()
    return ml_models["label_encoder"].inverse_transform([predicted_id])[0]


def process_ticket(instruction: str, category: str) -> dict:
    intent = classify_intent(instruction)

    flags = ""  # new live tickets have no pre-existing 'flags' field, only historical data does
    priority = assign_priority(intent, flags)

    retrieved_examples = retrieve_similar_responses(instruction)
    draft_response = generate_draft_response(instruction, retrieved_examples)

    return {"intent": intent, "priority": priority, "draft_response": draft_response}
# endregion


# region: Endpoints

@app.post("/tickets", response_model=TicketResponse)
def create_ticket(request: TicketRequest):
    result = process_ticket(request.instruction, request.category)

    session = SessionLocal()
    try:
        ticket = Ticket(
            instruction=request.instruction,
            category=request.category,
            intent=result["intent"],
            priority=result["priority"],
            draft_response=result["draft_response"],
        )
        session.add(ticket)
        session.commit()
        session.refresh(ticket)
        return ticket
    finally:
        session.close()


@app.get("/tickets/{ticket_id}", response_model=TicketResponse)
def get_ticket(ticket_id: int):
    session = SessionLocal()
    try:
        return session.get(Ticket, ticket_id)
    finally:
        session.close()


@app.get("/health")
def health_check():
    return {"status": "ok"}
# endregion