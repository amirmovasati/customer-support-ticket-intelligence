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
import time
import logging
from fastapi import Request

HF_MODEL_REPO = "amirmovasati/support-ticket-distilbert-intent"
# endregion


# region: Model Loading (runs once, at startup)
# Loaded into a shared dict so every request reuses the same in-memory models.

ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading models, please wait...")
    from huggingface_hub import hf_hub_download

    ml_models["tokenizer"] = AutoTokenizer.from_pretrained(HF_MODEL_REPO)
    ml_models["classifier"] = AutoModelForSequenceClassification.from_pretrained(HF_MODEL_REPO)
    label_encoder_path = hf_hub_download(repo_id=HF_MODEL_REPO, filename="label_encoder.joblib")
    ml_models["label_encoder"] = joblib.load(label_encoder_path)
    init_db()
    print("Models loaded. API ready.")
    yield
    ml_models.clear()

app = FastAPI(title="Customer Support Ticket Intelligence API", lifespan=lifespan)
# endregion

# region: Monitoring — Request Logging
# Logs every request's duration and status to a file, and tracks basic
# in-memory counters for the /metrics endpoint.

logging.basicConfig(
    filename=str(PROJECT_ROOT / "outputs" / "api_requests.log"),
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
)

request_stats = {"total_requests": 0, "total_errors": 0, "total_response_time": 0.0}

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    try:
        response = await call_next(request)
    except Exception as e:
        request_stats["total_errors"] += 1
        logging.error(f"{request.method} {request.url.path} - FAILED - {e}")
        raise

    duration = time.time() - start_time
    request_stats["total_requests"] += 1
    request_stats["total_response_time"] += duration
    if response.status_code >= 400:
        request_stats["total_errors"] += 1

    logging.info(f"{request.method} {request.url.path} - {response.status_code} - {duration:.3f}s")
    return response
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

@app.get("/metrics")
def get_metrics():
    avg_response_time = (
        request_stats["total_response_time"] / request_stats["total_requests"]
        if request_stats["total_requests"] > 0
        else 0
    )
    session = SessionLocal()
    try:
        total_tickets = session.query(Ticket).count()
        priority_counts = {
            p: session.query(Ticket).filter(Ticket.priority == p).count()
            for p in ["High", "Medium", "Low"]
        }
    finally:
        session.close()

    return {
        "total_requests": request_stats["total_requests"],
        "total_errors": request_stats["total_errors"],
        "avg_response_time_seconds": round(avg_response_time, 3),
        "total_tickets_in_db": total_tickets,
        "tickets_by_priority": priority_counts,
    }

# endregion