# region: Setup and Imports
from pathlib import Path
import sys
from contextlib import asynccontextmanager
from datetime import datetime
import time
import logging
from fastapi import FastAPI, Request
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT / "src"))

from pipeline import load_models, process_ticket
from database import SessionLocal, Ticket, init_db
# endregion


# region: Model Loading (runs once, at startup)
ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading models, please wait...")
    ml_models.update(load_models())
    init_db()
    print("Models loaded. API ready.")
    yield
    ml_models.clear()

app = FastAPI(title="Customer Support Ticket Intelligence API", lifespan=lifespan)
# endregion


# region: Monitoring — Request Logging
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


# region: Schemas
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


# region: Endpoints
@app.post("/tickets", response_model=TicketResponse)
def create_ticket(request: TicketRequest):
    result = process_ticket(request.instruction, request.category, ml_models)
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
        if request_stats["total_requests"] > 0 else 0
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