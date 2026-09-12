# region: Setup and Imports
# Shared ML pipeline logic, used by both the FastAPI backend (api.py) and the
# Streamlit dashboard (dashboard.py), so the logic exists in exactly one place.

from pathlib import Path
import sys
import joblib
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT / "src"))

from business_decision import assign_priority
from response_generation import retrieve_similar_responses, generate_draft_response

HF_MODEL_REPO = "amirmovasati/support-ticket-distilbert-intent"
# endregion


# region: Model Loading
# Returns a dict of loaded models. The caller (FastAPI's lifespan, or
# Streamlit's cache) decides how to cache this — this function itself is
# just "do the loading, once, when asked."

def load_models() -> dict:
    from huggingface_hub import hf_hub_download

    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_REPO)
    classifier = AutoModelForSequenceClassification.from_pretrained(HF_MODEL_REPO)
    label_encoder_path = hf_hub_download(repo_id=HF_MODEL_REPO, filename="label_encoder.joblib")
    label_encoder = joblib.load(label_encoder_path)
    return {"tokenizer": tokenizer, "classifier": classifier, "label_encoder": label_encoder}
# endregion


# region: Core Prediction Logic
# Classify -> prioritize -> retrieve+generate. Same logic as before, just relocated.

def classify_intent(text: str, models: dict) -> str:
    inputs = models["tokenizer"](text, return_tensors="pt", truncation=True, padding="max_length", max_length=32)
    with torch.no_grad():
        outputs = models["classifier"](**inputs)
    predicted_id = torch.argmax(outputs.logits, dim=1).item()
    return models["label_encoder"].inverse_transform([predicted_id])[0]


def process_ticket(instruction: str, category: str, models: dict) -> dict:
    intent = classify_intent(instruction, models)
    flags = ""  # live tickets have no pre-existing 'flags' field
    priority = assign_priority(intent, flags)
    retrieved_examples = retrieve_similar_responses(instruction)
    draft_response = generate_draft_response(instruction, retrieved_examples)
    return {"intent": intent, "priority": priority, "draft_response": draft_response}
# endregion