# region: Setup
# Uploads the fine-tuned DistilBERT model and label encoder to Hugging Face Hub,
# so the model is publicly downloadable (needed for anyone to run the Docker image).

from pathlib import Path
import joblib
from huggingface_hub import HfApi, create_repo
from transformers import AutoTokenizer, AutoModelForSequenceClassification

OUTPUTS_PATH = Path(r"C:\Projects\SupportTicketOutputs")
MODEL_DIR = OUTPUTS_PATH / "distilbert_intent_model"

HF_USERNAME = "amirmovasati"  # change if your Hugging Face username differs
REPO_ID = f"{HF_USERNAME}/support-ticket-distilbert-intent"
# endregion


# region: Upload Model and Tokenizer
# push_to_hub() creates the repo (if needed) and uploads model + tokenizer files.

def upload():
    create_repo(REPO_ID, exist_ok=True)

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

    model.push_to_hub(REPO_ID)
    tokenizer.push_to_hub(REPO_ID)
    print(f"Model and tokenizer pushed to https://huggingface.co/{REPO_ID}")

    # label_encoder.joblib isn't a model file, so we upload it manually
    api = HfApi()
    api.upload_file(
        path_or_fileobj=str(OUTPUTS_PATH / "label_encoder.joblib"),
        path_in_repo="label_encoder.joblib",
        repo_id=REPO_ID,
    )
    print("label_encoder.joblib uploaded.")
# endregion


if __name__ == "__main__":
    upload()