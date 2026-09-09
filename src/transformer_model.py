# region: Setup and Data Loading
# Loads the cleaned dataset and prepares label encoding.

from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLEAN_DATA_FILE = PROJECT_ROOT / "data" / "bitext_customer_support_clean.csv"
OUTPUTS_PATH = Path(r"C:\Projects\SupportTicketOutputs")
OUTPUTS_PATH.mkdir(parents=True, exist_ok=True)
MODEL_DIR = OUTPUTS_PATH / "distilbert_intent_model"

df = pd.read_csv(CLEAN_DATA_FILE)

label_encoder = LabelEncoder()
df["label"] = label_encoder.fit_transform(df["intent"])
joblib.dump(label_encoder, OUTPUTS_PATH / "label_encoder.joblib")
# endregion


# region: Train/Test Split
# Same stratified 80/20 split logic as the baseline, for a fair comparison.

train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["label"],
)
print(f"Train size: {len(train_df)}, Test size: {len(test_df)}")
# endregion


# region: Tokenization
# Converts raw text into the numeric input format DistilBERT expects.

MODEL_NAME = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_batch(batch):
    return tokenizer(batch["instruction"], truncation=True, padding="max_length", max_length=32)

train_dataset = Dataset.from_pandas(train_df[["instruction", "label"]])
test_dataset = Dataset.from_pandas(test_df[["instruction", "label"]])

train_dataset = train_dataset.map(tokenize_batch, batched=True)
test_dataset = test_dataset.map(tokenize_batch, batched=True)
# endregion


# region: Model Loading
# Loads the pretrained DistilBERT with a fresh classification head sized to our number of intents.

num_labels = df["label"].nunique()
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=num_labels)
# endregion


# region: Training Configuration and Run
# Defines how training proceeds (epochs, batch size, evaluation strategy).

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_weighted": f1_score(labels, preds, average="weighted"),
    }

training_args = TrainingArguments(
    output_dir=str(OUTPUTS_PATH / "training_checkpoints"),
    eval_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    load_best_model_at_end=True,
    metric_for_best_model="f1_weighted",
    logging_steps=50,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()
# endregion


# region: Final Evaluation and Save
# Reports final metrics and persists the fine-tuned model for later phases.

results = trainer.evaluate()
print("\nFinal evaluation results:", results)

trainer.save_model(str(MODEL_DIR))
tokenizer.save_pretrained(str(MODEL_DIR))
print(f"Saved fine-tuned model to {MODEL_DIR}")
# endregion