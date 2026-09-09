# region: Setup and Data Loading
# Loads the cleaned dataset that serves as our retrieval knowledge base.

from pathlib import Path
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLEAN_DATA_FILE = PROJECT_ROOT / "data" / "bitext_customer_support_clean.csv"
OUTPUTS_PATH = Path(r"C:\Projects\SupportTicketOutputs")
EMBEDDINGS_FILE = OUTPUTS_PATH / "instruction_embeddings.npy"

df = pd.read_csv(CLEAN_DATA_FILE)
# endregion


# region: Build or Load Embedding Index
# Encodes every ticket's `instruction` text into a vector once, and caches it
# to disk so future runs don't need to re-embed 24k+ rows.

embedder = SentenceTransformer("all-MiniLM-L6-v2")

if EMBEDDINGS_FILE.exists():
    print("Loading cached embeddings...")
    instruction_embeddings = np.load(EMBEDDINGS_FILE)
else:
    print("Encoding all instructions (first run only, this may take a few minutes)...")
    instruction_embeddings = embedder.encode(
        df["instruction"].tolist(), show_progress_bar=True
    )
    np.save(EMBEDDINGS_FILE, instruction_embeddings)
    print(f"Saved embeddings to {EMBEDDINGS_FILE}")
# endregion


# region: Retrieval Function
# Given a new ticket's text, finds the most semantically similar past tickets
# and returns their example responses as grounding context.

def retrieve_similar_responses(query_text: str, top_k: int = 3) -> list[str]:
    query_embedding = embedder.encode([query_text])
    similarities = cosine_similarity(query_embedding, instruction_embeddings)[0]
    top_indices = np.argsort(similarities)[::-1][:top_k]
    return df.iloc[top_indices]["response"].tolist()
# endregion


# region: Generation Model
# A small, free, fully local model — no API key or cost required to run this project.

GEN_MODEL_NAME = "google/flan-t5-base"
gen_tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL_NAME)
gen_model = AutoModelForSeq2SeqLM.from_pretrained(GEN_MODEL_NAME)

def generate_draft_response(query_text: str, retrieved_examples: list[str]) -> str:
    context = "\n".join(f"- {ex}" for ex in retrieved_examples)
    prompt = (
        "You are a customer support assistant. Using the example responses below "
        "as style and content guidance, write a short, helpful reply to the customer's message.\n\n"
        f"Example responses:\n{context}\n\n"
        f"Customer message: {query_text}\n\n"
        "Your reply:"
    )
    inputs = gen_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    outputs = gen_model.generate(
        **inputs,
        max_new_tokens=120,
        min_new_tokens=40,
        num_beams=4,
        no_repeat_ngram_size=3,
        early_stopping=True,
    )
    return gen_tokenizer.decode(outputs[0], skip_special_tokens=True)


# region: End-to-End Test
# Tries the full retrieve -> generate pipeline on a few example tickets.

if __name__ == "__main__":
    test_tickets = [
        "I want to cancel my order, it's taking too long",
        "how do i get money back for my broken item",
        "can you help me change my delivery address",
    ]
    for ticket in test_tickets:
        examples = retrieve_similar_responses(ticket)
        draft = generate_draft_response(ticket, examples)
        print(f"\nTicket: {ticket}")
        print(f"Draft response: {draft}")
# endregion