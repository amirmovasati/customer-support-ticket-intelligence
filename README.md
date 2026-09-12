# Customer Support Ticket Intelligence

*[نسخه فارسی این صفحه](README.fa.md)*

An end-to-end NLP system that classifies incoming customer support tickets, assigns a business-driven priority level, and drafts a suggested response — combining a fine-tuned Transformer, RAG-based response generation, a SQL-backed ticket log, and a live dashboard.

## What it does

For any incoming support ticket, the system:
1. **Classifies intent** — one of 27 categories (e.g. `cancel_order`, `get_refund`, `payment_issue`) using a fine-tuned DistilBERT model.
2. **Assigns priority** — High / Medium / Low, via a transparent, business-defined rule engine (not a black box).
3. **Drafts a response** — using semantic retrieval (RAG) over real example responses, generated with a free, fully local language model.
4. **Logs everything** — to a SQL database, with reporting queries a support manager could actually use.

## Try it

- **Live demo:** Not currently deployed (Docker is the primary zero-setup way to run this — see below).
- **Run locally with Docker** (recommended, no setup beyond Docker itself):
  ```
  git clone https://github.com/amirmovasati/customer-support-ticket-intelligence.git
  cd customer-support-ticket-intelligence
  docker build -t support-ticket-api .
  docker run -p 8000:8000 support-ticket-api
  ```
  Then open `http://localhost:8000/docs` for the interactive API, or run the dashboard separately (see below).

- **Run the dashboard locally:**
  ```
  pip install -r requirements.txt
  streamlit run src/dashboard.py
  ```

## Results

| Model | Accuracy | Notes |
|---|---|---|
| TF-IDF + Logistic Regression (baseline) | 99.03% | Simple, fast, surprisingly strong |
| LSTM (trained from scratch) | 98.17% | Weaker than the baseline — demonstrates the value of pretrained knowledge |
| **Fine-tuned DistilBERT** | **99.76%** | ~75% lower error rate than the baseline |

The dataset is synthetically generated with highly distinctive vocabulary per intent, which explains the unusually high accuracy across all models — this was investigated and confirmed during EDA, not assumed.

## Architecture

```
Ticket text
    │
    ▼
[DistilBERT intent classifier]  (fine-tuned, hosted on Hugging Face Hub)
    │
    ▼
[Rule-based priority engine]  (intent → High/Medium/Low, + escalation signals)
    │
    ▼
[RAG response generation]  (sentence-transformers retrieval + flan-t5 generation)
    │
    ▼
[SQLite database]  (SQLAlchemy — swappable to PostgreSQL via one config line)
    │
    ▼
[FastAPI backend]  +  [Streamlit dashboard]
```

## Tech stack

Python, PyTorch, Hugging Face Transformers & Hub, sentence-transformers, scikit-learn, SQLAlchemy, FastAPI, Streamlit, Docker.

## Dataset

[Bitext Customer Support LLM Chatbot Training Dataset](https://github.com/bitext/customer-support-llm-chatbot-training-dataset) — 26,872 examples across 27 intents (24,635 after deduplication).

## Design decisions worth knowing

- **Free and local by design:** every model used (DistilBERT, sentence-transformers, flan-t5) is open, free, and runs on CPU — no API keys, no per-request cost, no risk of the project breaking due to a pricing or access change. This was a deliberate trade-off over using a paid LLM API; the RAG architecture is provider-agnostic, so swapping in a stronger paid model later is a drop-in change.
- **Priority is rule-based, not ML-predicted:** the dataset has no ground-truth priority labels, so this layer is an explicit, auditable business rule table rather than a trained (and unverifiable) model.
- **SQLite over PostgreSQL:** chosen so the project runs anywhere with zero database setup; the code uses standard SQLAlchemy, so migrating to PostgreSQL is a one-line connection-string change.

## Known limitations

- The small local response-generation model occasionally produces generic or slightly repetitive phrasing, and responses can be cut off if they exceed the configured token limit.
- The `flags`-based priority escalation signal (offensive language / negation) only exists in the historical training data — live tickets don't have it, so escalation currently only applies to the pre-loaded historical dataset, not new tickets submitted through the dashboard.
- SQLite inside the Docker container is not persistent across container restarts — fine for a demo, not for production.

## Project structure

```
├── src/
│   ├── data_ingestion.py       # Downloads the raw dataset
│   ├── baseline_model.py       # TF-IDF + Logistic Regression baseline
│   ├── transformer_model.py    # DistilBERT fine-tuning
│   ├── lstm_comparison.py      # From-scratch LSTM comparison
│   ├── business_decision.py    # Priority rule engine
│   ├── response_generation.py  # RAG retrieval + generation
│   ├── database.py             # SQLAlchemy models and connection
│   ├── populate_database.py    # Bulk-loads historical tickets
│   ├── reports.py              # SQL reporting queries
│   ├── pipeline.py             # Shared classify/prioritize/generate logic
│   ├── api.py                  # FastAPI backend
│   └── dashboard.py            # Streamlit dashboard
├── notebooks/                  # EDA
├── data/                       # Dataset (raw + cleaned)
├── Dockerfile
└── requirements.txt
```