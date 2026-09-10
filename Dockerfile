FROM python:3.11-slim

WORKDIR /app

ENV OUTPUTS_PATH=/app/outputs
RUN mkdir -p /app/outputs

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY data/bitext_customer_support_clean.csv ./data/

EXPOSE 8000

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]