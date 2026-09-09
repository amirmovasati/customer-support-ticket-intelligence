# region: Setup and Data Loading
# Reuses the same cleaned data and label encoding as the Transformer phase.

from pathlib import Path
from collections import Counter
import pandas as pd
import joblib
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLEAN_DATA_FILE = PROJECT_ROOT / "data" / "bitext_customer_support_clean.csv"
OUTPUTS_PATH = Path(r"C:\Projects\SupportTicketOutputs")

df = pd.read_csv(CLEAN_DATA_FILE)
label_encoder = joblib.load(OUTPUTS_PATH / "label_encoder.joblib")
df["label"] = label_encoder.transform(df["intent"])
# endregion


# region: Train/Test Split and Vocabulary
# LSTM (unlike DistilBERT) has no pretrained vocabulary, so we build our own
# from the training text only (never from test data, to avoid leakage).

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])

def tokenize(text):
    return text.lower().split()

counter = Counter()
for text in train_df["instruction"]:
    counter.update(tokenize(text))

vocab = {"<pad>": 0, "<unk>": 1}
for word, freq in counter.items():
    if freq >= 2:
        vocab[word] = len(vocab)

print(f"Vocabulary size: {len(vocab)}")
# endregion


# region: Text Encoding and Dataset
# Converts each instruction into a fixed-length sequence of word indices.

MAX_LEN = 16

def encode(text):
    tokens = tokenize(text)
    ids = [vocab.get(tok, vocab["<unk>"]) for tok in tokens[:MAX_LEN]]
    ids += [vocab["<pad>"]] * (MAX_LEN - len(ids))
    return ids

class TicketDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = [encode(t) for t in texts]
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.tensor(self.texts[idx]), torch.tensor(self.labels[idx])

train_loader = DataLoader(TicketDataset(train_df["instruction"].tolist(), train_df["label"].tolist()), batch_size=32, shuffle=True)
test_loader = DataLoader(TicketDataset(test_df["instruction"].tolist(), test_df["label"].tolist()), batch_size=64)
# endregion


# region: LSTM Model Definition
# Embedding layer (learned from scratch) -> LSTM -> classification head.

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        embedded = self.embedding(x)
        _, (hidden, _) = self.lstm(embedded)
        return self.fc(hidden[-1])

num_labels = df["label"].nunique()
model = LSTMClassifier(vocab_size=len(vocab), embed_dim=64, hidden_dim=128, num_classes=num_labels)
# endregion


# region: Manual Training Loop
# Forward -> Loss -> Backpropagation -> Optimizer Update, same cycle from the course.

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

EPOCHS = 5
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for texts, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(texts)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{EPOCHS} - avg train loss: {total_loss/len(train_loader):.4f}")
# endregion


# region: Evaluation
# model.eval() + torch.no_grad() disable dropout/gradient tracking during inference.

model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for texts, labels in test_loader:
        outputs = model(texts)
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.tolist())

accuracy = accuracy_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds, average="weighted")
print(f"\nLSTM test accuracy: {accuracy:.4f}")
print(f"LSTM test f1_weighted: {f1:.4f}")
# endregion