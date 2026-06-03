
import os
import torch
import torch.nn as nn
from transformers import AutoTokenizer

class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=512, n_layers=2, dropout=0.2):
        super(SentimentLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout
        )
        self.fc      = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        emb = self.dropout(self.embedding(x))
        _, (hidden, _) = self.lstm(emb)
        last = hidden[-1]
        return self.fc(self.dropout(last)).squeeze(1)

# ── 2. Carrega tokenizer e modelo ──────────────────────────────────
tokenizer  = AutoTokenizer.from_pretrained("bert-base-uncased")
VOCAB_SIZE = tokenizer.vocab_size
device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SentimentLSTM(
    vocab_size = VOCAB_SIZE,
    embed_dim  = 64,
    hidden_dim = 128,
    n_layers   = 3,
    dropout    = 0.2
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "sentiment_model.pth")

model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()
print("✅ Modelo carregado!\n")

# ── 3. Funções de suporte ──────────────────────────────────────────
MAX_LEN = 256

def text_pipeline(text):
    encoding = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=MAX_LEN,
        return_tensors="pt"
    )
    return encoding["input_ids"].squeeze(0)

def predict(text):
    ids = text_pipeline(text).unsqueeze(0).to(device)
    with torch.no_grad():
        score = torch.sigmoid(model(ids)).item()
    label = "Positivo 😊" if score > 0.5 else "Negativo 😞"
    emoji = "😊" if score > 0.5 else "😞"
    return {"label": label, "emoji": emoji, "score": round(score * 100, 2)}
