from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
from sklearn.metrics import classification_report
import torch.nn as nn
import torch
import numpy

from tqdm import tqdm

print(torch.__version__)
print(numpy.__version__)
print("GPU disponível:", torch.cuda.is_available())

# Step 1: Carregar o dataset IMDB
dataset = load_dataset("stanfordnlp/imdb")

print("🔤 Carregando tokenizer BERT...")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
print("✅ Tokenizer pronto!\n")

# Step 2: Pré-processamento
MAX_LEN = 128

def text_pipeline(text):
    encoding = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=MAX_LEN,
        return_tensors="pt"
    )
    return encoding["input_ids"].squeeze(0)

def collate_fn(batch):
    texts = torch.stack([text_pipeline(item["text"]) for item in batch])
    labels = torch.tensor([item["label"] for item in batch])
    return texts, labels

train_loader = DataLoader(dataset["train"], batch_size=8, shuffle=True, collate_fn=collate_fn)
test_loader  = DataLoader(dataset["test"],  batch_size=8, shuffle=False, collate_fn=collate_fn)

# Step 3: Construindo o modelo LSTM
class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=512, n_layers=2, dropout=0.2):
        super(SentimentLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout
        )
        self.fc = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        emb = self.dropout(self.embedding(x))
        out, (hidden, _) = self.lstm(emb)
        last = hidden[-1]
        return self.fc(self.dropout(last)).squeeze(1)

VOCAB_SIZE = tokenizer.vocab_size
EMBED_DIM  = 64
HIDDEN_DIM = 128
N_LAYERS   = 3

model = SentimentLSTM(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, N_LAYERS)
print(model)
print(f"\nParâmetros: {sum(p.numel() for p in model.parameters()):,}")

# Step 4: Treinamento
device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model     = model.to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

# ✅ Fix 3: scheduler declarado ANTES do loop
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.8)

def trainEpoch(model, loader, epoch, total_epochs):
    model.train()
    total_loss, correct = 0, 0

    bar = tqdm(loader, desc=f"Época {epoch}/{total_epochs} [treino]",
               unit="batch", colour="blue")
    
    for texts, labels in bar:
        texts, labels = texts.to(device), labels.float().to(device)
        optimizer.zero_grad()
        preds = model(texts)
        loss  = criterion(preds, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
        correct    += ((preds > 0) == labels.bool()).sum().item()
    return total_loss / len(loader), correct / len(loader.dataset)

EPOCHS = 20

for epoch in range(EPOCHS):
    train_loss, train_acc = trainEpoch(model, train_loader, epoch, EPOCHS)
    scheduler.step()  # ✅ Fix 3: chamado dentro do loop
    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {train_loss:.4f} - Accuracy: {train_acc:.4f}")

# Step 5: Avaliação
# ✅ Fix 1: função recebe model e loader como parâmetros
def evaluate(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for texts, labels in loader:
            texts = texts.to(device)
            preds = (model(texts) > 0).cpu().int().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.int().tolist())
    return all_preds, all_labels

preds, labels = evaluate(model, test_loader)
print("\n📊 Relatório de classificação:")
print(classification_report(labels, preds, target_names=["negativo", "positivo"]))

# Step 6: Testando com frases próprias
def predict(text):
    model.eval()
    # ✅ Fix 2: unsqueeze(0) para adicionar dimensão do batch corretamente
    ids = text_pipeline(text).unsqueeze(0).to(device)
    with torch.no_grad():
        score = torch.sigmoid(model(ids)).item()
    label = "Positivo 😊" if score > 0.5 else "Negativo 😞"
    return f"{label}  (confiança: {score:.2%})"

print(predict("This movie was absolutely amazing!"))
print(predict("I wasted two hours of my life watching this."))
print(predict("Not bad, but could have been better."))

# Salvar modelo 
torch.save(model.state_dict(), "sentiment_model.pth")
print("✅ Modelo salvo!")
