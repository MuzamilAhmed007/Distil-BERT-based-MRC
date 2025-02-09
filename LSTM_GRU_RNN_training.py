import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer
from datasets import load_dataset
from tqdm import tqdm

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load WikiQA dataset
dataset = load_dataset("wiki_qa")

# Load BERT tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Tokenization function
def tokenize_function(examples):
    return tokenizer(examples["question"], examples["answer"], truncation=True, padding="max_length", max_length=128)

# Apply tokenization
train_data = dataset["train"].map(tokenize_function, batched=True)
val_data = dataset["validation"].map(tokenize_function, batched=True)

# Convert to PyTorch tensor format
train_data.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
val_data.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# Custom Dataset class
class WikiQADataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return {
            "input_ids": self.dataset[idx]["input_ids"],
            "attention_mask": self.dataset[idx]["attention_mask"],
            "label": self.dataset[idx]["label"]
        }

# Create dataset instances
train_dataset = WikiQADataset(train_data)
val_dataset = WikiQADataset(val_data)

# Define batch size
BATCH_SIZE = 16

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Define model with RNN, LSTM, and GRU
class TextClassifier(nn.Module):
    def __init__(self, model_type="LSTM", embedding_dim=768, hidden_dim=128, output_dim=2, num_layers=2):
        super(TextClassifier, self).__init__()
        self.model_type = model_type
        self.embedding = nn.Embedding(30522, embedding_dim)  # Using BERT vocabulary size
        self.hidden_dim = hidden_dim

        if model_type == "RNN":
            self.rnn = nn.RNN(embedding_dim, hidden_dim, num_layers, batch_first=True)
        elif model_type == "LSTM":
            self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        elif model_type == "GRU":
            self.gru = nn.GRU(embedding_dim, hidden_dim, num_layers, batch_first=True)

        self.fc = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input_ids):
        embedded = self.embedding(input_ids)

        if self.model_type == "RNN":
            out, _ = self.rnn(embedded)
        elif self.model_type == "LSTM":
            out, _ = self.lstm(embedded)
        elif self.model_type == "GRU":
            out, _ = self.gru(embedded)

        out = self.fc(out[:, -1, :])  # Use the last hidden state
        return self.softmax(out)

# Initialize model (change model_type to "RNN", "LSTM", or "GRU")
model_type = "LSTM"  # Change to "RNN" or "GRU"
model = TextClassifier(model_type=model_type).to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=2e-4)

# Number of epochs
EPOCHS = 5

# Store training history
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

# Training loop
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    print(f"\nEpoch {epoch + 1}/{EPOCHS} - Training {model_type}...")
    for batch in tqdm(train_loader):
        input_ids = batch["input_ids"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = torch.argmax(outputs, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    train_loss = total_loss / len(train_loader)
    train_accuracy = correct / total
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)

    print(f"Training Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.4f}")

    # Validation loop
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    print(f"\nValidating {model_type}...")
    with torch.no_grad():
        for batch in tqdm(val_loader):
            input_ids = batch["input_ids"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_loss = total_loss / len(val_loader)
    val_accuracy = correct / total
    val_losses.append(val_loss)
    val_accuracies.append(val_accuracy)

    print(f"Validation Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}")

# Plot training and validation loss
plt.figure(figsize=(8, 5))
plt.plot(range(1, EPOCHS + 1), train_losses, label="Training Loss", marker="o")
plt.plot(range(1, EPOCHS + 1), val_losses, label="Validation Loss", marker="s")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title(f"{model_type} - Training vs Validation Loss")
plt.legend()
plt.show()

# Plot training and validation accuracy
plt.figure(figsize=(8, 5))
plt.plot(range(1, EPOCHS + 1), train_accuracies, label="Training Accuracy", marker="o")
plt.plot(range(1, EPOCHS + 1), val_accuracies, label="Validation Accuracy", marker="s")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title(f"{model_type} - Training vs Validation Accuracy")
plt.legend()
plt.show()
