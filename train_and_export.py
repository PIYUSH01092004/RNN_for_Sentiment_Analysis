import os
import re
import pickle
import json
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Download NLTK resources quietly
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=1):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        return out

def clean_text(text, english_stopwords):
    if not isinstance(text, str):
        return ""
    # 1. Lowercase
    text = text.lower()
    # 2. Remove URLs
    text = re.sub(r"http\S+", "", text)
    # 3. Remove HTML tags
    text = re.sub(r"<.*?>", "", text)
    # 4. Remove Punctuations
    text = re.sub(r"[^A-Za-z0-9\s]", "", text)
    # 5. Remove Stopwords
    words = text.split()
    filtered_words = [w for w in words if w not in english_stopwords]
    return " ".join(filtered_words)

def main():
    print("=== Training & Exporting RNN Sentiment Analysis Model ===")
    
    csv_path = "IMDB Dataset.csv"
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"{csv_path} not found!")
        
    print("1. Loading dataset...")
    df = pd.read_csv(csv_path)
    print(f"Dataset shape: {df.shape}")
    
    # Drop duplicates as in notebook
    df.drop_duplicates(inplace=True)
    print(f"Shape after removing duplicates: {df.shape}")
    
    # Use subset for fast & reliable training if full dataset takes time, or full dataset
    # We can use 20,000 samples for fast training while retaining high accuracy
    sample_df = df.sample(n=min(25000, len(df)), random_state=42).copy()
    
    print("2. Preprocessing text reviews...")
    english_stopwords = set(stopwords.words("english"))
    sample_df["clean_review"] = sample_df["review"].apply(lambda t: clean_text(t, english_stopwords))
    
    print("3. Vectorizing with TF-IDF (max_features=5000)...")
    tfidf = TfidfVectorizer(max_features=5000)
    X_tfidf = tfidf.fit_transform(sample_df["clean_review"]).toarray()
    
    le = LabelEncoder()
    y = le.fit_transform(sample_df["sentiment"])
    
    print("4. Splitting Train/Test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)
    
    # Prepare PyTorch DataLoaders
    train_set = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float())
    test_set = TensorDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).float())
    
    train_loader = DataLoader(train_set, shuffle=True, batch_size=64)
    test_loader = DataLoader(test_set, shuffle=False, batch_size=64)
    
    print("5. Initializing PyTorch SimpleRNN Model...")
    input_size = X_train.shape[1] # 5000
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = RNN(input_size=input_size, hidden_size=128, num_layers=1).to(device)
    criterion = nn.BCEWithLogitsLoss() # Numerically stable BCE
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    epochs = 8
    print(f"6. Training for {epochs} epochs...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            
            # Add sequence dim: (batch_size, seq_len=1, features=5000)
            xb = xb.unsqueeze(1)
            outputs = model(xb).squeeze(1)
            
            loss = criterion(outputs, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader)
        print(f"   Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")
        
    print("7. Evaluating model on test set...")
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            xb = xb.unsqueeze(1)
            outputs = model(xb).squeeze(1)
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()
            total += yb.size(0)
            correct += (preds == yb).sum().item()
            
    accuracy = (correct / total) * 100
    print(f"Test Accuracy: {accuracy:.2f}%")
    
    print("8. Saving model artifacts...")
    # Save TF-IDF Vectorizer
    with open("tfidf_vectorizer.pkl", "wb") as f:
        pickle.dump(tfidf, f)
        
    # Save PyTorch Model state dict
    torch.save(model.state_dict(), "rnn_model.pth")
    
    # Save metadata
    metadata = {
        "input_size": 5000,
        "hidden_size": 128,
        "num_layers": 1,
        "test_accuracy": round(accuracy, 2),
        "total_dataset_size": len(df),
        "training_samples": len(sample_df)
    }
    with open("model_meta.json", "w") as f:
        json.dump(metadata, f, indent=2)
        
    print("SUCCESS: Artifacts saved (tfidf_vectorizer.pkl, rnn_model.pth, model_meta.json)")

if __name__ == "__main__":
    main()
