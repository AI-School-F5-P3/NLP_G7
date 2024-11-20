import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score
import torch
from torch import nn
from transformers import BertTokenizer, BertModel
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.nn import functional as F
import random
from typing import Tuple
from nltk.tokenize import word_tokenize
import nltk
import logging
nltk.download('punkt')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HateSpeechDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128, augment=False):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.augment = augment

    def __len__(self):
        return len(self.texts)

    def _augment_text(self, text):
        tokens = word_tokenize(text)
        
        if random.random() < 0.3:
            tokens = [token for token in tokens if random.random() > 0.15]
        
        if random.random() < 0.2:
            random.shuffle(tokens)
        
        if random.random() < 0.1:
            tokens.insert(random.randint(0, len(tokens)), '[UNK]')
        
        return ' '.join(tokens)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        if self.augment and self.labels[idx] == 1 and random.random() < 0.5:
            text = self._augment_text(text)
        label = int(self.labels[idx])

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

class BERTHateSpeechClassifier(nn.Module):
    def __init__(self, dropout=0.5):
        super(BERTHateSpeechClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout1 = nn.Dropout(dropout)
        self.linear1 = nn.Linear(768, 384)
        self.layer_norm1 = nn.LayerNorm(384)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(384, 192)
        self.layer_norm2 = nn.LayerNorm(192)
        self.dropout3 = nn.Dropout(dropout)
        self.linear3 = nn.Linear(192, 2)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]
        x = self.dropout1(pooled_output)
        x = self.linear1(x)
        x = self.layer_norm1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.linear2(x)
        x = self.layer_norm2(x)
        x = F.relu(x)
        x = self.dropout3(x)
        x = self.linear3(x)
        return x
    

def prepare_data(df: pd.DataFrame, test_size=0.15) -> Tuple:
    train_df, val_df = train_test_split(df, test_size=test_size, stratify=df['IsHatespeech'], random_state=42)
    return train_df['Text'].values, val_df['Text'].values, train_df['IsHatespeech'].values, val_df['IsHatespeech'].values

def prepare_balanced_sampler(labels):
    class_counts = np.bincount(labels)
    weights = 1. / class_counts
    weights /= weights.sum()
    sample_weights = weights[labels.astype(int)]
    return WeightedRandomSampler(sample_weights, len(sample_weights))

def train_model(model, train_loader, val_loader, device, epochs=4):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    criterion = nn.CrossEntropyLoss()

    best_val_f1 = 0
    patience = 5
    no_improve = 0

    train_metrics = []
    val_metrics = []

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        train_preds = []
        train_labels = []

        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_train_loss += loss.item()
            train_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            train_labels.extend(labels.cpu().numpy())

        model.eval()
        total_val_loss = 0
        val_preds = []
        val_labels = []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(outputs, labels)

                total_val_loss += loss.item()
                val_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        train_report = classification_report(train_labels, train_preds, output_dict=True)
        val_report = classification_report(val_labels, val_preds, output_dict=True)

        train_f1 = train_report['macro avg']['f1-score']
        val_f1 = val_report['macro avg']['f1-score']

        train_metrics.append({
            'loss': total_train_loss / len(train_loader),
            'f1_score': train_f1,
            'precision': precision_score(train_labels, train_preds, average='macro'),
            'recall': recall_score(train_labels, train_preds, average='macro')
        })

        val_metrics.append({
            'loss': total_val_loss / len(val_loader),
            'f1_score': val_f1,
            'precision': precision_score(val_labels, val_preds, average='macro'),
            'recall': recall_score(val_labels, val_preds, average='macro')
        })

        overfitting_percentage = abs(train_f1 - val_f1) / train_f1 * 100

        print(f"Epoch {epoch+1}:")
        print(f"Training Loss: {total_train_loss / len(train_loader):.4f}")
        print(f"Validation Loss: {total_val_loss / len(val_loader):.4f}")
        print(f"Training F1-score: {train_f1:.4f}")
        print(f"Validation F1-score: {val_f1:.4f}")
        print(f"Overfitting Percentage: {overfitting_percentage:.2f}%")
        print("-" * 50)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            no_improve = 0
            torch.save(model.state_dict(), 'models/bert_model.pt')
        else:
            no_improve += 1

        if no_improve >= patience:
            print("Early stopping triggered")
            break

    # Guardar métricas en archivo JSON
    import json
    with open('model_metrics.json', 'w') as f:
        json.dump({
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
        }, f, indent=4)

    print("\nMetrics saved to model_metrics.json")
    return train_metrics, val_metrics

def main():
    df = pd.read_csv('data/processed/train_data_bert.csv')
    X_train, X_val, y_train, y_val = prepare_data(df)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    train_dataset = HateSpeechDataset(X_train, y_train, tokenizer, augment=True)
    val_dataset = HateSpeechDataset(X_val, y_val, tokenizer, augment=False)

    train_sampler = prepare_balanced_sampler(y_train)

    # Ensure batch size is a factor of dataset size to avoid single-sample batches
    batch_size = 16
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        sampler=train_sampler,
        drop_last=True  # Drop the last batch if it's smaller than batch_size
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        drop_last=True
    )

    model = BERTHateSpeechClassifier()
    model = model.to(device)

    train_model(model, train_loader, val_loader, device)

if __name__ == "__main__":
    main()