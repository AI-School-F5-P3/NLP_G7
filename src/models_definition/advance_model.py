# advanced_model.py
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from kafka import KafkaConsumer, KafkaProducer
import json
from datetime import datetime
import asyncio
import aiohttp
from fastapi import FastAPI, WebSocket
import redis
import logging
from typing import List, Dict

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuración de Redis para caché
REDIS_CONFIG = {
    'host': 'localhost',
    'port': 6379,
    'db': 0
}

# Configuración de Kafka
KAFKA_CONFIG = {
    'bootstrap_servers': ['localhost:9092'],
    'comment_topic': 'youtube_comments',
    'result_topic': 'analysis_results'
}

class YouTubeCommentsDataset(Dataset):
    def __init__(self, comments: List[str], labels: List[int], tokenizer, max_len=128):
        self.comments = comments
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.comments)

    def __getitem__(self, idx):
        comment = str(self.comments[idx])
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            comment,
            add_special_tokens=True,
            max_length=self.max_len,
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

class HateSpeechBERT(nn.Module):
    def __init__(self, n_classes=2):
        super(HateSpeechBERT, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-multilingual-cased')
        self.drop = nn.Dropout(p=0.3)
        self.fc = nn.Linear(self.bert.config.hidden_size, n_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled_output = outputs[1]
        output = self.drop(pooled_output)
        output = self.fc(output)
        return self.softmax(output)

class RealTimeProcessor:
    def __init__(self):
        self.redis_client = redis.Redis(**REDIS_CONFIG)
        self.producer = KafkaProducer(
            bootstrap_servers=KAFKA_CONFIG['bootstrap_servers'],
            value_serializer=lambda x: json.dumps(x).encode('utf-8')
        )
        self.consumer = KafkaConsumer(
            KAFKA_CONFIG['comment_topic'],
            bootstrap_servers=KAFKA_CONFIG['bootstrap_servers'],
            value_deserializer=lambda x: json.loads(x.decode('utf-8'))
        )
        
        # Cargar modelo y tokenizer
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
        self.model = HateSpeechBERT()
        self.model.load_state_dict(torch.load('hate_speech_bert.pth'))
        self.model.eval()

    async def process_comments(self):
        """Procesa comentarios en tiempo real desde Kafka."""
        while True:
            for message in self.consumer:
                comment = message.value
                
                # Verificar caché
                cache_key = f"comment:{hash(comment['text'])}"
                cached_result = self.redis_client.get(cache_key)
                
                if cached_result:
                    result = json.loads(cached_result)
                else:
                    # Procesar con el modelo
                    result = await self.analyze_comment(comment['text'])
                    # Guardar en caché
                    self.redis_client.setex(
                        cache_key,
                        3600,  # TTL 1 hora
                        json.dumps(result)
                    )

                # Enviar resultado
                self.producer.send(
                    KAFKA_CONFIG['result_topic'],
                    {
                        'comment_id': comment.get('id'),
                        'result': result,
                        'timestamp': datetime.now().isoformat()
                    }
                )

    async def analyze_comment(self, text: str) -> Dict:
        """Analiza un comentario usando el modelo BERT."""
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        with torch.no_grad():
            outputs = self.model(
                input_ids=encoding['input_ids'],
                attention_mask=encoding['attention_mask']
            )
            probabilities = outputs.numpy()[0]

        return {
            'is_hate': bool(np.argmax(probabilities)),
            'confidence': float(max(probabilities)),
            'text': text
        }

# FastAPI app para WebSocket
app = FastAPI()

@app.websocket("/ws/comments/{video_id}")
async def websocket_endpoint(websocket: WebSocket, video_id: str):
    await websocket.accept()
    processor = RealTimeProcessor()
    
    try:
        while True:
            # Recibir comentario
            data = await websocket.receive_text()
            comment = json.loads(data)
            
            # Analizar comentario
            result = await processor.analyze_comment(comment['text'])
            
            # Enviar resultado
            await websocket.send_json(result)
    except Exception as e:
        logger.error(f"Error in WebSocket connection: {e}")
    finally:
        await websocket.close()

# Función de entrenamiento del modelo
def train_model(train_dataloader, val_dataloader, epochs=3):
    model = HateSpeechBERT()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        for batch in train_dataloader:
            optimizer.zero_grad()
            
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['label']
            
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validación
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in val_dataloader:
                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask']
                labels = batch['label']
                
                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_accuracy = correct / total
        logger.info(f'Epoch {epoch+1}/{epochs}:')
        logger.info(f'Train Loss: {train_loss/len(train_dataloader)}')
        logger.info(f'Val Loss: {val_loss/len(val_dataloader)}')
        logger.info(f'Val Accuracy: {val_accuracy}')
    
    return model

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)