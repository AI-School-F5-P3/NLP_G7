import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple, Any
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
    set_seed
)
import pickle
from tqdm.auto import tqdm

# Configuración de reproducibilidad
set_seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

class ModelEvaluationReport:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.metrics = {}
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def add_metric(self, name: str, value: Any) -> None:
        self.metrics[name] = value
        
    def calculate_overfitting(self, train_score: float, val_score: float) -> float:
        overfitting = abs(train_score - val_score)
        self.metrics['overfitting'] = overfitting
        return overfitting
        
    def save_report(self, report_dir: str) -> None:
        os.makedirs(report_dir, exist_ok=True)
        
        metrics_dict = {}
        for key, value in self.metrics.items():
            if isinstance(value, np.ndarray):
                metrics_dict[key] = value.tolist()
            elif isinstance(value, (np.float64, np.float32)):
                metrics_dict[key] = float(value)
            else:
                metrics_dict[key] = value
        
        report = {
            'model_name': self.model_name,
            'timestamp': self.timestamp,
            'metrics': metrics_dict
        }
        
        filename = f"{self.model_name}_report_{self.timestamp}.json"
        filepath = os.path.join(report_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=4)

    def save_report_pickle(self, report_dir: str) -> None:
        os.makedirs(report_dir, exist_ok=True)
        filename = f"{self.model_name}_report_{self.timestamp}.pkl"
        filepath = os.path.join(report_dir, filename)
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

class HateSpeechDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

class BERTHateSpeechModel:
    def __init__(self, model_path: str = None, model_name: str = "bert-base-uncased"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=2,
            problem_type="single_label_classification"
        )
        
        # Cargar pesos del modelo si se proporciona una ruta
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        
        # Aplicar técnicas de regularización
        self.model.config.hidden_dropout_prob = 0.3
        self.model.config.attention_probs_dropout_prob = 0.3
        
        self.model = self.model.to(self.device)
        self.evaluation_report = ModelEvaluationReport("bert_hate_speech_classifier")

    def create_dataloaders(
        self,
        train_texts: List[str],
        train_labels: List[int],
        val_texts: List[str],
        val_labels: List[int],
        batch_size: int = 16
    ) -> Tuple[DataLoader, DataLoader]:
        # Ensure labels are numpy array of integers
        train_labels_np = np.array(train_labels, dtype=int)
        
        # Compute class weights correctly
        class_counts = np.bincount(train_labels_np)
        class_weights = len(train_labels_np) / (len(class_counts) * class_counts)
        
        # Create sample weights
        sample_weights = class_weights[train_labels_np]
        sample_weights = torch.FloatTensor(sample_weights)
        
        sampler = torch.utils.data.WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )

        train_dataset = HateSpeechDataset(train_texts, train_labels, self.tokenizer)
        val_dataset = HateSpeechDataset(val_texts, val_labels, self.tokenizer)

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=2
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2
        )

        return train_loader, val_loader

    def train_epoch(
        self,
        train_loader: DataLoader,
        optimizer,
        scheduler,
        epoch: int
    ) -> float:
        self.model.train()
        total_loss = 0
        correct_preds = 0
        total_preds = 0

        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}')
        for batch in progress_bar:
            optimizer.zero_grad()

            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['label'].to(self.device)

            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss
            total_loss += loss.item()

            # Calcular accuracy
            preds = torch.argmax(outputs.logits, dim=1)
            correct_preds += (preds == labels).sum().item()
            total_preds += len(labels)

            loss.backward()
            
            # Gradient clipping más conservador
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
            
            optimizer.step()
            scheduler.step()

            # Actualizar la barra de progreso
            progress_bar.set_postfix({
                'loss': loss.item(),
                'acc': correct_preds / total_preds
            })

        return total_loss / len(train_loader), correct_preds / total_preds

    def evaluate(self, val_loader: DataLoader) -> Tuple[float, float, np.ndarray, np.ndarray]:
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

                total_loss += outputs.loss.item()
                preds = torch.softmax(outputs.logits, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        # Calcular métricas
        pred_classes = np.argmax(all_preds, axis=1)
        accuracy = (pred_classes == all_labels).mean()

        return (
            total_loss / len(val_loader),
            accuracy,
            all_preds,
            all_labels
        )

    def train(
        self,
        X: pd.Series,
        y: pd.Series,
        epochs: int = 3,
        batch_size: int = 32,  # Aumentado para mejor estabilidad
        learning_rate: float = 1e-5  # Reducido para mejor generalización
    ) -> None:
        print("Iniciando entrenamiento del modelo BERT...")
        
        # Separar datos de validación de producción de manera estratificada
        stratify_indices = []
        for label in [0, 1]:
            label_indices = y[y == label].index
            n_samples = min(10, len(label_indices))  # 10 muestras por clase
            stratify_indices.extend(np.random.choice(label_indices, n_samples, replace=False))
        
        X_prod = X[stratify_indices]
        y_prod = y[stratify_indices]
        X_train = X.drop(stratify_indices)
        y_train = y.drop(stratify_indices)
        
        # División train/validation
        X_train_data, X_val, y_train_data, y_val = train_test_split(
            X_train, y_train,
            test_size=0.2,
            stratify=y_train,
            random_state=42
        )
        
        # Crear dataloaders con manejo de desbalanceo
        train_loader, val_loader = self.create_dataloaders(
            X_train_data.tolist(),
            y_train_data.tolist(),
            X_val.tolist(),
            y_val.tolist(),
            batch_size
        )
        
        # Configurar optimizador con mayor regularización
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=0.1,  # Aumentada la regularización L2
            betas=(0.9, 0.999)
        )
        
        num_training_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_training_steps * 0.1,
            num_training_steps=num_training_steps
        )
        
        # Training loop con early stopping mejorado
        best_val_loss = float('inf')
        patience = 2  # épocas de paciencia para early stopping
        no_improve = 0
        best_epoch = 0
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            
            # Training
            train_loss, train_acc = self.train_epoch(
                train_loader,
                optimizer,
                scheduler,
                epoch
            )
            
            # Validation
            val_loss, val_acc, val_preds, val_labels = self.evaluate(val_loader)
            
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
            
            # Calcular overfitting
            overfitting = self.evaluation_report.calculate_overfitting(train_acc, val_acc)
            print(f"Overfitting: {overfitting:.4f}")
            
            # Early stopping mejorado
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                no_improve = 0
                # Guardar mejor modelo
                torch.save(self.model.state_dict(), 'models/best_model.pt')
            else:
                no_improve += 1
            
            if overfitting > 0.05 and epoch > 0:
                print("¡Advertencia! Overfitting superior al 5%")
                if no_improve >= patience:
                    print("Deteniendo entrenamiento debido al alto overfitting")
                    break
        
        print(f"\nMejor época: {best_epoch + 1}")
        
        # Cargar mejor modelo
        self.model.load_state_dict(torch.load('models/best_model.pt'))
        
        # Evaluación final
        _, final_val_acc, final_preds, final_labels = self.evaluate(val_loader)
        
        # Calcular métricas finales
        final_pred_classes = np.argmax(final_preds, axis=1)
        class_report = classification_report(final_labels, final_pred_classes, output_dict=True)
        cm = confusion_matrix(final_labels, final_pred_classes)
        fpr, tpr, _ = roc_curve(final_labels, final_preds[:, 1])
        roc_auc = auc(fpr, tpr)
        
        # Guardar métricas en el reporte
        self.evaluation_report.add_metric('train_accuracy', train_acc)
        self.evaluation_report.add_metric('final_val_accuracy', final_val_acc)
        self.evaluation_report.add_metric('classification_report', class_report)
        self.evaluation_report.add_metric('confusion_matrix', cm.tolist())
        self.evaluation_report.add_metric('roc_auc', roc_auc)
        self.evaluation_report.add_metric('best_epoch', best_epoch + 1)
        self.evaluation_report.add_metric('production_validation_size', len(X_prod))
        
        # Guardar datos de validación de producción
        self.save_production_validation(X_prod, y_prod)

    def predict_proba(self, text: str) -> float:
        """Realiza la predicción de probabilidad para un texto."""
        self.model.eval()
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=128,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=1)
            
        return probs[0][1].item()

    def save_production_validation(self, X_prod: pd.Series, y_prod: pd.Series) -> None:
        """Guarda los datos de validación de producción."""
        prod_validation = pd.DataFrame({
            'text': X_prod,
            'label': y_prod
        })
        
        os.makedirs('data/production_validation', exist_ok=True)
        prod_validation.to_csv(
            'data/production_validation/validation_set.csv',
            index=False
        )

    def save(self, filepath: str) -> None:
        """Guarda el modelo, el tokenizer y el reporte de evaluación."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save(self.model.state_dict(), filepath)
        self.tokenizer.save_pretrained(os.path.dirname(filepath))
        self.evaluation_report.save_report_pickle(os.path.dirname(filepath))


def main():
    # Ensure the necessary directories exist
    os.makedirs('models', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('data/production_validation', exist_ok=True)

    # Load dataset
    try:
        df = pd.read_csv('data/processed/train_data_bert.csv')
    except FileNotFoundError:
        print("Error: Training data file not found. Please prepare your dataset first.")
        return

    # Separate features and labels
    X = df['Text']
    y = df['IsHatespeech']

    # Initialize the BERT Hate Speech Model
    model = BERTHateSpeechModel(
        model_name="bert-base-uncased" #"microsoft/deberta-v3-small"
    )

    # Train the model
    model.train(
        X=X, 
        y=y, 
        epochs=3,  # Configurable based on your needs
        batch_size=32,
        learning_rate=1e-5
    )

    # Save the trained model
    model_save_path = 'models/bert_hate_speech_classifier.pt'
    model.save(model_save_path)

    # Optional: Demonstrate prediction on a sample text
    test_text = "An example text to test hate speech detection"
    hate_probability = model.predict_proba(test_text)
    print(f"Hate speech probability for test text: {hate_probability:.4f}")

    # Print evaluation report details
    print("\nModel Evaluation Metrics:")
    for metric_name, metric_value in model.evaluation_report.metrics.items():
        print(f"{metric_name}: {metric_value}")

if __name__ == "__main__":
    main()