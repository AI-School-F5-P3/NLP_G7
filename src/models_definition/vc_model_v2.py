import pickle
import re
import nltk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score, learning_curve, train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import seaborn as sns
from imblearn.combine import SMOTETomek
from datetime import datetime
import os
import json
from typing import Tuple, Dict, Any

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

class ModelEvaluationReport:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.metrics = {}
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def add_metric(self, name: str, value: Any) -> None:
        self.metrics[name] = value
        
    def calculate_overfitting(self, train_score: float, test_score: float) -> float:
        overfitting = abs(train_score - test_score)
        self.metrics['overfitting'] = overfitting
        return overfitting
        
    def save_report(self, report_dir: str) -> None:
        os.makedirs(report_dir, exist_ok=True)
        
        # Convertir métricas numpy a tipos nativos de Python
        metrics_dict = {}
        for key, value in self.metrics.items():
            if isinstance(value, np.ndarray):
                metrics_dict[key] = value.tolist()
            elif isinstance(value, np.float64) or isinstance(value, np.float32):
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

class HateSpeechModel:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            min_df=2
        )
        
        # Ajustados los parámetros para reducir overfitting
        rf1 = RandomForestClassifier(
            n_estimators=150,
            max_depth=15,
            min_samples_split=8,
            min_samples_leaf=4,
            class_weight='balanced',
            random_state=1
        )
        
        rf2 = RandomForestClassifier(
            n_estimators=150,
            max_depth=12,
            min_samples_leaf=3,
            class_weight='balanced_subsample',
            random_state=2
        )
        
        rf3 = RandomForestClassifier(
            n_estimators=150,
            max_depth=10,
            min_samples_split=10,
            class_weight='balanced',
            random_state=3
        )
        
        xgb = XGBClassifier(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=7,
            random_state=4
        )
        
        self.model = VotingClassifier(
            estimators=[
                ('rf1', rf1),
                ('rf2', rf2),
                ('rf3', rf3),
                ('xgb', xgb)
            ],
            voting='soft',
            weights=[1, 1, 1, 2]
        )
        
        # Reemplazamos SMOTE por SMOTETomek que es mejor para datos textuales
        self.resampler = SMOTETomek(random_state=42)
        self.evaluation_report = ModelEvaluationReport("hate_speech_classifier")

    def tokenize_text(self, text: str) -> list:
        """Tokeniza el texto usando una aproximación simple basada en espacios y puntuación."""
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        tokens = [token.strip() for token in text.split()]
        tokens = [token for token in tokens if token not in stop_words]
        return tokens

    def preprocess_text(self, text: str) -> str:
        """Preprocesa el texto para el análisis."""
        if not isinstance(text, str):
            return ""
        tokens = self.tokenize_text(text)
        return ' '.join(tokens)

    def separate_production_validation(self, X: pd.Series, y: pd.Series, n_samples: int = 20) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        """Separa un conjunto de datos para validación en producción."""
        # Asegurar que tenemos suficientes muestras de cada clase
        pos_indices = y[y == 1].index
        neg_indices = y[y == 0].index
        
        n_pos = min(n_samples // 2, len(pos_indices))
        n_neg = n_samples - n_pos
        
        # Seleccionar muestras estratificadas
        prod_pos_indices = np.random.choice(pos_indices, n_pos, replace=False)
        prod_neg_indices = np.random.choice(neg_indices, n_neg, replace=False)
        prod_indices = np.concatenate([prod_pos_indices, prod_neg_indices])
        
        # Separar datos
        X_prod = X[prod_indices]
        y_prod = y[prod_indices]
        X_train_full = X.drop(prod_indices)
        y_train_full = y.drop(prod_indices)
        
        return X_train_full, y_train_full, X_prod, y_prod

    def train(self, X: pd.Series, y: pd.Series) -> None:
        """Entrena el modelo con los datos proporcionados y realiza la validación."""
        print("Iniciando entrenamiento del modelo...")
        
        # Separar datos de validación de producción
        X_train_full, y_train_full, X_prod, y_prod = self.separate_production_validation(X, y)
        
        # Preprocesar textos
        print("Preprocesando textos...")
        X_processed = [self.preprocess_text(text) for text in X_train_full]
        
        # División train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full
        )
        
        # Vectorización
        print("Vectorizando textos...")
        X_train_vectorized = self.vectorizer.fit_transform(X_train)
        X_test_vectorized = self.vectorizer.transform(X_test)
        
        # Aplicar SMOTETomek solo a los datos de entrenamiento
        print("Aplicando SMOTETomek para balance de clases...")
        X_train_resampled, y_train_resampled = self.resampler.fit_resample(
            X_train_vectorized, y_train
        )
        
        # Entrenamiento
        print("Entrenando el modelo...")
        self.model.fit(X_train_resampled, y_train_resampled)
        
        # Evaluación
        train_score = self.model.score(X_train_resampled, y_train_resampled)
        test_score = self.model.score(X_test_vectorized, y_test)
        
        # Calcular overfitting
        overfitting = self.evaluation_report.calculate_overfitting(train_score, test_score)
        print(f"\nOverfitting: {overfitting:.4f}")
        
        if overfitting > 0.05:
            print("¡Advertencia! Overfitting superior al 5%")
        
        # Predicciones y métricas
        y_pred = self.model.predict(X_test_vectorized)
        y_pred_proba = self.model.predict_proba(X_test_vectorized)[:, 1]
        
        # Calcular métricas
        class_report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        # Guardar métricas en el reporte
        self.evaluation_report.add_metric('train_score', train_score)
        self.evaluation_report.add_metric('test_score', test_score)
        self.evaluation_report.add_metric('classification_report', class_report)
        self.evaluation_report.add_metric('confusion_matrix', cm.tolist())
        self.evaluation_report.add_metric('roc_auc', roc_auc)
        self.evaluation_report.add_metric('production_validation_size', len(X_prod))
        
        # Guardar datos de validación de producción
        self.save_production_validation(X_prod, y_prod)

    def save_production_validation(self, X_prod: pd.Series, y_prod: pd.Series) -> None:
        """Guarda los datos de validación de producción."""
        prod_validation = pd.DataFrame({
            'text': X_prod,
            'label': y_prod
        })
        
        # os.makedirs('data/production_validation', exist_ok=True)
        # prod_validation.to_csv(
        #     'data/production_validation/validation_set.csv',
        #     index=False
        # )

    def predict_proba(self, text: str) -> float:
        """Realiza la predicción de probabilidad para un texto."""
        processed_text = self.preprocess_text(text)
        vectorized_text = self.vectorizer.transform([processed_text])
        return self.model.predict_proba(vectorized_text)[0, 1]

    def save(self, filepath: str) -> None:
        """Guarda el modelo entrenado y el vectorizador."""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'vectorizer': self.vectorizer
            }, f)

def train_and_save_model():
    # Crear directorios necesarios
    os.makedirs('models', exist_ok=True)
    os.makedirs('reports', exist_ok=True)
    
    print("Cargando datos de entrenamiento...")
    train_data = pd.read_csv('data/processed/train_augmented_dataset.csv')
    
    print("\nDistribución original en el dataset completo:")
    print(train_data['IsHatespeech'].value_counts(normalize=True))
    
    model = HateSpeechModel()
    model.train(train_data['Text'], train_data['IsHatespeech'])
    
    # Guardar modelo
    model.save('models/hspeech_model_aug.pkl')
    
    # Guardar reporte de evaluación
    model.evaluation_report.save_report('reports')
    
    print("\nModelo entrenado y guardado exitosamente.")
    print("Reporte de evaluación guardado en la carpeta 'reports'.")

if __name__ == "__main__":
    train_and_save_model()