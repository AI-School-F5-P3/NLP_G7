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
from sklearn.model_selection import cross_val_score, learning_curve
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import seaborn as sns
import os

# Descarga solo stopwords
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

class HateSpeechModel:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            min_df=2
        )
        self.model = VotingClassifier(estimators=[
            ('rf1', RandomForestClassifier(n_estimators=100, max_depth=20, min_samples_split=5, random_state=1)),
            ('rf2', RandomForestClassifier(n_estimators=100, max_depth=15, min_samples_leaf=2, random_state=2)),
            ('rf3', RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=3)),
            ('xgb', XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=4))
        ], voting='soft')

    def tokenize_text(self, text):
        """Tokeniza el texto usando una aproximación simple basada en espacios y puntuación."""
        # Convertir a minúsculas
        text = text.lower()
        # Reemplazar puntuación con espacios
        text = re.sub(r'[^\w\s]', ' ', text)
        # Dividir por espacios y filtrar tokens vacíos
        tokens = [token.strip() for token in text.split()]
        # Filtrar stop words
        tokens = [token for token in tokens if token not in stop_words]
        return tokens

    def preprocess_text(self, text):
        """Preprocesa el texto para el análisis."""
        if not isinstance(text, str):
            return ""
        
        # Usar nuestra propia función de tokenización
        tokens = self.tokenize_text(text)
        return ' '.join(tokens)

    def train(self, X, y):
        """Entrena el modelo con los datos proporcionados y realiza la validación cruzada."""
        print("Iniciando entrenamiento del modelo...")
        
        # Preprocesar textos
        print("Preprocesando textos...")
        X_processed = [self.preprocess_text(text) for text in X]
        
        print("Vectorizando textos...")
        X_vectorized = self.vectorizer.fit_transform(X_processed)
        
        print("Entrenando el modelo...")
        self.model.fit(X_vectorized, y)
        
        # Realizar validación cruzada
        scores = cross_val_score(self.model, X_vectorized, y, cv=5)
        print(f"\nResultados de la validación cruzada:")
        print(f"Precisión media: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")

    def save(self, filepath):
        """Guarda el modelo entrenado y el vectorizador."""
        with open(filepath, 'wb') as f:
            pickle.dump({'model': self.model, 'vectorizer': self.vectorizer}, f)

def train_and_save_model():
    # Asegurarse de que el directorio models existe
    os.makedirs('models', exist_ok=True)
    
    # Cargar datos de entrenamiento
    print("Cargando datos de entrenamiento...")
    train_data = pd.read_csv('data/processed/train_data.csv')
    
    # Crear y entrenar modelo
    model = HateSpeechModel()
    model.train(train_data['Text'], train_data['IsHatespeech'])
    
    # Guardar modelo
    model.save('models/modelo_hate_speech.pkl')
    print("\nModelo entrenado y guardado exitosamente.")

if __name__ == "__main__":
    train_and_save_model()