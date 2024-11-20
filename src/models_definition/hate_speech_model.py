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
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import os

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
        
        rf1 = RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            class_weight='balanced',
            random_state=1
        )
        
        rf2 = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_leaf=2,
            class_weight='balanced_subsample',
            random_state=2
        )
        
        rf3 = RandomForestClassifier(
            n_estimators=200,
            class_weight='balanced',
            random_state=3
        )
        
        xgb = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
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
        
        self.smote = SMOTE(random_state=42)

    def tokenize_text(self, text):
        """Tokeniza el texto usando una aproximación simple basada en espacios y puntuación."""
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        tokens = [token.strip() for token in text.split()]
        tokens = [token for token in tokens if token not in stop_words]
        return tokens

    def preprocess_text(self, text):
        """Preprocesa el texto para el análisis."""
        if not isinstance(text, str):
            return ""
        tokens = self.tokenize_text(text)
        return ' '.join(tokens)

    def train(self, X, y):
        """Entrena el modelo con los datos proporcionados y realiza la validación cruzada."""
        print("Iniciando entrenamiento del modelo...")
        
        # Preprocesar textos
        print("Preprocesando textos...")
        X_processed = [self.preprocess_text(text) for text in X]
        
        # Dividir en conjuntos de entrenamiento y prueba antes de cualquier transformación
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print("\nDistribución original en conjunto de entrenamiento:")
        print(pd.Series(y_train).value_counts(normalize=True))
        
        # Vectorizar los datos de entrenamiento
        print("Vectorizando textos de entrenamiento...")
        X_train_vectorized = self.vectorizer.fit_transform(X_train)
        
        # Aplicar SMOTE solo a los datos de entrenamiento
        print("Aplicando SMOTE a los datos de entrenamiento...")
        X_train_resampled, y_train_resampled = self.smote.fit_resample(X_train_vectorized, y_train)
        
        print("\nDistribución después de SMOTE en conjunto de entrenamiento:")
        print(pd.Series(y_train_resampled).value_counts(normalize=True))
        
        # Entrenar el modelo con los datos balanceados
        print("Entrenando el modelo...")
        self.model.fit(X_train_resampled, y_train_resampled)
        
        # Vectorizar datos de prueba (sin aplicar SMOTE)
        X_test_vectorized = self.vectorizer.transform(X_test)
        
        # Evaluación en conjunto de prueba (sin SMOTE)
        print("\nEvaluación en conjunto de prueba (sin SMOTE):")
        y_pred = self.model.predict(X_test_vectorized)
        print("\nReporte de clasificación:")
        print(classification_report(y_test, y_pred))
        
        # Matriz de confusión
        cm = confusion_matrix(y_test, y_pred)
        print("\nMatriz de confusión:")
        print(cm)
        
        # ROC curve y AUC
        y_pred_proba = self.model.predict_proba(X_test_vectorized)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        print(f"\nÁrea bajo la curva ROC: {roc_auc:.3f}")
        
        # Validación cruzada en datos originales (no SMOTE)
        scores = cross_val_score(self.model, X_train_vectorized, y_train, cv=5)
        print(f"\nResultados de la validación cruzada (sin SMOTE):")
        print(f"Precisión media: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")

    def predict_proba(self, text):
        """Realiza la predicción de probabilidad para un texto."""
        processed_text = self.preprocess_text(text)
        vectorized_text = self.vectorizer.transform([processed_text])
        return self.model.predict_proba(vectorized_text)[0, 1]

    def save(self, filepath):
        """Guarda el modelo entrenado y el vectorizador."""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'vectorizer': self.vectorizer
            }, f)

def train_and_save_model():
    os.makedirs('models', exist_ok=True)
    
    print("Cargando datos de entrenamiento...")
    train_data = pd.read_csv('data/raw/youtoxic_english_1000.csv')
    
    print("\nDistribución original en el dataset completo:")
    print(train_data['IsHatespeech'].value_counts(normalize=True))
    
    model = HateSpeechModel()
    model.train(train_data['Text'], train_data['IsHatespeech'])
    
    model.save('models/hate_speech_model_aug.pkl')
    print("\nModelo entrenado y guardado exitosamente.")

if __name__ == "__main__":
    train_and_save_model()