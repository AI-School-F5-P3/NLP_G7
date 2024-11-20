import pickle
import re
import os
import sys
import logging
import numpy as np
import pandas as pd
import spacy
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

# Configure comprehensive logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s: %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('hate_speech_model.log')
    ]
)
logger = logging.getLogger(__name__)

class SpacyEmbeddingTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, vector_size=300):
        try:
            self.nlp = spacy.load('en_core_web_md')
        except OSError:
            logger.error("SpaCy model 'en_core_web_md' not found. Please install using: python -m spacy download en_core_web_md")
            raise
        
        self.vector_size = vector_size
        self.stop_words = self.nlp.Defaults.stop_words

    def _text_to_vector(self, text):
        """Convert text to averaged word embeddings using spaCy."""
        if not isinstance(text, str):
            return np.zeros(self.vector_size)
        
        # Preprocess text
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Process with spaCy
        doc = self.nlp(text)
        
        # Filter out stop words and very short tokens
        vectors = [token.vector for token in doc 
                   if not token.is_stop and len(token.text) > 1]
        
        if not vectors:
            return np.zeros(self.vector_size)
        
        # Average word vectors
        return np.mean(vectors, axis=0)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.array([self._text_to_vector(text) for text in X])

class HateSpeechModel:
    def __init__(self):
        # Base classifiers with embedding
        rf1 = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=10,
            class_weight='balanced',
            random_state=42
        )
        
        rf2 = RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            min_samples_split=12,
            class_weight='balanced_subsample',
            random_state=43
        )
        
        # Voting Classifier with pipelines
        self.model = VotingClassifier(
            estimators=[
                ('rf1', Pipeline([
                    ('embedding', SpacyEmbeddingTransformer()),
                    ('scaler', StandardScaler()),
                    ('clf', rf1)
                ])),
                ('rf2', Pipeline([
                    ('embedding', SpacyEmbeddingTransformer()),
                    ('scaler', StandardScaler()),
                    ('clf', rf2)
                ]))
            ],
            voting='soft'
        )

    def train(self, X: pd.Series, y: pd.Series):
        try:
            # Validar datos de entrada
            if X.empty or y.empty:
                raise ValueError("Input data cannot be empty")
            
            if len(X) != len(y):
                raise ValueError(f"Mismatch in input data lengths. X: {len(X)}, y: {len(y)}")
            
            # Dividir datos en conjuntos de entrenamiento y prueba
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            
            # Entrenar modelo en conjunto de entrenamiento
            self.model.fit(X_train, y_train)
            
            # Predicciones en conjuntos de entrenamiento y prueba
            y_train_pred = self.model.predict(X_train)
            y_test_pred = self.model.predict(X_test)
            
            # Calcular métricas de clasificación
            train_classification_rep = classification_report(y_train, y_train_pred, output_dict=True)
            test_classification_rep = classification_report(y_test, y_test_pred, output_dict=True)
            
            # Calcular métricas de overfitting
            overfitting_metrics = {
                'accuracy_diff': abs(train_classification_rep['accuracy'] - test_classification_rep['accuracy']),
                'precision_diff': {
                    'weighted': abs(train_classification_rep['weighted avg']['precision'] - test_classification_rep['weighted avg']['precision']),
                    'macro': abs(train_classification_rep['macro avg']['precision'] - test_classification_rep['macro avg']['precision'])
                },
                'recall_diff': {
                    'weighted': abs(train_classification_rep['weighted avg']['recall'] - test_classification_rep['weighted avg']['recall']),
                    'macro': abs(train_classification_rep['macro avg']['recall'] - test_classification_rep['macro avg']['recall'])
                },
                'f1_diff': {
                    'weighted': abs(train_classification_rep['weighted avg']['f1-score'] - test_classification_rep['weighted avg']['f1-score']),
                    'macro': abs(train_classification_rep['macro avg']['f1-score'] - test_classification_rep['macro avg']['f1-score'])
                }
            }
            
            # Registrar resultados
            logger.info("Modelo entrenado exitosamente")
            logger.info(f"Informe de Clasificación (Entrenamiento): {train_classification_rep}")
            logger.info(f"Informe de Clasificación (Prueba): {test_classification_rep}")
            logger.info(f"Métricas de Overfitting: {overfitting_metrics}")
            
            return {
                'train_classification_report': train_classification_rep,
                'test_classification_report': test_classification_rep,
                'overfitting_metrics': overfitting_metrics
            }
        
        except Exception as e:
            logger.error(f"Entrenamiento fallido: {e}")
            raise

    def predict_proba(self, text: str) -> float:
        """Predict probability for a single text."""
        return self.model.predict_proba([text])[0, 1]

    def save(self, filepath: str):
        """Save the model"""
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, 'wb') as f:
                pickle.dump(self.model, f)
            logger.info(f"Model saved successfully to {filepath}")
        except Exception as e:
            logger.error(f"Model saving failed: {e}")
            raise

def train_hate_speech_model():
    try:
        # Check dataset path
        dataset_path = 'data/processed/train_augmented_dataset.csv'
        
        if not os.path.exists(dataset_path):
            logger.error(f"Dataset not found at {dataset_path}")
            print(f"Error: Dataset not found at {dataset_path}")
            print("Current working directory:", os.getcwd())
            print("Available files in current directory:", os.listdir())
            return
        
        # Load dataset
        data = pd.read_csv(dataset_path)
        
        # Validate dataset
        logger.info(f"Dataset loaded. Total records: {len(data)}")
        logger.info(f"Columns: {list(data.columns)}")
        
        if 'Text' not in data.columns or 'IsHatespeech' not in data.columns:
            logger.error("Dataset missing required columns 'Text' or 'IsHatespeech'")
            print("Error: Dataset missing required columns")
            return
        
        model = HateSpeechModel()
        results = model.train(data['Text'], data['IsHatespeech'])
        
        # Imprimir resultados de overfitting
        print("\nMétricas de Overfitting:")
        print("Diferencia de Precisión (Weighted Avg):", results['overfitting_metrics']['precision_diff']['weighted'])
        print("Diferencia de Recall (Weighted Avg):", results['overfitting_metrics']['recall_diff']['weighted'])
        print("Diferencia de F1-Score (Weighted Avg):", results['overfitting_metrics']['f1_diff']['weighted'])
        print("Diferencia de Exactitud:", results['overfitting_metrics']['accuracy_diff'])

        # Guardar modelo 
        os.makedirs('models', exist_ok=True)
        model.save('models/hate_speech_spacy_model.pkl')

        print("\nModelo entrenado exitosamente!")
        print("\nInforme de Clasificación (Entrenamiento):")
        print(results['train_classification_report'])
        print("\nInforme de Clasificación (Prueba):")
        print(results['test_classification_report'])

    except Exception as e:
        logger.error(f"Error inesperado durante el entrenamiento del modelo: {e}")
        print(f"Ocurrió un error: {e}")

if __name__ == "__main__":
    train_hate_speech_model()