import pickle
import re
import nltk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score, train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
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
        
    def calculate_overfitting(self, cv_scores: np.ndarray) -> float:
        """Calculate overfitting using cross-validation scores."""
        mean_cv_score = np.mean(cv_scores)
        std_cv_score = np.std(cv_scores)
        
        # Track variance as a measure of overfitting
        overfitting = std_cv_score
        self.metrics['cv_scores'] = cv_scores.tolist()
        self.metrics['mean_cv_score'] = float(mean_cv_score)
        self.metrics['cv_score_variance'] = float(std_cv_score)
        
        return overfitting
        
    def save_report(self, report_dir: str) -> None:
        os.makedirs(report_dir, exist_ok=True)
        
        # Convert numpy metrics to native Python types
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

class HateSpeechModel:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=3000,  # Reduced features
            ngram_range=(1, 2),
            min_df=3,  # Increased minimum document frequency
            stop_words='english'  # Built-in stop words
        )
        
        # More regularized base models
        rf1 = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=5,
            max_features='sqrt',  # Feature sampling
            class_weight='balanced',
            random_state=1
        )
        
        rf2 = RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            min_samples_split=12,
            min_samples_leaf=4,
            max_features='log2',
            class_weight='balanced_subsample',
            random_state=2
        )
        
        lr = LogisticRegression(
            penalty='l2',
            C=0.1,  # More regularization
            solver='liblinear',
            class_weight='balanced',
            random_state=3
        )
        
        xgb = XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.7,
            colsample_bytree=0.7,
            reg_alpha=0.1,  # L1 regularization
            reg_lambda=0.1,  # L2 regularization
            scale_pos_weight=7,
            random_state=4
        )
        
        self.model = VotingClassifier(
            estimators=[
                ('rf1', rf1),
                ('rf2', rf2),
                ('lr', lr),
                ('xgb', xgb)
            ],
            voting='soft',
            weights=[1, 1, 1, 2]
        )
        
        self.resampler = SMOTETomek(random_state=42)
        self.evaluation_report = ModelEvaluationReport("hate_speech_classifier_aug_over")

    def tokenize_text(self, text: str) -> list:
        """Enhanced tokenization with more preprocessing."""
        text = text.lower()
        # Remove special characters and extra whitespaces
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        tokens = [token.strip() for token in text.split()]
        tokens = [token for token in tokens if token not in stop_words and len(token) > 1]
        return tokens

    def preprocess_text(self, text: str) -> str:
        """More robust text preprocessing."""
        if not isinstance(text, str):
            return ""
        tokens = self.tokenize_text(text)
        return ' '.join(tokens)

    def train(self, X: pd.Series, y: pd.Series) -> None:
        """Enhanced training with cross-validation and feature selection."""
        print("Starting model training...")
        
        # Preprocess texts
        X_processed = [self.preprocess_text(text) for text in X]
        
        # Vectorize texts
        X_vectorized = self.vectorizer.fit_transform(X_processed)
        
        # Stratified K-Fold for robust cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        # Perform cross-validation
        cv_scores = cross_val_score(self.model, X_vectorized, y, cv=cv, scoring='balanced_accuracy')
        
        # Calculate overfitting metric
        overfitting = self.evaluation_report.calculate_overfitting(cv_scores)
        print(f"\nCross-Validation Scores: {cv_scores}")
        print(f"Mean CV Score: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        print(f"Overfitting Variance: {overfitting:.4f}")
        
        # Resample data
        X_resampled, y_resampled = self.resampler.fit_resample(X_vectorized, y)
        
        # Final model training
        self.model.fit(X_resampled, y_resampled)
        
        # Evaluation
        y_pred = self.model.predict(X_vectorized)
        y_pred_proba = self.model.predict_proba(X_vectorized)[:, 1]
        
        # Calculate metrics
        class_report = classification_report(y, y_pred, output_dict=True)
        cm = confusion_matrix(y, y_pred)
        fpr, tpr, _ = roc_curve(y, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        # Save metrics
        self.evaluation_report.add_metric('classification_report', class_report)
        self.evaluation_report.add_metric('confusion_matrix', cm.tolist())
        self.evaluation_report.add_metric('roc_auc', roc_auc)

    def predict_proba(self, text: str) -> float:
        """Prediction with preprocessing."""
        processed_text = self.preprocess_text(text)
        vectorized_text = self.vectorizer.transform([processed_text])
        return self.model.predict_proba(vectorized_text)[0, 1]

    def save(self, filepath: str) -> None:
        """Save trained model and vectorizer."""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'vectorizer': self.vectorizer
            }, f)

def train_and_save_model():
    os.makedirs('models', exist_ok=True)
    os.makedirs('reports', exist_ok=True)
    
    print("Loading training data...")
    train_data = pd.read_csv('data/processed/train_augmented_dataset.csv')
    
    print("\nOriginal dataset distribution:")
    print(train_data['IsHatespeech'].value_counts(normalize=True))
    
    model = HateSpeechModel()
    model.train(train_data['Text'], train_data['IsHatespeech'])
    
    # Save model
    model.save('models/hspeech_model_aug.pkl')
    
    # Save evaluation report
    model.evaluation_report.save_report('reports')
    
    print("\nModel trained and saved successfully.")
    print("Evaluation report saved in 'reports' folder.")

if __name__ == "__main__":
    train_and_save_model()