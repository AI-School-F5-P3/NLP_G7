import pickle
import re
import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
import os

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

stop_words = list(stopwords.words('english'))
additional_stop_words = ['black', 'white', 'color', 'dark', 'light']
stop_words.extend(additional_stop_words)

class HateSpeechModel:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=2000,
            ngram_range=(1, 3),
            min_df=5,
            max_df=0.7,
            stop_words=stop_words
        )
        
        self.rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            min_samples_split=8,
            min_samples_leaf=4,
            max_features='sqrt',
            class_weight='balanced',
            random_state=42
        )
        
        self.xgb = XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=1,
            reg_lambda=2,
            scale_pos_weight=7,
            min_child_weight=3,
            gamma=1,
            random_state=42
        )
        self.xgb.set_params(eval_metric='logloss')
        
        self.lr = LogisticRegression(
            C=1,
            penalty='l2',
            class_weight='balanced',
            random_state=42,
            max_iter=1000
        )
        
        self.model = VotingClassifier(
            estimators=[
                ('rf', self.rf),
                ('xgb', self.xgb),
                ('lr', self.lr)
            ],
            voting='soft',
            weights=[1, 1, 1]
        )

    def tokenize_text(self, text):
        if not isinstance(text, str):
            return []
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        tokens = [token.strip() for token in text.split()]
        return [token for token in tokens if token not in stop_words and len(token) > 2]

    def preprocess_text(self, text):
        tokens = self.tokenize_text(text)
        return ' '.join(tokens) if tokens else ""

    def train(self, X, y):
        print("Starting model training...")
        
        print("Preprocessing texts...")
        X_processed = [self.preprocess_text(text) for text in X]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print("Vectorizing texts...")
        X_train_vectorized = self.vectorizer.fit_transform(X_train)
        X_test_vectorized = self.vectorizer.transform(X_test)
        
        print("Applying SMOTE...")
        X_train_resampled, y_train_resampled = SMOTE(random_state=42).fit_resample(X_train_vectorized, y_train)
        
        print("Tuning model hyperparameters...")
        param_grid = {
            'rf__n_estimators': [80, 100, 120],
            'rf__max_depth': [6, 8, 10],
            'xgb__n_estimators': [80, 100, 120],
            'xgb__max_depth': [4, 5, 6],
            'xgb__learning_rate': [0.03, 0.05, 0.07],
            'lr__C': [0.8, 1, 1.2]
        }
        
        grid_search = GridSearchCV(self.model, param_grid, cv=5, n_jobs=-1, verbose=1)
        grid_search.fit(X_train_resampled, y_train_resampled)
        self.model = grid_search.best_estimator_
        
        train_score = self.model.score(X_train_vectorized, y_train)
        test_score = self.model.score(X_test_vectorized, y_test)
        overfitting = train_score - test_score
        
        print("\nOverfitting Analysis:")
        print(f"Training score: {train_score:.4f}")
        print(f"Test score: {test_score:.4f}")
        print(f"Degree of overfitting: {overfitting:.4f}")
        
        if overfitting < 0.03:
            print("\nModel meets the overfitting requirement.")
        else:
            print("\nModel does not meet the overfitting requirement. Further tuning may be needed.")
        
        print("\nFinal Evaluation:")
        y_pred = self.model.predict(X_test_vectorized)
        print(classification_report(y_test, y_pred))
        
        return self.model

    def predict_proba(self, text):
        processed_text = self.preprocess_text(text)
        vectorized_text = self.vectorizer.transform([processed_text])
        return self.model.predict_proba(vectorized_text)[0, 1]

    def save(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'vectorizer': self.vectorizer
            }, f)

def train_and_save_model():
    os.makedirs('models', exist_ok=True)
    
    print("Loading training data...")
    train_data = pd.read_csv('data/processed/train_data.csv')
    
    model = HateSpeechModel()
    model.train(train_data['Text'], train_data['IsHatespeech'])
    
    model.save('models/optimized_hate_speech_model.pkl')
    print("\nOptimized model trained and saved successfully.")

if __name__ == "__main__":
    train_and_save_model()