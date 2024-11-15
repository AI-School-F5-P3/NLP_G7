"""Enhanced feature extraction and analysis module for text classification"""

import numpy as np
import pandas as pd
from typing import Dict, List, Union
import nltk
from textblob import TextBlob
import spacy
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_selection import mutual_info_classif
import scipy.stats as stats

class EnhancedFeatureExtractor:
    """Enhanced feature extraction with robust error handling and optimized processing"""
    
    def __init__(self):
        """Initialize the feature extractor with necessary models and tokenizers"""
        # Initialize tokenizer with comprehensive pattern
        self.tokenizer = RegexpTokenizer(r'\w+|\$[\d\.]+|\S+')
        
        # Initialize spaCy
        try:
            self.nlp = spacy.load('en_core_web_sm')
            print("spaCy model loaded successfully")
        except OSError:
            print("Downloading spaCy model...")
            import os
            os.system('python -m spacy download en_core_web_sm')
            self.nlp = spacy.load('en_core_web_sm')
    
    def extract_enhanced_features(self, text: str) -> Dict[str, float]:
        """Extract comprehensive set of text features with error handling"""
        if not isinstance(text, str):
            return self._get_empty_features()
            
        try:
            features = {}
            features.update(self._get_basic_features(text))
            features.update(self._get_sentiment_features(text))
            features.update(self._get_linguistic_features(text))
            return features
        except Exception as e:
            print(f"Warning: Error extracting features: {str(e)}")
            return self._get_empty_features()
    
    def _get_empty_features(self) -> Dict[str, float]:
        """Return dictionary with zero values for all features"""
        return {
            'avg_word_length': 0.0,
            'token_count': 0.0,
            'unique_pos_tags': 0.0,
            'noun_ratio': 0.0,
            'verb_ratio': 0.0,
            'adj_ratio': 0.0,
            'entity_ratio': 0.0,
            'sentiment_polarity': 0.0,
            'sentiment_subjectivity': 0.0
        }
    
    def _get_basic_features(self, text: str) -> Dict[str, float]:
        """Extract basic text features using regex tokenizer"""
        try:
            tokens = self.tokenizer.tokenize(text.lower())
            return {
                'avg_word_length': np.mean([len(token) for token in tokens]) if tokens else 0.0,
                'token_count': float(len(tokens))
            }
        except Exception as e:
            print(f"Warning: Error in basic feature extraction: {str(e)}")
            return {
                'avg_word_length': 0.0,
                'token_count': 0.0
            }
    
    def _get_sentiment_features(self, text: str) -> Dict[str, float]:
        """Extract sentiment features using TextBlob"""
        try:
            blob = TextBlob(text)
            return {
                'sentiment_polarity': blob.sentiment.polarity,
                'sentiment_subjectivity': blob.sentiment.subjectivity
            }
        except Exception as e:
            print(f"Warning: Error in sentiment analysis: {str(e)}")
            return {
                'sentiment_polarity': 0.0,
                'sentiment_subjectivity': 0.0
            }
    
    def _get_linguistic_features(self, text: str) -> Dict[str, float]:
        """Extract linguistic features using spaCy"""
        try:
            # Limit text length for memory efficiency
            doc = self.nlp(text[:1000000])
            total_tokens = len(doc) if len(doc) > 0 else 1
            
            return {
                'unique_pos_tags': len(set([token.pos_ for token in doc])),
                'noun_ratio': len([token for token in doc if token.pos_ == 'NOUN']) / total_tokens,
                'verb_ratio': len([token for token in doc if token.pos_ == 'VERB']) / total_tokens,
                'adj_ratio': len([token for token in doc if token.pos_ == 'ADJ']) / total_tokens,
                'entity_ratio': len(doc.ents) / total_tokens
            }
        except Exception as e:
            print(f"Warning: Error in linguistic feature extraction: {str(e)}")
            return {
                'unique_pos_tags': 0.0,
                'noun_ratio': 0.0,
                'verb_ratio': 0.0,
                'adj_ratio': 0.0,
                'entity_ratio': 0.0
            }

def calculate_feature_importance(df: pd.DataFrame, 
                               category: str, 
                               features: List[str]) -> np.ndarray:
    """Calculate feature importance using mutual information with proper error handling"""
    try:
        # Ensure category exists
        if category not in df.columns:
            raise KeyError(f"Category {category} not found in DataFrame")
            
        # Prepare features
        X = df[features].fillna(0).astype(np.float32)
        y = df[category].astype(int)
        
        # Calculate importance
        importances = mutual_info_classif(X, y, random_state=42)
        return pd.Series(importances, index=features)
        
    except Exception as e:
        print(f"Error calculating feature importance for {category}: {str(e)}")
        return pd.Series(0, index=features)

def analyze_feature_stability(df: pd.DataFrame, 
                            toxic_categories: List[str],
                            features: List[str],
                            n_iterations: int = 10) -> pd.DataFrame:
    """Analyze feature importance stability"""
    # Initialize results DataFrame
    stability_matrix = pd.DataFrame(0.0, 
                                  index=toxic_categories,
                                  columns=features)
    
    print(f"Analyzing stability for {len(features)} features...")
    
    for category in toxic_categories:
        print(f"\nProcessing category: {category}")
        importance_scores = []
        
        for i in range(n_iterations):
            # Sample 80% of data
            n_samples = int(0.8 * len(df))
            sample_indices = np.random.choice(len(df), n_samples, replace=False)
            sample_df = df.iloc[sample_indices]
            
            # Calculate importance for this sample
            importance = calculate_feature_importance(sample_df, category, features)
            importance_scores.append(importance)
        
        # Calculate stability as std of importance scores
        if importance_scores:
            importance_array = np.array(importance_scores)
            stability = np.std(importance_array, axis=0)
            stability_matrix.loc[category] = stability
    
    return stability_matrix

def analyze_cross_category_interactions(df: pd.DataFrame, 
                                     toxic_categories: List[str]) -> pd.DataFrame:
    """Analyze interactions between toxicity categories with proper type handling"""
    try:
        interactions = pd.DataFrame(0.0, 
                                  index=toxic_categories, 
                                  columns=toxic_categories,
                                  dtype=np.float32)
        
        for cat1 in toxic_categories:
            for cat2 in toxic_categories:
                if cat1 != cat2:
                    try:
                        joint_prob = (df[cat1].astype(bool) & df[cat2].astype(bool)).mean()
                        base_prob = df[cat1].astype(bool).mean()
                        
                        cond_prob = float(joint_prob / base_prob) if base_prob > 0 else 0.0
                        interactions.loc[cat1, cat2] = cond_prob
                        
                    except Exception as e:
                        print(f"Error calculating interaction {cat1}-{cat2}: {str(e)}")
                        interactions.loc[cat1, cat2] = 0.0
        
        return interactions
    except Exception as e:
        print(f"Error in cross-category analysis: {str(e)}")
        return pd.DataFrame()

# Module testing
if __name__ == "__main__":
    try:
        extractor = EnhancedFeatureExtractor()
        print("Enhanced features module loaded successfully!")
        
        # Test feature extraction
        sample_text = "This is a test sentence."
        features = extractor.extract_enhanced_features(sample_text)
        print("\nFeature extraction test successful!")
        print("Sample features:", features)
        
    except Exception as e:
        print(f"Error in module initialization: {str(e)}")