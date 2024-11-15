import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = DATA_DIR / "models"

# Ensure directories exist
for dir_path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Model parameters
MODEL_PARAMS = {
    'vectorizer': {
        'max_features': 5000,
        'ngram_range': (1, 2)
    },
    'classifier': {
        'random_state': 42,
        'n_jobs': -1
    }
}

# Toxic categories
TOXIC_CATEGORIES = [
    'IsToxic', 'IsAbusive', 'IsThreat', 'IsProvocative',
    'IsObscene', 'IsHatespeech', 'IsRacist', 'IsNationalist',
    'IsSexist', 'IsHomophobic', 'IsReligiousHate', 'IsRadicalism'
]

# Prediction threshold
PREDICTION_THRESHOLD = 0.5

# YouTube API settings
YOUTUBE_API_BATCH_SIZE = 50  # Number of comments to fetch per request
MAX_COMMENTS_PER_VIDEO = 1000

# App settings
APP_TITLE = "YouTube Comment Toxicity Analyzer"
APP_DESCRIPTION = """
This application analyzes YouTube comments for various types of toxic content.
Upload a YouTube video URL or input text directly to analyze for toxic content.
"""

# Cache settings
CACHE_TTL = 3600  # Cache time to live in seconds