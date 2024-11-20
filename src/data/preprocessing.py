import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

class TextPreprocessor:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
    
    def clean_text(self, text: str) -> str:
        """Clean and preprocess text"""
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenization
        tokens = word_tokenize(text)
        
        # Remove stopwords and lemmatize
        tokens = [
            self.lemmatizer.lemmatize(token)
            for token in tokens
            if token not in self.stop_words
        ]
        
        # Join tokens back
        return ' '.join(tokens)
    
    def process_batch(self, texts: list) -> list:
        """Process a batch of texts"""
        return [self.clean_text(text) for text in texts]

class CommentProcessor:
    def __init__(self):
        self.text_preprocessor = TextPreprocessor()
    
    def extract_metadata(self, comment: dict) -> dict:
        """Extract metadata from a YouTube comment"""
        return {
            'comment_id': comment.get('id', ''),
            'author': comment.get('author', ''),
            'timestamp': comment.get('publishedAt', ''),
            'likes': comment.get('likeCount', 0),
            'replies': comment.get('replyCount', 0)
        }
    
    def process_comment(self, comment: dict) -> dict:
        """Process a single YouTube comment"""
        text = comment.get('text', '')
        processed_text = self.text_preprocessor.clean_text(text)
        metadata = self.extract_metadata(comment)
        
        return {
            'raw_text': text,
            'processed_text': processed_text,
            **metadata
        }
    
    def process_comments(self, comments: list) -> list:
        """Process a batch of YouTube comments"""
        return [self.process_comment(comment) for comment in comments]