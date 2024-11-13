import unittest
import pickle
import os
import sys
from unittest.mock import patch, MagicMock
from dotenv import load_dotenv

# Obtener el path absoluto del directorio actual (tests)
current_dir = os.path.dirname(os.path.abspath(__file__))
# Obtener el path del directorio raíz del proyecto
project_root = os.path.dirname(current_dir)

# Agregar el directorio raíz y el directorio app al path de Python
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    sys.path.insert(0, os.path.join(project_root, 'app'))

# Cargar variables de entorno
load_dotenv()

# Importar después de configurar el path
try:
    import db
    from db import connect_db, init_db
    print("Módulo db importado correctamente")
except ImportError as e:
    print(f"Error importando db: {e}")
    print(f"Python path: {sys.path}")
    print(f"Contenido del directorio app: {os.listdir(os.path.join(project_root, 'app'))}")
    sys.exit(1)

try:
    from app import predict_text, get_youtube_comments
    print("Módulo app importado correctamente")
except ImportError as e:
    print(f"Error importando app: {e}")
    sys.exit(1)

class TestDB(unittest.TestCase):
    def test_db_connection(self):
        """Test para verificar la conexión a la base de datos"""
        connection = connect_db()
        self.assertIsNotNone(connection)
        connection.close()

class TestHateSpeechModel(unittest.TestCase):
    def setUp(self):
        """Configuración inicial para las pruebas del modelo"""
        model_path = os.path.join(project_root, 'models', 'optimized_hspeech_model_low_overfit.pkl')
        try:
            with open(model_path, 'rb') as file:
                self.model_dict = pickle.load(file)
        except Exception as e:
            raise unittest.SkipTest(f"No se pudo cargar el modelo optimizado desde {model_path}: {str(e)}")
    
    def test_predict_safe_text(self):
        """Test para texto seguro"""
        safe_texts = [
            "Me encanta este video, es muy educativo",
            "Gracias por compartir tu conocimiento",
            "Excelente explicación del tema"
        ]
        for text in safe_texts:
            result = predict_text(text, self.model_dict)
            self.assertLess(result, 0.5, f"El texto '{text}' no debería ser clasificado como odio")
    
    def test_predict_hate_text(self):
        """Test para texto con odio"""
        hate_texts = [
            "Odio este video, es basura",
            "Eres un idiota",
            "Deberías desaparecer"
        ]
        for text in hate_texts:
            result = predict_text(text, self.model_dict)
            self.assertGreater(result, 0.5, f"El texto '{text}' debería ser clasificado como odio")

class TestApp(unittest.TestCase):
    def setUp(self):
        """Configuración inicial para las pruebas de la aplicación"""
        self.api_key = os.getenv('API_YOUTUBE')
        if not self.api_key:
            raise unittest.SkipTest("API_YOUTUBE no encontrada en variables de entorno")

    @patch('app.build')
    def test_get_youtube_comments(self, mock_build):
        """Test para obtener comentarios de YouTube"""
        mock_youtube = MagicMock()
        mock_build.return_value = mock_youtube
        mock_response = {
            'items': [{
                'snippet': {
                    'topLevelComment': {
                        'snippet': {
                            'authorDisplayName': 'Test User',
                            'textDisplay': 'Test comment',
                            'publishedAt': '2024-01-01T00:00:00Z',
                            'likeCount': 0
                        }
                    }
                }
            }]
        }
        mock_youtube.commentThreads().list().execute.return_value = mock_response
        
        comments, video_id = get_youtube_comments('https://youtube.com/watch?v=test')
        
        self.assertIsNotNone(comments)
        self.assertEqual(len(comments), 1)
        self.assertEqual(comments[0]['author'], 'Test User')
    
    def test_invalid_youtube_url(self):
        """Test para URL inválida de YouTube"""
        comments, video_id = get_youtube_comments('https://invalid-url.com')
        self.assertIsNone(comments)
        self.assertIsNone(video_id)

class TestPredictionEdgeCases(unittest.TestCase):
    def setUp(self):
        """Configuración inicial para las pruebas de casos límite"""
        model_path = os.path.join(project_root, 'models', 'optimized_hspeech_model_low_overfit.pkl')
        try:
            with open(model_path, 'rb') as file:
                self.model_dict = pickle.load(file)
        except Exception as e:
            raise unittest.SkipTest(f"No se pudo cargar el modelo optimizado desde {model_path}: {str(e)}")
    
    def test_empty_text(self):
        """Test para texto vacío"""
        result = predict_text("", self.model_dict)
        self.assertIsNotNone(result)
        self.assertLess(result, 0.5, "Texto vacío no debería ser clasificado como odio")
    
    def test_very_long_text(self):
        """Test para texto muy largo"""
        long_text = "Este es un texto muy amable " * 1000
        result = predict_text(long_text, self.model_dict)
        self.assertIsNotNone(result)
        self.assertLess(result, 0.5, "Texto largo amable no debería ser clasificado como odio")
    
    def test_special_characters(self):
        """Test para caracteres especiales"""
        special_chars = "!@#$%^&*()"
        result = predict_text(special_chars, self.model_dict)
        self.assertIsNotNone(result)
        self.assertLess(result, 0.5, "Caracteres especiales no deberían ser clasificados como odio")

if __name__ == '__main__':
    # Verificar la estructura del proyecto
    print("\nVerificando estructura del proyecto:")
    print(f"Existe directorio app: {os.path.exists(os.path.join(project_root, 'app'))}")
    print(f"Existe archivo db.py: {os.path.exists(os.path.join(project_root, 'app', 'db.py'))}")
    print(f"Existe archivo app.py: {os.path.exists(os.path.join(project_root, 'app', 'app.py'))}")
    print(f"Existe archivo __init__.py: {os.path.exists(os.path.join(project_root, 'app', '__init__.py'))}")
    
    # Ejecutar tests
    unittest.main(verbosity=2)