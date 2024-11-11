import psycopg2
import streamlit as st
from dotenv import load_dotenv
import os

load_dotenv()

def connect_db():
    """Establece conexión con la base de datos PostgreSQL."""
    try:
        conn = psycopg2.connect(
            dbname=os.getenv('DB_NAME'),
            user=os.getenv('DB_USER'),
            password=os.getenv('DB_PASSWORD'),
            host=os.getenv('DB_HOST'),
            port=os.getenv('DB_PORT')
        )
        return conn
    except Exception as e:
        st.error(f"Error conectando a la base de datos: {e}")
        return None

def init_db():
    """Inicializa la base de datos con las tablas necesarias."""
    conn = connect_db()
    if conn:
        with conn.cursor() as cur:
            # Tabla de comentarios básica
            cur.execute("""
                CREATE TABLE IF NOT EXISTS comments (
                    id SERIAL PRIMARY KEY,
                    text TEXT NOT NULL,
                    is_hate BOOLEAN,
                    video_id VARCHAR(20),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Tabla para el análisis de comentarios
            cur.execute("""
                CREATE TABLE IF NOT EXISTS comment_analysis (
                    id SERIAL PRIMARY KEY,
                    video_id TEXT,
                    author TEXT,
                    comment_text TEXT,
                    hate_probability REAL,
                    analysis_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
        conn.commit()
        conn.close()