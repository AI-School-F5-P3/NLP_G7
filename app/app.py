import streamlit as st
import pandas as pd
from db import connect_db, init_db
from googleapiclient.discovery import build
from datetime import datetime
import plotly.express as px
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from nltk.corpus import stopwords
import nltk

# Asegurar que tenemos las stopwords
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    """Preprocesa el texto de la misma manera que en el entrenamiento."""
    if not isinstance(text, str):
        return ""
    
    # Convertir a minúsculas
    text = text.lower()
    # Reemplazar puntuación con espacios
    text = re.sub(r'[^\w\s]', ' ', text)
    # Dividir por espacios y filtrar tokens vacíos
    tokens = [token.strip() for token in text.split()]
    # Filtrar stop words
    tokens = [token for token in tokens if token not in stop_words]
    return ' '.join(tokens)

def predict_text(text, model_dict):
    """Realiza la predicción de un texto usando el modelo y vectorizador cargados."""
    try:
        # Extraer modelo y vectorizador
        model = model_dict['model']
        vectorizer = model_dict['vectorizer']
        
        # Preprocesar texto
        processed_text = preprocess_text(text)
        
        # Vectorizar
        text_vectorized = vectorizer.transform([processed_text])
        
        # Realizar predicción
        prediction = model.predict_proba(text_vectorized)
        probability = prediction[0][1]  # Probabilidad de la clase positiva
        
        return probability
    except Exception as e:
        st.error(f"Error en la predicción: {str(e)}")
        return None

def get_youtube_comments(video_url, api_key):
    """Obtiene los comentarios de un video de YouTube."""
    try:
        # Extraer video ID de la URL
        if 'v=' in video_url:
            video_id = video_url.split('v=')[1].split('&')[0]
        else:
            st.error("URL de video inválida")
            return None
            
        youtube = build('youtube', 'v3', developerKey=api_key)
        
        comments = []
        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            textFormat="plainText",
            maxResults=100
        )
        
        while request and len(comments) < 500:  # Límite de 500 comentarios
            response = request.execute()
            
            for item in response['items']:
                comment = item['snippet']['topLevelComment']['snippet']
                comments.append({
                    'author': comment['authorDisplayName'],
                    'text': comment['textDisplay'],
                    'date': comment['publishedAt'],
                    'likes': comment['likeCount']
                })
            
            if 'nextPageToken' in response:
                request = youtube.commentThreads().list(
                    part="snippet",
                    videoId=video_id,
                    textFormat="plainText",
                    maxResults=100,
                    pageToken=response['nextPageToken']
                )
            else:
                break
                
        return comments, video_id
    except Exception as e:
        st.error(f"Error al obtener comentarios: {str(e)}")
        return None, None

def save_analysis_to_db(video_id, comments_analysis):
    """Guarda los resultados del análisis en la base de datos."""
    try:
        conn = connect_db()
        if not conn:
            st.error("No se pudo conectar a la base de datos")
            return
            
        cursor = conn.cursor()
        
        # Guardar en ambas tablas
        for analysis in comments_analysis:
            # Convertir numpy.bool_ a Python bool nativo
            is_hate = bool(analysis['hate_probability'] > 0.5)
            
            # Asegurar que hate_probability es un float nativo de Python
            hate_prob = float(analysis['hate_probability'])
            
            # Guardar en la tabla comments
            cursor.execute('''
                INSERT INTO comments (text, is_hate, video_id)
                VALUES (%s, %s, %s)
                RETURNING id
            ''', (
                analysis['text'],
                is_hate,  # Usando el bool nativo
                video_id
            ))
            
            # Guardar en la tabla comment_analysis
            cursor.execute('''
                INSERT INTO comment_analysis 
                (video_id, author, comment_text, hate_probability)
                VALUES (%s, %s, %s, %s)
            ''', (
                video_id,
                analysis['author'],
                analysis['text'],
                hate_prob  # Usando el float nativo
            ))
        
        conn.commit()
        conn.close()
        st.success("Análisis guardado exitosamente en la base de datos")
    except Exception as e:
        st.error(f"Error al guardar en la base de datos: {str(e)}")
        # Añadir más información para debugging
        print(f"Error completo: {str(e)}")
        print(f"Tipo de video_id: {type(video_id)}")
        print(f"Ejemplo de análisis: {comments_analysis[0] if comments_analysis else 'No hay análisis'}")

def get_analysis_history():
    """Obtiene el historial de análisis de la base de datos."""
    try:
        conn = connect_db()
        if not conn:
            st.error("No se pudo conectar a la base de datos")
            return pd.DataFrame()
            
        query = """
            SELECT 
                ca.video_id,
                COUNT(*) as total_comments,
                AVG(ca.hate_probability) as avg_hate_prob,
                COUNT(CASE WHEN ca.hate_probability > 0.5 THEN 1 END) as hate_comments,
                MAX(ca.analysis_date) as last_analysis
            FROM comment_analysis ca
            GROUP BY ca.video_id
            ORDER BY MAX(ca.analysis_date) DESC
        """
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        # Formatear columnas
        if not df.empty:
            df['avg_hate_prob'] = df['avg_hate_prob'].round(3)
            df['hate_percentage'] = (df['hate_comments'] / df['total_comments'] * 100).round(1)
            
        return df
    except Exception as e:
        st.error(f"Error al obtener el historial: {str(e)}")
        return pd.DataFrame()

def get_video_details(video_id):
    """Obtiene los detalles de un video específico."""
    try:
        conn = connect_db()
        if not conn:
            st.error("No se pudo conectar a la base de datos")
            return pd.DataFrame()
            
        query = """
            SELECT 
                author,
                comment_text,
                hate_probability,
                analysis_date
            FROM comment_analysis
            WHERE video_id = %s
            ORDER BY hate_probability DESC
        """
        
        df = pd.read_sql_query(query, conn, params=[video_id])
        conn.close()
        
        if not df.empty:
            df['hate_probability'] = df['hate_probability'].round(3)
            
        return df
    except Exception as e:
        st.error(f"Error al obtener detalles del video: {str(e)}")
        return pd.DataFrame()

def show_analysis_results(comments_analysis):
    """Muestra los resultados del análisis con visualizaciones."""
    if not comments_analysis:
        st.warning("No hay comentarios para analizar")
        return
        
    # Convertir a DataFrame para facilitar el análisis
    df = pd.DataFrame(comments_analysis)
    
    # Estadísticas básicas
    total_comments = len(df)
    hate_comments = len(df[df['hate_probability'] > 0.5])
    hate_percentage = (hate_comments / total_comments * 100) if total_comments > 0 else 0
    
    # Mostrar métricas principales
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total de Comentarios", total_comments)
    with col2:
        st.metric("Comentarios de Odio", hate_comments)
    with col3:
        st.metric("Porcentaje de Odio", f"{hate_percentage:.1f}%")
    
    # Gráfico de distribución
    fig_hist = px.histogram(
        df,
        x='hate_probability',
        nbins=20,
        title="Distribución de Probabilidades de Discurso de Odio",
        labels={'hate_probability': 'Probabilidad', 'count': 'Cantidad de Comentarios'}
    )
    st.plotly_chart(fig_hist)
    
    # Top autores con comentarios de odio
    if not df.empty:
        hate_authors = df[df['hate_probability'] > 0.5]['author'].value_counts().head(10)
        if not hate_authors.empty:
            fig_bar = px.bar(
                hate_authors,
                title="Top 10 Usuarios con Comentarios de Odio",
                labels={'index': 'Usuario', 'value': 'Cantidad de Comentarios'}
            )
            st.plotly_chart(fig_bar)
    
    # Tabla de comentarios
    st.subheader("Comentarios Analizados")
    df['Clasificación'] = df['hate_probability'].apply(
        lambda x: "Discurso de Odio" if x > 0.5 else "Normal"
    )
    st.dataframe(
        df[['author', 'text', 'hate_probability', 'Clasificación']]
        .sort_values('hate_probability', ascending=False)
    )

def main():
    st.set_page_config(page_title="Detector de Mensajes de Odio en YouTube", layout="wide")
    st.title("Detector de Mensajes de Odio en YouTube")
    
    # Cargar modelo
    try:
        with open('../models/modelo_hate_speech.pkl', 'rb') as file:
            model_dict = pickle.load(file)
    except Exception as e:
        st.error(f"Error al cargar el modelo: {str(e)}")
        return
    
    # Sidebar para navegación
    page = st.sidebar.selectbox(
        "Selecciona una opción",
        ["Analizar Texto", "Analizar Video", "Historial de Análisis"]
    )
    
    if page == "Analizar Texto":
        st.header("Análisis de Texto Individual")
        text_input = st.text_area("Ingresa el texto a analizar:")
        
        if st.button("Analizar"):
            if text_input:
                probability = predict_text(text_input, model_dict)
                if probability is not None:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric(
                            "Probabilidad de mensaje de odio",
                            f"{probability:.2%}"
                        )
                    with col2:
                        if probability > 0.5:
                            st.error("⚠️ Este texto contiene discurso de odio")
                        else:
                            st.success("✅ Este texto parece seguro")
                    
                    # Guardar en la base de datos
                    analysis = [{
                        'author': 'Usuario Individual',
                        'text': text_input,
                        'hate_probability': probability
                    }]
                    save_analysis_to_db('texto_individual', analysis)
    
    elif page == "Analizar Video":
        st.header("Análisis de Video de YouTube")
        
        api_key = st.text_input("Ingresa tu API Key de YouTube:", type="password")
        video_url = st.text_input("Ingresa la URL del video de YouTube:")
        
        if st.button("Analizar Video"):
            if not api_key or not video_url:
                st.error("Por favor, ingresa tanto la API Key como la URL del video")
                return
            
            with st.spinner('Obteniendo y analizando comentarios...'):
                comments, video_id = get_youtube_comments(video_url, api_key)
                
                if comments and video_id:
                    st.info(f"Analizando {len(comments)} comentarios...")
                    
                    # Analizar comentarios
                    for comment in comments:
                        probability = predict_text(comment['text'], model_dict)
                        comment['hate_probability'] = probability if probability is not None else 0.0
                    
                    # Guardar resultados
                    save_analysis_to_db(video_id, comments)
                    
                    # Mostrar resultados
                    show_analysis_results(comments)
    
    elif page == "Historial de Análisis":
        st.header("Historial de Análisis")
        
        history_df = get_analysis_history()
        if not history_df.empty:
            st.dataframe(
                history_df,
                columns=[
                    'video_id',
                    'total_comments',
                    'hate_comments',
                    'hate_percentage',
                    'avg_hate_prob',
                    'last_analysis'
                ]
            )
            
            # Ver detalles de un video específico
            selected_video = st.selectbox(
                "Selecciona un video para ver detalles:",
                history_df['video_id'].tolist()
            )
            
            if selected_video:
                details_df = get_video_details(selected_video)
                if not details_df.empty:
                    st.subheader(f"Detalles del video {selected_video}")
                    st.dataframe(details_df)
        else:
            st.info("No hay análisis previos registrados")

if __name__ == "__main__":
    init_db()  # Inicializar las tablas si no existen
    main()