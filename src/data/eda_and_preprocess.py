import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import nltk
from nltk.corpus import stopwords
from collections import Counter
import re
import unidecode

def load_and_explore_data(filepath):
    """Carga y realiza el análisis exploratorio inicial de los datos."""
    # Cargar datos
    df = pd.read_csv(filepath)
    
    # Información básica del dataset
    print("Información básica del dataset:")
    print(df.info())
    print("\nEstadísticas descriptivas:")
    print(df.describe())
    
    # Análisis de valores faltantes
    missing_values = df.isnull().sum()
    print("\nValores faltantes por columna:")
    print(missing_values)
    
    # Distribución de clases
    plot_class_distribution(df)
        
    return df

def plot_class_distribution(df):
    """Grafica la distribución de las clases (discurso de odio / normal)"""
    plt.figure(figsize=(8, 6))
    df['IsHatespeech'].map({True: 'Discurso de odio', False: 'Normal'}).value_counts().plot(kind='bar')
    plt.title('Distribución de Clases')
    plt.xlabel('Categoría')
    plt.ylabel('Cantidad')
    plt.savefig('reports/figures/class_distribution.png')
    plt.close()

def preprocess_text(text):
    """Preprocesa el texto eliminando caracteres especiales, tildes y convirtiendo a minúsculas"""
    text = text.lower()
    text = unidecode.unidecode(text)
    text = re.sub(r'[^\w\s]', '', text)
    return text

def get_words(text, stop_words):
    """Obtiene las palabras más comunes en un texto, excluyendo las stopwords"""
    word_counts = Counter([word for word in text.split() if word not in stop_words])
    return pd.Series(word_counts).sort_values(ascending=False)

def preprocess_and_split_data(df):
    """Preprocesa y divide los datos en conjuntos de entrenamiento y validación."""
    # Limpiar textos
    df['Text'] = df['Text'].apply(preprocess_text)
    
    # Separar conjunto de validación final (200 registros)
    train_df, validation_df = train_test_split(
        df, 
        test_size=200,
        stratify=df['IsHatespeech'],
        random_state=42
    )
    
    # Del resto, tomar 800 registros para entrenamiento
    if len(train_df) > 800:
        train_df, _ = train_test_split(
            train_df,
            train_size=800,
            stratify=train_df['IsHatespeech'],
            random_state=42
        )
    
    # Guardar los conjuntos de datos
    train_df.to_csv('data/processed/train_data.csv', index=False)
    validation_df.to_csv('data/processed/validation_data.csv', index=False)
    
    print(f"Conjunto de entrenamiento guardado: {len(train_df)} registros")
    print(f"Conjunto de validación guardado: {len(validation_df)} registros")
    
    return train_df, validation_df

def text_analysis(df):
    """Realiza análisis específico del texto."""
      
    # Palabras más comunes
    nltk.download('stopwords', quiet=True)
    stop_words = set(stopwords.words('english'))
    
    words_hate = df[df['IsHatespeech']]['Text'].apply(lambda x: get_words(x, stop_words))
    words_normal = df[~df['IsHatespeech']]['Text'].apply(lambda x: get_words(x, stop_words))
    
    # Visualizar palabras más comunes
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    words_hate.iloc[:10].plot(kind='bar')
    plt.title('Palabras más comunes en discurso de odio')
    plt.subplot(1, 2, 2)
    words_normal.iloc[:10].plot(kind='bar')
    plt.title('Palabras más comunes en texto normal')
    plt.tight_layout()
    plt.savefig('reports/figures/common_words.png')
    plt.close()

def main():
    # Cargar y explorar datos
    df = load_and_explore_data('data/raw/youtoxic_english_1000.csv')
    
    # Preprocesar y dividir datos
    train_df, validation_df = preprocess_and_split_data(df)
    
    # Análisis de texto
    text_analysis(train_df)
    
    print("\nProceso de EDA y preprocesamiento completado.")
    print("Se han generado visualizaciones en archivos PNG en la carpeta 'reports/figures'.")
    print("Los datos han sido divididos y guardados en archivos CSV en la carpeta 'data/processed'.")

if __name__ == "__main__":
    main()