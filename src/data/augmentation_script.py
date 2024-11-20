import pandas as pd
import nltk
import random
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from googletrans import Translator
from transformers import pipeline

# Descargas de NLTK
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

class EnglishDataAugmentation:
    def __init__(self, dataset_path):
        # Cargar dataset
        self.df = pd.read_csv(dataset_path)
        
        # Cargar modelo de lenguaje para generación
        self.generator = pipeline('text-generation', model='gpt2')
        
        # Inicializar traductor
        self.translator = Translator()
    
    def synonym_replacement(self, text, num_replacements=2):
        """Reemplaza palabras con sinónimos en inglés"""
        tokens = word_tokenize(text)
        replaced_tokens = tokens.copy()
        
        for _ in range(num_replacements):
            for i, token in enumerate(tokens):
                synsets = wordnet.synsets(token)
                if synsets:
                    try:
                        synonym = synsets[0].lemmas()[0].name()
                        replaced_tokens[i] = synonym
                    except:
                        pass
        
        return ' '.join(replaced_tokens)
    
    def back_translation(self, text):
        """Realiza back-translation entre idiomas"""
        try:
            # Traducir a francés
            fr_translation = self.translator.translate(text, dest='fr').text
            
            # Volver al inglés
            en_translation = self.translator.translate(fr_translation, dest='en').text
            
            return en_translation
        except:
            return text
    
    def contextual_perturbation(self, text):
        """Perturba contextualmente el texto"""
        tokens = word_tokenize(text)
        perturbed_tokens = []
        
        for token in tokens:
            # Modificación simple de estructura
            if random.random() < 0.3:
                perturbed_tokens.append(token + '_mod')
            else:
                perturbed_tokens.append(token)
        
        return ' '.join(perturbed_tokens)
    
    def generate_text(self, text, max_length=50):
        """Genera texto similar usando GPT-2"""
        try:
            generated = self.generator(text, max_length=max_length)[0]['generated_text']
            return generated
        except:
            return text
    
    def augment_data(self):
        """Aumenta el dataset aplicando técnicas de augmentación"""
        augmented_data = []
        
        # Separar clase minoritaria
        hate_speech = self.df[self.df['IsHatespeech'] == True]
        
        # Aumentar clase minoritaria
        for _, row in hate_speech.iterrows():
            augmented_rows = []
            
            augmented_rows.append({
                **row.to_dict(),
                'source': 'synonym_replacement',
                'Text': self.synonym_replacement(row['Text'])
            })
            
            augmented_rows.append({
                **row.to_dict(),
                'source': 'back_translation',
                'Text': self.back_translation(row['Text'])
            })
            
            augmented_rows.append({
                **row.to_dict(),
                'source': 'contextual_perturbation',
                'Text': self.contextual_perturbation(row['Text'])
            })
            
            augmented_rows.append({
                **row.to_dict(),
                'source': 'text_generation',
                'Text': self.generate_text(row['Text'])
            })
            
            augmented_data.extend(augmented_rows)
        
        # Combinar datos originales y aumentados
        augmented_df = pd.concat([
            self.df, 
            pd.DataFrame(augmented_data)
        ], ignore_index=True)
        
        # Separar 20 registros para validación
        validation_set = augmented_df.sample(n=20)
        train_set = augmented_df.drop(validation_set.index)
        
        return train_set, validation_set

# Uso del script
augmenter = EnglishDataAugmentation('data/raw/youtoxic_english_1000.csv')
train_data, validation_data = augmenter.augment_data()

# Guardar resultados
train_data.to_csv('train_augmented_dataset.csv', index=False)
validation_data.to_csv('validation_dataset.csv', index=False)