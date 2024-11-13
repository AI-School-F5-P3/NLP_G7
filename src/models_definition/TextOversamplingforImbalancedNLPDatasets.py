import os
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

class TextOversampler:
    def __init__(self, dataset, target_column, model_path='gpt2'):
        self.dataset = dataset
        self.target_column = target_column
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = GPT2LMHeadModel.from_pretrained(model_path).to(self.device)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        self.pad_token_id = self.tokenizer.eos_token_id

    def oversample(self, minority_class, target_ratio):
        minority_indices = self.dataset[self.target_column] == minority_class
        minority_text = self.dataset.loc[minority_indices, self.target_column].tolist()

        oversampled_text = []
        for text in minority_text:
            if isinstance(text, str):
                input_ids = self.tokenizer.encode(text, return_tensors='pt').to(self.device)
                attention_mask = torch.ones_like(input_ids)
            else:
                input_ids = self.tokenizer.encode(str(text), return_tensors='pt').to(self.device)
                attention_mask = torch.ones_like(input_ids)

            output = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                num_return_sequences=int(target_ratio),
                max_length=len(input_ids[0]) + 100,
                num_beams=5,
                early_stopping=True,
                pad_token_id=self.pad_token_id
            )
            oversampled_text.extend([self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output])

        oversampled_df = pd.DataFrame({self.target_column: oversampled_text})
        oversampled_df[self.target_column] = minority_class

        balanced_dataset = pd.concat([self.dataset, oversampled_df], ignore_index=True)
        return balanced_dataset

    def train_test_split(self, test_size=0.2, random_state=42):
        X_train, X_test, y_train, y_test = train_test_split(
            self.dataset.drop(self.target_column, axis=1),
            self.dataset[self.target_column],
            test_size=test_size,
            random_state=random_state,
            stratify=self.dataset[self.target_column]
        )
        return X_train, X_test, y_train, y_test

def save_balanced_dataset(balanced_dataset, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    balanced_dataset.to_csv(output_path, index=False)
    print(f"Balanced dataset saved to: {output_path}")

if __name__ == "__main__":
    # Load your dataset
    dataset = pd.read_csv('data/processed/train_data.csv')

    # Initialize the oversampler
    oversampler = TextOversampler(dataset, 'IsHatespeech')

    # Oversample the minority class to achieve a target ratio
    balanced_dataset = oversampler.oversample(minority_class=0, target_ratio=1)

    # Save the balanced dataset
    save_balanced_dataset(balanced_dataset, 'data/processed/balanced_train_data.csv')

    # Split the balanced dataset into training and test sets
    X_train, X_test, y_train, y_test = oversampler.train_test_split()