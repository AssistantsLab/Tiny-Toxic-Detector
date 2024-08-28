#
# This file contains a ToxicityClassifier class that can be used to evaluate the performance of a given toxic text classification model on the Jigsaw dataset.
# The user can specify the model name to be tested. If the model has one of the following, this script may not work correctly:
# - If it requires a custom architecture
# - If it has a different tokenizer and model (such as tokenizer 't5-large' and model 'toxicchat-t5-large')
# - If it requires a specific prompting template (such as 'toxicchat-t5-large')
#
# Note: This script requires 'test.csv' and 'test_labels.csv' from the Jigsaw challenge to be in the same folder.
# These files will be included in the GitHub repository.
#
# Author: Michielo
#

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import pandas as pd


class JigsawDataset(Dataset):
    def __init__(self, comments, labels, tokenizer, max_length=512):
        self.comments = comments
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.comments)

    def __getitem__(self, idx):
        comment = self.comments[idx]
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            comment,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class ToxicityClassifier:
    def __init__(self, test_file, labels_file, model_name, batch_size=32, device=None):
        self.test_file = test_file
        self.labels_file = labels_file
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
        self.model, self.tokenizer = self.load_model_and_tokenizer()
        self.comments, self.labels = self.load_data()
        self.dataloader = self.preprocess_data()

    def load_data(self):
        test_df = pd.read_csv(self.test_file)
        labels_df = pd.read_csv(self.labels_file)

        merged_df = pd.merge(test_df, labels_df, on='id')
        filtered_df = merged_df[merged_df['toxic'] != -1]

        return filtered_df['comment_text'].tolist(), filtered_df['toxic'].tolist()

    def load_model_and_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        return model, tokenizer

    def preprocess_data(self):
        dataset = JigsawDataset(self.comments, self.labels, self.tokenizer)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        return dataloader

    def evaluate_model(self):
        self.model.eval()
        self.model.to(self.device)

        correct_predictions = 0
        total_predictions = 0

        progress_bar = tqdm(self.dataloader, desc="Evaluating")

        with torch.no_grad():
            for batch in progress_bar:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.model(input_ids, attention_mask=attention_mask)
                predictions = torch.argmax(outputs.logits, dim=-1)

                correct_predictions += (predictions == labels).sum().item()
                total_predictions += labels.size(0)

                current_accuracy = correct_predictions / total_predictions
                progress_bar.set_postfix(accuracy=f"{current_accuracy:.2%}")

        final_accuracy = correct_predictions / total_predictions
        return final_accuracy


def main():
    print("Welcome to the Toxicity Classifier Evaluation Script!")
    print("This script evaluates the performance of a given toxic text classification model on the Jigsaw dataset.")
    print("Please ensure that 'test.csv' and 'test_labels.csv' are in the same folder as this script.\n")

    model_name = input("Enter the model name to test: ")

    classifier = ToxicityClassifier(
        test_file="test.csv",
        labels_file="test_labels.csv",
        model_name=model_name
    )

    print(f"\nUsing device: {classifier.device}")
    print(f"Model and tokenizer loaded: {model_name}")
    print("Dataset and DataLoader created")

    accuracy = classifier.evaluate_model()
    print(f"\nEvaluation complete. Accuracy: {accuracy:.2%}")


if __name__ == "__main__":
    main()