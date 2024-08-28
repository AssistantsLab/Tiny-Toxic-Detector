#
# This file contains a ToxicityClassifier class that can be used to evaluate the performance of a given toxic text classification model on the Toxigen dataset.
# The user can specify the model name to be tested. This script is specifically designed for the ToxicChat model as it requires custom prompting.
#
# Why the train section?!?!
# - Currently none of the models are known to have data contamination. The test-section of the original HF dataset is too small to give an accurate view.
#   This *does* mean this test becomes invalidated for any models that trained on the toxigen dataset, but for the paper in question this is irrelevant.
#
# Author: Michielo
#

import torch
from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from torch.utils.data import DataLoader
from tqdm import tqdm

class ToxicityClassifier:
    def __init__(self, dataset_name, model_name, tokenizer_name="t5-large", batch_size=32, device=None):
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.tokenizer_name = tokenizer_name
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
        self.model, self.tokenizer = self.load_model_and_tokenizer()
        self.dataloader = self.preprocess_data()

    def load_custom_dataset(self, split="train"):
        # Loads a custom dataset from the specified name.
        dataset = load_dataset(self.dataset_name, split=split)
        return dataset

    def load_model_and_tokenizer(self):
        # Loads the specified model and tokenizer.
        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        return model, tokenizer

    def preprocess_data(self):
        # Preprocesses the dataset, preparing it for evaluation.
        dataset = self.load_custom_dataset()

        def preprocess_function(examples):
            # Convert labels from ["toxic"] to 1 and [] to 0
            labels = [1 if label == ["toxic"] else 0 for label in examples["labels"]]
            return {"text": examples["text"], "labels": labels}

        processed_dataset = dataset.map(preprocess_function, batched=True, remove_columns=dataset.column_names)
        processed_dataset.set_format(type="torch", columns=["text", "labels"])
        dataloader = DataLoader(processed_dataset, batch_size=self.batch_size, shuffle=False)
        return dataloader

    def evaluate_model(self):
        # Evaluates the model's performance on the test dataset and returns the accuracy.
        self.model.eval()
        self.model.to(self.device)
        correct_predictions = 0
        total_predictions = 0
        progress_bar = tqdm(self.dataloader, desc="Evaluating")

        with torch.no_grad():
            for batch in progress_bar:
                texts = batch["text"]
                labels = batch["labels"].to(self.device)

                batch_predictions = []
                for text in texts:
                    prefix = "ToxicChat: " # custom prefix as outlined by the huggingface repository for this model
                    inputs = self.tokenizer.encode(prefix + text, return_tensors="pt").to(self.device)
                    outputs = self.model.generate(inputs, max_new_tokens=20)
                    prediction = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                    batch_predictions.append(1 if prediction.lower() == 'positive' else 0)

                predictions = torch.tensor(batch_predictions, device=self.device)

                correct_predictions += (predictions == labels).sum().item()
                total_predictions += labels.size(0)
                current_accuracy = correct_predictions / total_predictions
                progress_bar.set_postfix(accuracy=f"{current_accuracy:.2%}")

        final_accuracy = correct_predictions / total_predictions
        return final_accuracy

def main():
    # The main function loads the specified model and dataset, and evaluates the model's performance.
    classifier = ToxicityClassifier(
        dataset_name="Intuit-GenSRF/toxigen-train",
        model_name="lmsys/toxicchat-t5-large-v1.0",
        tokenizer_name="t5-large"
    )
    accuracy = classifier.evaluate_model()
    print(f"Evaluation complete. Accuracy: {accuracy:.2%}")

if __name__ == "__main__":
    main()