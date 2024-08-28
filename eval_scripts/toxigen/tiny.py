#
# This file contains a ToxicityClassifier class that can be used to evaluate the performance of the tiny-toxic-detector on the Toxigen dataset.
# For other models, please see the other files for evaluating on the toxigen dataset.
#
# Why the train section?!?!
# - Currently none of the models are known to have data contamination. The test-section of the original HF dataset is too small to give an accurate view.
#   This *does* mean this test becomes invalidated for any models that trained on the toxigen dataset, but for the paper in question this is irrelevant.
#
# Author: Michielo
#

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import PreTrainedModel, PretrainedConfig, AutoTokenizer
from tqdm import tqdm

# Model Architecture
class TinyTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, ff_dim, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoding = nn.Parameter(torch.zeros(1, 512, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=ff_dim, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embed_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.embedding(x) + self.pos_encoding[:, :x.size(1), :]
        x = self.transformer(x)
        x = x.mean(dim=1)  # Global average pooling
        x = self.fc(x)
        return self.sigmoid(x)

class TinyTransformerConfig(PretrainedConfig):
    model_type = "tiny_transformer"
    def __init__(
        self,
        vocab_size=30522,
        embed_dim=64,
        num_heads=2,
        ff_dim=128,
        num_layers=4,
        max_position_embeddings=512,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.num_layers = num_layers
        self.max_position_embeddings = max_position_embeddings

class TinyTransformerForSequenceClassification(PreTrainedModel):
    config_class = TinyTransformerConfig

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = 1
        self.transformer = TinyTransformer(
            config.vocab_size,
            config.embed_dim,
            config.num_heads,
            config.ff_dim,
            config.num_layers
        )

    def forward(self, input_ids, attention_mask=None):
        outputs = self.transformer(input_ids)
        return {"logits": outputs}

# Evaluation Functions
def load_custom_dataset(dataset_name, split="train"):
    dataset = load_dataset(dataset_name, split=split)
    return dataset

def load_model_and_tokenizer(model_path, tokenizer_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = TinyTransformerConfig.from_pretrained(model_path)
    model = TinyTransformerForSequenceClassification.from_pretrained(model_path, config=config).to(device)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    return model, tokenizer, device

def preprocess_data(dataset, tokenizer, batch_size=32):
    def preprocess_function(examples):
        tokenized = tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

        # Filter out samples with more than 128 tokens
        input_ids = tokenized["input_ids"]
        filtered_input_ids = [ids for ids in input_ids if len(ids) <= 128]

        labels = [1 if label == ["toxic"] else 0 for label in examples["labels"]]

        # Only keep the filtered samples
        tokenized["input_ids"] = filtered_input_ids
        tokenized["labels"] = [labels[i] for i in range(len(labels)) if len(input_ids[i]) <= 128]

        return tokenized

    tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=dataset.column_names)
    tokenized_dataset.set_format("torch")
    dataloader = DataLoader(tokenized_dataset, batch_size=batch_size, shuffle=False)
    return dataloader

def evaluate_model(model, dataloader, device):
    model.eval()
    correct_predictions = 0
    total_predictions = 0
    progress_bar = tqdm(dataloader, desc="Evaluating")

    with torch.no_grad():
        for batch in progress_bar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs["logits"].squeeze()
            predictions = (logits > 0.5).float()

            correct_predictions += (predictions == labels).sum().item()
            total_predictions += labels.size(0)

            current_accuracy = correct_predictions / total_predictions
            progress_bar.set_postfix(accuracy=f"{current_accuracy:.2%}")

    final_accuracy = correct_predictions / total_predictions
    return final_accuracy

def main():
    # Load dataset
    dataset_name = "Intuit-GenSRF/toxigen-train"
    dataset = load_custom_dataset(dataset_name)
    print("Dataset loaded successfully")

    # Load the model and tokenizer
    model_path = "AssistantsLab/Tiny-Toxic-Detector"
    tokenizer_name = "AssistantsLab/Tiny-Toxic-Detector"
    model, tokenizer, device = load_model_and_tokenizer(model_path, tokenizer_name)
    print(f"Model and tokenizer loaded: {model_path}")

    # Preprocess the data and create a DataLoader
    dataloader = preprocess_data(dataset, tokenizer)
    print("Data preprocessed and DataLoader created")

    # Evaluate the model
    accuracy = evaluate_model(model, dataloader, device)
    print(f"Evaluation complete. Accuracy: {accuracy:.2%}")

if __name__ == "__main__":
    main()