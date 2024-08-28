#
# This file contains a ToxicityClassifier class that can be used to evaluate the performance of the Tiny-Toxic-Detector on the Jigsaw dataset.
# It loads the pre-trained model and evaluates it on the Jigsaw test set, providing the accuracy percentage.
#
# Note: This script requires 'test.csv' and 'test_labels.csv' from the Jigsaw challenge to be in the same folder.
# These files will be included in the GitHub repository.
#
# Author: Michielo
#

import torch
import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig, AutoTokenizer
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import pandas as pd

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

# Configuration Class
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

# Sequence Classification Model
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

# Dataset Class
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
            'labels': torch.tensor(label, dtype=torch.float)
        }

# Function to Load Data
def load_data(test_file, labels_file):
    test_df = pd.read_csv(test_file)
    labels_df = pd.read_csv(labels_file)

    merged_df = pd.merge(test_df, labels_df, on='id')
    filtered_df = merged_df[merged_df['toxic'] != -1]

    return filtered_df['comment_text'].tolist(), filtered_df['toxic'].tolist()

# Function to Load Model and Tokenizer
def load_model_and_tokenizer(model_path, tokenizer_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = TinyTransformerConfig.from_pretrained(model_path)
    model = TinyTransformerForSequenceClassification.from_pretrained(model_path, config=config).to(device)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    return model, tokenizer, device

# Evaluation Function
def evaluate_model(model, dataloader, device):
    model.eval()
    correct_predictions = 0
    total_predictions = 0

    progress_bar = tqdm(dataloader, desc="Evaluating")

    with torch.no_grad():
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs['logits'].squeeze()

            predictions = (logits > 0.5).float()
            correct_predictions += (predictions == labels).sum().item()
            total_predictions += labels.size(0)

            current_accuracy = correct_predictions / total_predictions
            progress_bar.set_postfix(accuracy=f"{current_accuracy:.2%}")

    final_accuracy = correct_predictions / total_predictions
    return final_accuracy

def main():
    # Load dataset
    test_file = "test.csv"
    labels_file = "test_labels.csv"
    comments, labels = load_data(test_file, labels_file)
    print("Dataset loaded successfully")

    # Load the model and tokenizer
    model_path = "AssistantsLab/Tiny-Toxic-Detector"
    tokenizer_name = "AssistantsLab/Tiny-Toxic-Detector"
    model, tokenizer, device = load_model_and_tokenizer(model_path, tokenizer_name)
    print(f"Model and tokenizer loaded: {model_path}")

    # Print model details
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")

    # Create dataset and dataloader
    dataset = JigsawDataset(comments, labels, tokenizer)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    print("Dataset and DataLoader created")

    # Evaluate the model
    accuracy = evaluate_model(model, dataloader, device)

    # Print results
    print(f"\nEvaluation complete. Accuracy: {accuracy:.2%}")

if __name__ == "__main__":
    main()