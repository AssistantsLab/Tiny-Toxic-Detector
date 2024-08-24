#
# This is an example for how to run inference on the Tiny-Toxic-Detector
# As it uses a custom architecture this needs to be defined as well.
#
# No configuration necessary unless you want to incorporate this in a separate system.
#
# Author: Michielo
#

import torch
import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig, AutoTokenizer
from huggingface_hub import login
import os

# Define TinyTransformer model
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

    def __init__(self, vocab_size=30522, embed_dim=64, num_heads=2, ff_dim=128, num_layers=4, max_position_embeddings=512, **kwargs):
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

# Load the Tiny-Toxic-Detector model and tokenizer
def load_model_and_tokenizer():
    device = torch.device("cpu")

    # Load Tiny-toxic-detector
    config = TinyTransformerConfig.from_pretrained("AssistantsLab/Tiny-Toxic-Detector")
    model = TinyTransformerForSequenceClassification.from_pretrained("AssistantsLab/Tiny-Toxic-Detector", config=config).to(device)
    tokenizer = AutoTokenizer.from_pretrained("AssistantsLab/Tiny-Toxic-Detector")

    return model, tokenizer, device

# Prediction function
def predict_toxicity(text, model, tokenizer, device):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128, padding="max_length").to(device)
    if "token_type_ids" in inputs:
        del inputs["token_type_ids"]

    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs["logits"].squeeze()
    prediction = "Toxic" if logits > 0.5 else "Not Toxic"
    return prediction

def main():
    model, tokenizer, device = load_model_and_tokenizer()

    while True:
        print("Enter text to classify (or type 'exit' to quit):")
        text = input()

        if text.lower() == 'exit':
            print("Exiting...")
            break

        if text:
            prediction = predict_toxicity(text, model, tokenizer, device)
            print(f"Prediction: {prediction}")
        else:
            print("No text provided. Please enter some text.")

if __name__ == "__main__":
    main()
