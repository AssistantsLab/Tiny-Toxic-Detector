#
# This info script measures inference speed. It is not intended to replicate the exact numbers from the paper,
# as these measurements are highly machine-specific. Any correlations drawn between models may be slightly skewed,
# as your machine may also be performing other tasks.
#
# No configuration necessary.
#
# Author: Michielo
#

import torch
import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig, AutoTokenizer
import time

# Define the TinyTransformer architecture
class TinyTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, ff_dim, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoding = nn.Parameter(torch.zeros(1, 512, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=ff_dim, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embed_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.embedding(x) + self.pos_encoding[:, :x.size(1), :]
        x = self.transformer(x)
        x = x.mean(dim=1)  # Global average pooling
        x = self.fc(x)
        return self.sigmoid(x)

# Define the configuration class
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

# Sequence classification model using TinyTransformer
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

"""Function to load the TinyTransformer model and tokenizer"""
def load_model_and_tokenizer():
    config = TinyTransformerConfig.from_pretrained("AssistantsLab/Tiny-Toxic-Detector")
    model = TinyTransformerForSequenceClassification.from_pretrained("AssistantsLab/Tiny-Toxic-Detector", config=config)
    tokenizer = AutoTokenizer.from_pretrained("AssistantsLab/Tiny-Toxic-Detector")
    return model, tokenizer


"""Run inference with the specified number of tokens and return the time taken."""
def run_inference(model, tokenizer, num_tokens, device):
    input_text = " ".join(["token"] * num_tokens)
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=num_tokens)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs.get("attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    # Run inference
    model.to(device)
    with torch.no_grad():
        start_time = time.time()
        outputs = model(input_ids, attention_mask=attention_mask)
        if device.type == 'cuda':
            torch.cuda.synchronize()  # Ensure GPU operations are complete
        end_time = time.time()

    return end_time - start_time, outputs["logits"]


"""Measure the inference time of the TinyTransformer model on both CPU and GPU."""
def measure_inference_time_tiny_transformer(inference_tokens=128):
    print(f"Loading TinyTransformer model: tiny_toxic_detector")

    model, tokenizer = load_model_and_tokenizer()
    print("Model and tokenizer loaded.")

    # Measure inference time on CPU
    device = "cpu"
    print(f"\nRunning inference with {inference_tokens} tokens on {device.upper()}")
    inference_time, logits = run_inference(model, tokenizer, inference_tokens, torch.device(device))
    print(f"Inference time on CPU: {inference_time:.4f} seconds")

    # Measure inference time on GPU (if available)
    if torch.cuda.is_available():
        device = "cuda"
        print(f"\nRunning inference with {inference_tokens} tokens on {device.upper()}")
        inference_time, logits = run_inference(model, tokenizer, inference_tokens, torch.device(device))
        print(f"Inference time on GPU: {inference_time:.4f} seconds")
    else:
        print("\nNo GPU available for inference")


"""Main function to run the TinyTransformer inference measurement."""
def main():
    print("Welcome to the TinyTransformer Inference Time Measurement Tool")

    inference_option = input("Choose an option for inference (128 tokens, 512 tokens, or press Enter for 128 tokens): ")
    inference_tokens = 128
    if inference_option == "512":
        inference_tokens = 512

    try:
        measure_inference_time_tiny_transformer(inference_tokens)
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()