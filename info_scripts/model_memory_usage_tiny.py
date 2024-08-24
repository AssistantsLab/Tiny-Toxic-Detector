#
# This info script measures VRAM and RAM usage. It is not intended to replicate the exact numbers from the paper,
# as these measurements are highly machine-specific. Any correlations drawn between models may be slightly skewed,
# as your machine may also be performing other tasks. If you are looking to benchmark models besides the
# Tiny-Toxic-Detector please check out model_memory_usage.py instead.
#
# No configuration necessary.
#
# Author: Michielo
#

import torch
import torch.nn as nn
import psutil
import time
from transformers import PreTrainedModel, PretrainedConfig, AutoTokenizer

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

"""
Get the current RAM usage of the process in MB.
"""
def get_ram_usage():
    process = psutil.Process()
    return process.memory_info().rss / (1024 * 1024)

def get_vram_usage():
    """
    Get the current VRAM usage of the GPU in MB.
    """
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024 * 1024)
    else:
        return 0

"""
Load a TinyTransformer model using the provided model class.
"""
def load_model(model_name):
    config = TinyTransformerConfig.from_pretrained(model_name)
    model = TinyTransformerForSequenceClassification.from_pretrained(model_name, config=config)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

"""
Measure the memory usage of loading a TinyTransformer model.
"""
def main():
    model_name = "AssistantsLab/Tiny-Toxic-Detector"
    print(f"Loading TinyTransformer model: {model_name}")

    # Get initial RAM usage
    initial_ram = get_ram_usage()
    print(f"Initial RAM usage: {initial_ram:.2f} MB")

    # Load the model
    start_time = time.time()
    model, tokenizer = load_model(model_name)
    end_time = time.time()
    print(f"Model loaded in {end_time - start_time:.2f} seconds")

    # Get RAM usage after loading the model
    final_ram = get_ram_usage()
    print(f"Final RAM usage: {final_ram:.2f} MB")

    # Calculate the increase in RAM usage
    ram_increase = final_ram - initial_ram
    print(f"RAM increase: {ram_increase:.2f} MB")

    # Get initial VRAM usage
    initial_vram = get_vram_usage()
    print(f"Initial VRAM usage: {initial_vram:.2f} MB")

    # Move the model to the GPU
    if torch.cuda.is_available():
        model.to(torch.device("cuda:0"))
        print("Model moved to GPU")

        # Get VRAM usage after moving the model to the GPU
        final_vram = get_vram_usage()
        print(f"Final VRAM usage: {final_vram:.2f} MB")

        # Calculate the increase in VRAM usage
        vram_increase = final_vram - initial_vram
        print(f"VRAM increase: {vram_increase:.2f} MB")
    else:
        print("No GPU available")

if __name__ == "__main__":
    main()