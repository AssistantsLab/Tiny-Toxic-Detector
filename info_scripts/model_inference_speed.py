# This info script measures inference speed. It is not intended to replicate the exact numbers from the paper,
# as these measurements are highly machine-specific. Any correlations drawn between models may be slightly skewed,
# as your machine may also be performing other tasks.
#
# No configuration necessary.
#
# Author: Michielo
#

import torch
import time
from transformers import AutoModelForSequenceClassification, AutoTokenizer

"""Function to load a model and tokenizer"""
def load_model_and_tokenizer(model_name):
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

"""Run inference with the specified number of tokens and return the time taken."""
def run_inference(model, tokenizer, num_tokens, device):
    input_text = " ".join(["token"] * num_tokens)
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=num_tokens)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs.get("attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    model.to(device)
    with torch.no_grad():
        start_time = time.time()
        outputs = model(input_ids, attention_mask=attention_mask)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        end_time = time.time()

    return end_time - start_time, outputs.logits

"""Measure the inference time of a model on both CPU and GPU."""
def measure_inference_time(model_name, inference_tokens=128):
    print(f"\nLoading model: {model_name}")

    try:
        model, tokenizer = load_model_and_tokenizer(model_name)
        print("Model and tokenizer loaded successfully.")
    except Exception as e:
        print(f"Error loading model {model_name}: {str(e)}")
        return

    for device in ["cpu", "cuda"]:
        if device == "cuda" and not torch.cuda.is_available():
            print("No GPU available for inference")
            continue

        print(f"\nRunning inference with {inference_tokens} tokens on {device.upper()}")
        try:
            inference_time, _ = run_inference(model, tokenizer, inference_tokens, torch.device(device))
            print(f"Inference time on {device.upper()}: {inference_time:.4f} seconds")
        except Exception as e:
            print(f"Error during inference on {device.upper()}: {str(e)}")

"""Main function to run the inference speed measurement(s)."""
def main():
    print("Welcome to the Flexible Model Inference Time Measurement Tool")

    models_to_benchmark = []
    while True:
        model_name = input("Enter a model name to benchmark (or press Enter to finish): ")
        if not model_name:
            break
        models_to_benchmark.append(model_name)

    if not models_to_benchmark:
        print("No models specified. Using default models.")
        models_to_benchmark = ["distilbert-base-uncased", "bert-base-uncased", "roberta-base"]

    inference_option = input("Choose an option for inference (128 tokens, 512 tokens, or press Enter for 128 tokens): ")
    inference_tokens = 128 if inference_option != "512" else 512

    for model_name in models_to_benchmark:
        measure_inference_time(model_name, inference_tokens)

    print("\nBenchmarking completed.")

if __name__ == "__main__":
    main()