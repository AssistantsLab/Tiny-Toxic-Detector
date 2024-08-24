#
# This info script measures VRAM and RAM usage. It is not intended to replicate the exact numbers from the paper,
# as these measurements are highly machine-specific. Any correlations drawn between models may be slightly skewed,
# as your machine may also be performing other tasks.
#
# No configuration necessary.
#
# Author: Michielo
#

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import psutil
import time

"""Get the current RAM usage of the process in MB."""
def get_ram_usage():
    process = psutil.Process()
    return process.memory_info().rss / (1024 * 1024)


"""Get the current VRAM usage of the GPU in MB."""
def get_vram_usage():
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024 * 1024)
    else:
        return 0

"""Load a transformer model and tokenizer."""
def load_model_and_tokenizer(model_name, tokenizer_name=None):
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name or model_name)
    return model, tokenizer

""""Calculate and print memory usage increase."""
def measure_memory_usage(initial_usage, final_usage, memory_type):
    increase = final_usage - initial_usage
    print(f"Initial {memory_type} usage: {initial_usage:.2f} MB")
    print(f"Final {memory_type} usage: {final_usage:.2f} MB")
    print(f"{memory_type} increase: {increase:.2f} MB")

"""Run inference with the specified number of tokens."""
def run_inference(model, tokenizer, num_tokens):
    input_text = " ".join(["token"] * num_tokens)
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=num_tokens)

    # Move inputs to the same device as the model
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Run inference
    with torch.no_grad():
        start_time = time.time()
        outputs = model(**inputs)
        end_time = time.time()

    return end_time - start_time

"""Measure the memory usage of loading and optionally running inference with a transformer model."""
def measure_model_memory(model_name, tokenizer_name=None, inference_tokens=None):
    print(f"Loading model: {model_name}")
    print(f"Tokenizer: {tokenizer_name or model_name}")

    initial_ram = get_ram_usage()
    initial_vram = get_vram_usage()

    start_time = time.time()
    model, tokenizer = load_model_and_tokenizer(model_name, tokenizer_name)
    end_time = time.time()
    print(f"Model and tokenizer loaded in {end_time - start_time:.2f} seconds")

    # Measure RAM usage after loading
    cpu_ram = get_ram_usage()
    print("\nMemory usage after loading (CPU):")
    measure_memory_usage(initial_ram, cpu_ram, "RAM")

    # Move model to GPU if available and measure VRAM usage
    if torch.cuda.is_available():
        model.to(torch.device("cuda:0"))
        print("\nModel moved to GPU")
        gpu_vram = get_vram_usage()
        print("Memory usage after loading (GPU):")
        measure_memory_usage(initial_vram, gpu_vram, "VRAM")
    else:
        print("\nNo GPU available")

    if inference_tokens:
        print(f"\nRunning inference with {inference_tokens} tokens")

        # CPU Inference
        print("\nCPU Inference:")
        model.to("cpu")
        inference_start_ram = get_ram_usage()
        inference_time = run_inference(model, tokenizer, inference_tokens)
        inference_end_ram = get_ram_usage()

        print(f"Inference time on CPU: {inference_time:.4f} seconds")
        measure_memory_usage(inference_start_ram, inference_end_ram, "RAM")

        # GPU Inference (if available)
        if torch.cuda.is_available():
            print("\nGPU Inference:")
            model.to("cuda")
            torch.cuda.empty_cache()  # Clear any leftover memory
            inference_start_vram = get_vram_usage()
            inference_time = run_inference(model, tokenizer, inference_tokens)
            inference_end_vram = get_vram_usage()

            print(f"Inference time on GPU: {inference_time:.4f} seconds")
            measure_memory_usage(inference_start_vram, inference_end_vram, "VRAM")
        else:
            print("\nNo GPU available for inference")

"""Main function to run the interactive program."""
def main():
    print("Welcome to the Transformer Model Memory Usage Measurement Tool")
    print("Enter 'q' to quit the program.")

    while True:
        model_name = input("\nEnter the name of the model to measure (or 'q' to quit): ")

        if model_name.lower() == 'q':
            print("Thank you for using the Memory Usage Measurement Tool. Goodbye!")
            break

        tokenizer_name = input("Enter the name of the tokenizer (press Enter to use the same as the model): ")
        tokenizer_name = tokenizer_name if tokenizer_name else None

        inference_option = input("Choose an option for inference (128 tokens, 4096 tokens, or press Enter to skip): ")
        inference_tokens = None
        if inference_option == "128":
            inference_tokens = 128
        elif inference_option == "4096":
            inference_tokens = 4096

        try:
            measure_model_memory(model_name, tokenizer_name, inference_tokens)
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            print("Please check the model and tokenizer names and try again.")

if __name__ == "__main__":
    main()