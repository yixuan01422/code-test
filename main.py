import os
from train import train, train_fixed_bits, train_cyclic_bits
from evaluate import test_quantization_configs
from model_utils import load_model
from new_quantized_gpt2 import GPT2QuantModel
from transformers import GPT2Config

def main():
    # 1. Train the model
    print("Starting training...")
    # model = train(total_steps=100)
    # model = train_fixed_bits(total_steps=100, w_bits=8, a_bits=8)
    model = train_cyclic_bits(total_steps=100, steps_per_config=10)
    # 2. Load the saved model
    print("\nLoading saved model...")
    checkpoint_path = "checkpoints/model_checkpoint.pt"
    model = load_model(model, checkpoint_path)
    
    # 3. Evaluate with different quantization configurations
    print("\nTesting different quantization configurations...")
    results = test_quantization_configs(model)
    
    print("\nPipeline completed successfully!")
    print(f"Results saved to: results/quantization_results.json")

if __name__ == "__main__":
    main() 