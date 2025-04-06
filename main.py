import os
from train import train
from evaluate import test_quantization_configs
from model_utils import load_model

def main():
    print("Starting the training and evaluation pipeline...")
    print("=============================================")
    
    # 1. Train the model for specified number of steps
    total_steps=1
    print(f"\n1. Training the model for {total_steps} steps...")
    train(total_steps=total_steps)
    
    # 2. Load the saved model
    print("\n2. Loading the saved model...")
    checkpoint_path = os.path.join("checkpoints", "model_checkpoint.pt")
    model = load_model(checkpoint_path)
    print("Model loaded successfully")
    # # 3. Evaluate the model with different quantization configurations
    # print("\n3. Evaluating the model with different quantization configurations...")
    # results = test_quantization_configs(checkpoint_path)
    
    # print("\nPipeline completed successfully!")
    # print("Results have been saved to 'results/quantization_results.json'")

if __name__ == "__main__":
    main() 