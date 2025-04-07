import torch
import os
from new_quantized_gpt2 import GPT2QuantModel
from transformers import GPT2Config, AdamW
from datetime import datetime

def save_model(model, training_method="default", save_dir="checkpoints"):
    """Save model state dictionary with training method and timestamp in filename"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Create filename with training method and timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    checkpoint_path = os.path.join(save_dir, f"model_{training_method}_{timestamp}.pt")
    
    torch.save(model.state_dict(), checkpoint_path)
    print(f"\nSaved model state to {checkpoint_path}")
    
    return checkpoint_path

def load_model(model, checkpoint_path):
    """Load model state dictionary"""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint {checkpoint_path} not found")
    
    state_dict = torch.load(checkpoint_path)
    model.load_state_dict(state_dict)
    print(f"\nLoaded model state from {checkpoint_path}")
    
    return model 