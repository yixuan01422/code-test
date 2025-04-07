import torch
import os
from new_quantized_gpt2 import GPT2QuantModel
from transformers import GPT2Config, AdamW

def save_model(model, save_dir="checkpoints"):
    """Save model state dictionary"""
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_path = os.path.join(save_dir, "model_checkpoint.pt")
    
    torch.save(model.state_dict(), checkpoint_path)
    print(f"\nSaved model state to {checkpoint_path}")

def load_model(model, checkpoint_path):
    """Load model state dictionary"""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint {checkpoint_path} not found")
    
    state_dict = torch.load(checkpoint_path)
    model.load_state_dict(state_dict)
    print(f"\nLoaded model state from {checkpoint_path}")
    
    return model 