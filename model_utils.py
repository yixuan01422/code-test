import torch
import os
from new_quantized_gpt2 import GPT2QuantModel
from transformers import GPT2Config, AdamW

def save_model(model, save_dir="checkpoints"):
    """Save model checkpoint"""
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_path = os.path.join(save_dir, "model_checkpoint.pt")
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': model.config,
        'layer_bit_config': model.layer_bit_config,
        'lora_configs': model.lora_configs
    }, checkpoint_path)
    print(f"\nSaved model to {checkpoint_path}")

def load_model(checkpoint_path):
    """Load model checkpoint"""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint {checkpoint_path} not found")
    
    checkpoint = torch.load(checkpoint_path)
    
    # Recreate model with saved configuration
    model = GPT2QuantModel(
        config=checkpoint['config'],
        layer_bit_config=checkpoint['layer_bit_config'],
        lora_configs=checkpoint['lora_configs']
    )
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"\nLoaded model from {checkpoint_path}")
    
    return model 