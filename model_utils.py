import torch
import os
from new_quantized_gpt2 import GPT2QuantModel
from transformers import GPT2Config, AdamW

def save_model(model, optimizer, epoch, step, loss, save_dir="checkpoints"):
    """Save model checkpoint"""
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_path = os.path.join(save_dir, f"checkpoint_epoch{epoch}_step{step}.pt")
    
    torch.save({
        'epoch': epoch,
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'config': model.config,
        'layer_bit_config': model.layer_bit_config,
        'lora_configs': model.lora_configs
    }, checkpoint_path)
    print(f"\nSaved checkpoint to {checkpoint_path}")

def load_model(checkpoint_path):
    """Load model checkpoint"""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint {checkpoint_path} not found")
    
    checkpoint = torch.load(checkpoint_path)
    
    # Recreate model with saved configuration
    model = GPT2QuantModel(
        checkpoint['config'],
        checkpoint['layer_bit_config'],
        lora_configs=checkpoint['lora_configs']
    )
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Create and load optimizer state
    optimizer = AdamW(model.parameters(), lr=5e-5)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f"\nLoaded checkpoint from {checkpoint_path}")
    print(f"Resuming from epoch {checkpoint['epoch']}, step {checkpoint['step']}")
    
    return model, optimizer, checkpoint['epoch'], checkpoint['step'] 