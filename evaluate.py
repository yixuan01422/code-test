import torch
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer, GPT2Config
from new_quantized_gpt2 import GPT2QuantModel
from load_squad import load_squad
from model_utils import load_model
from data_utils import collate_fn
import numpy as np
from tqdm import tqdm
import json
import os

def evaluate_model(model, tokenizer, eval_loader, device="cuda" if torch.cuda.is_available() else "cpu"):
    """Evaluate model performance on validation set"""
    model.eval()
    total_loss = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            # Forward pass
            hidden_states = model(input_ids=input_ids, attention_mask=attention_mask)
            
            # Compute logits
            logits = model.wte.weight @ hidden_states.transpose(-1, -2)
            logits = logits.transpose(-1, -2)
            
            # Shift logits and labels for causal LM loss
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()
            
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
            total_loss += loss.item() * input_ids.size(0)
            total_samples += input_ids.size(0)
    
    avg_loss = total_loss / total_samples
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    
    return {
        "loss": avg_loss,
        "perplexity": perplexity
    }

def test_quantization_configs(model_path=None):
    """Test different quantization configurations on SQuAD dataset"""
    print("Loading SQuAD dataset...")
    _, val_data = load_squad()
    
    # Initialize tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    # Prepare validation data loader
    val_loader = DataLoader(
        val_data.select(range(100)),  # Use a subset for quick testing
        batch_size=8,
        shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, tokenizer)
    )
    
    # Define different quantization configurations to test
    base_config = GPT2Config()
    num_layers = base_config.n_layer
    
    # Test configurations
    configs = [
        {
            "name": "8-bit uniform",
            "layer_bit_config": [[8, 8]] * num_layers
        },
        {
            "name": "4-bit uniform",
            "layer_bit_config": [[4, 4]] * num_layers
        },
        {
            "name": "Mixed precision (8-4-8)",
            "layer_bit_config": [[8, 8] if i % 3 != 1 else [4, 4] for i in range(num_layers)]
        },
        {
            "name": "Mixed precision (8-4-4)",
            "layer_bit_config": [[8, 8] if i == 0 else [4, 4] for i in range(num_layers)]
        }
    ]
    
    results = []
    
    for config in configs:
        print(f"\nTesting configuration: {config['name']}")
        
        # Initialize model with current configuration
        if model_path:
            # Load pretrained model if path is provided
            model, _, _, _ = load_model(model_path)
            # Update quantization configuration
            model.layer_bit_config = config["layer_bit_config"]
        else:
            # Create new model with current configuration
            model = GPT2QuantModel(
                base_config,
                config["layer_bit_config"],
                lora_configs=None  # No LoRA for evaluation
            )
        
        # Move model to device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        
        # Evaluate model
        metrics = evaluate_model(model, tokenizer, val_loader, device)
        
        # Calculate model size
        param_size = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
        total_size = (param_size + buffer_size) / 1024**2  # Convert to MB
        
        results.append({
            "config_name": config["name"],
            "layer_bit_config": config["layer_bit_config"],
            "metrics": metrics,
            "model_size_mb": total_size
        })
        
        print(f"Results for {config['name']}:")
        print(f"  Loss: {metrics['loss']:.4f}")
        print(f"  Perplexity: {metrics['perplexity']:.2f}")
        print(f"  Model size: {total_size:.2f} MB")
    
    # Save results
    os.makedirs("results", exist_ok=True)
    with open("results/quantization_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    return results

if __name__ == "__main__":
    # To evaluate a pretrained model, specify the path:
    # results = test_quantization_configs("checkpoints/checkpoint_epoch0_step0.pt")
    results = test_quantization_configs() 