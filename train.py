import torch
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer, GPT2Config, AdamW
from new_quantized_gpt2 import GPT2QuantModel, set_active_lora
from load_squad import load_squad
from data_utils import collate_fn
from model_utils import save_model
import random

def train(total_steps=1000):
    print("Starting training setup...")
    print("========================")
    
    # 1. Load dataset and tokenizer
    print("\n1. Loading dataset and tokenizer...")
    train_data, _ = load_squad()
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    # 2. Prepare data loader
    print("\n2. Preparing data loader...")
    # train_data = train_data.select(range(1000))
    train_loader = DataLoader(
        train_data,
        batch_size=16,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, tokenizer)
    )
    
    # 3. Initialize model and configurations
    print("\n3. Initializing model...")
    base_config = GPT2Config()
    num_layers = base_config.n_layer
    
    # Configure quantization and LoRA
    layer_bit_config = [[8, 8]] * num_layers  # Use 8-bit for all layers
    
    # Define LoRA configurations
    lora_configs = [
        {"r": 4, "alpha": 16, "dropout": 0.1, "enable_lora": [True], "fan_in_fan_out": False},  # LoRA 1
        {"r": 8, "alpha": 32, "dropout": 0.1, "enable_lora": [True], "fan_in_fan_out": False},  # LoRA 2
    ]
    
    # Initialize model
    model = GPT2QuantModel(base_config, layer_bit_config, lora_configs=lora_configs)
    model.train()
    
    # 4. Create LM head with weight tying
    print("\n4. Setting up LM head...")
    lm_head = torch.nn.Linear(model.embed_dim, base_config.vocab_size, bias=False)
    lm_head.weight = model.wte.weight  # weight tying
    
    # 5. Initialize optimizer
    print("\n5. Setting up optimizer...")
    optimizer = AdamW(list(model.parameters()) + list(lm_head.parameters()), lr=5e-5)
    
    # 6. Training loop
    print("\n6. Starting training...")
    step = 0
    
    while step < total_steps:
        for batch in train_loader:
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            
            # Shift labels by one for causal language modeling
            labels = input_ids.clone()
            
            # Simulate switchable precision: choose an active LoRA adapter index for each layer
            active_config = [random.choice([0, 1]) for _ in range(num_layers)]
            set_active_lora(model, active_config)
            
            optimizer.zero_grad()
            
            # Forward pass through the quantized model
            hidden_states = model(input_ids=input_ids, attention_mask=attention_mask)
            
            # Compute logits
            logits = lm_head(hidden_states)
            
            # Shift logits and labels for causal LM loss
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
            loss.backward()
            optimizer.step()
            
            if step % 50 == 0:
                print(f"Step {step} / {total_steps} | Loss: {loss.item():.4f} | Active LoRA config: {active_config}")
            
            step += 1
            if step >= total_steps:
                break
    
    # Save the model after training
    print("\nSaving trained model...")
    save_model(model)
    
    print("\nTraining completed!")
    return model

if __name__ == "__main__":
    train(total_steps=1000) 