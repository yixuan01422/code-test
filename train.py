import torch
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer, GPT2Config, AdamW
from new_quantized_gpt2 import GPT2QuantModel, set_active_lora, set_active_quantization
from load_squad import load_squad
from data_utils import collate_fn
from model_utils import save_model
import random
import os
from datetime import datetime

def log_message(message, log_file):
    """Print message to console and write to log file"""
    print(message)
    with open(log_file, 'a') as f:
        f.write(message + '\n')

def train(total_steps=1000):
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Create log file with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = f'logs/training_{timestamp}.txt'
    
    log_message("Starting training setup...", log_file)
    log_message("========================", log_file)
    
    # 1. Load dataset and tokenizer.
    log_message("\n1. Loading dataset and tokenizer...", log_file)
    train_data, _ = load_squad()
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    # 2. Prepare data loader.
    log_message("\n2. Preparing data loader...", log_file)
    train_loader = DataLoader(
        train_data,
        batch_size=16,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, tokenizer)
    )
    
    # 3. Initialize model and configurations.
    log_message("\n3. Initializing model...", log_file)
    base_config = GPT2Config()  # Using default configuration.
    num_layers = base_config.n_layer
    
    # layer_bit_config defines the default per-layer configuration (here [8, 8] for all layers),
    # but candidate lists allow switching to, e.g., 4-bit instead.
    layer_bit_config = [[8, 8]] * num_layers
    
    candidate_w_bits = [4, 8, 16]
    candidate_a_bits = [4, 8, 16]
    
    # Define two different LoRA configurations.
    lora_configs = [
        {"r": 4, "alpha": 16, "dropout": 0.1, "enable_lora": [True], "fan_in_fan_out": False},  # LoRA 1.
        {"r": 8, "alpha": 32, "dropout": 0.1, "enable_lora": [True], "fan_in_fan_out": False},  # LoRA 2.
    ]
    
    model = GPT2QuantModel(base_config, layer_bit_config, lora_configs=lora_configs,
                           candidate_w_bits=candidate_w_bits, candidate_a_bits=candidate_a_bits)
    model.train()
    
    # 4. Create LM head with weight tying.
    log_message("\n4. Setting up LM head...", log_file)
    lm_head = torch.nn.Linear(model.embed_dim, base_config.vocab_size, bias=False)
    lm_head.weight = model.wte.weight  # Tie LM head weights to token embeddings.
    
    # 5. Initialize optimizer.
    log_message("\n5. Setting up optimizer...", log_file)
    optimizer = AdamW(list(model.parameters()) + list(lm_head.parameters()), lr=5e-5)
    
    # 6. Training loop: Jointly train with all quantization candidates.
    log_message("\n6. Starting training...", log_file)
    step = 0
    
    while step < total_steps:
        for batch in train_loader:
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            labels = input_ids.clone()
            
            # For joint training, accumulate losses for each candidate quantization configuration.
            total_loss = 0.0
            
            # Randomly select quantization and LoRA configurations
            rand_w_bit_indices = [random.choice(range(len(candidate_w_bits))) for _ in range(num_layers)]
            rand_a_bit_indices = [random.choice(range(len(candidate_a_bits))) for _ in range(num_layers)]
            set_active_quantization(model, rand_w_bit_indices, rand_a_bit_indices)
            
            w_bits = [model.candidate_w_bits[idx] for idx in rand_w_bit_indices]
            a_bits = [model.candidate_a_bits[idx] for idx in rand_a_bit_indices]
            
            log_message(f"\nWeight bitwidth choices: {w_bits}", log_file)
            log_message(f"Activation bitwidth choices: {a_bits}", log_file)
            
            lora_choices = [random.choice([0, 1]) for _ in range(num_layers)]
            set_active_lora(model, lora_choices)
            log_message(f"LoRA choices: {lora_choices}", log_file)
            
            hidden_states = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = lm_head(hidden_states)
            
            # Shift logits and labels for causal language modeling.
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            total_loss += loss
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            if step % 1 == 0:
                log_message(f"Step {step} / {total_steps} | Loss: {total_loss.item():.4f}", log_file)
            step += 1
            if step >= total_steps:
                break
    
    # 7. Save the model after training.
    log_message("\nSaving trained model...", log_file)
    save_model(model)
    
    log_message("\nTraining completed!", log_file)
    return model

def train_fixed_bits(total_steps=1000, w_bits=8, a_bits=8):
    """Train with fixed bit configuration for all steps"""
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Create log file with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = f'logs/training_fixed_{w_bits}w_{a_bits}a_{timestamp}.txt'
    
    log_message(f"Starting training with fixed bits (w_bits={w_bits}, a_bits={a_bits})...", log_file)
    log_message("==========================================", log_file)
    
    # 1. Load dataset and tokenizer.
    log_message("\n1. Loading dataset and tokenizer...", log_file)
    train_data, _ = load_squad()
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    # 2. Prepare data loader.
    log_message("\n2. Preparing data loader...", log_file)
    train_loader = DataLoader(
        train_data,
        batch_size=16,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, tokenizer)
    )
    
    # 3. Initialize model and configurations.
    log_message("\n3. Initializing model...", log_file)
    base_config = GPT2Config()
    num_layers = base_config.n_layer
    
    layer_bit_config = [[w_bits, a_bits]] * num_layers
    
    candidate_w_bits = [4, 8, 16]
    candidate_a_bits = [4, 8, 16]
    
    # Define two different LoRA configurations.
    lora_configs = [
        {"r": 4, "alpha": 16, "dropout": 0.1, "enable_lora": [True], "fan_in_fan_out": False},  # LoRA 1.
        {"r": 8, "alpha": 32, "dropout": 0.1, "enable_lora": [True], "fan_in_fan_out": False},  # LoRA 2.
    ]
    model = GPT2QuantModel(base_config, layer_bit_config, lora_configs=lora_configs,
                           candidate_w_bits=candidate_w_bits, candidate_a_bits=candidate_a_bits)
    model.train()
    
    # Set the fixed bit configuration
    log_message(f"\nUsing fixed configuration: w_bits={w_bits}, a_bits={a_bits}", log_file)
    
    # 4. Create LM head with weight tying.
    log_message("\n4. Setting up LM head...", log_file)
    lm_head = torch.nn.Linear(model.embed_dim, base_config.vocab_size, bias=False)
    lm_head.weight = model.wte.weight
    
    # 5. Initialize optimizer.
    log_message("\n5. Setting up optimizer...", log_file)
    optimizer = AdamW(list(model.parameters()) + list(lm_head.parameters()), lr=5e-5)
    
    # 6. Training loop
    log_message("\n6. Starting training...", log_file)
    step = 0
    
    while step < total_steps:
        for batch in train_loader:
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            labels = input_ids.clone()
            
            hidden_states = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = lm_head(hidden_states)
            
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if step % 1 == 0:
                log_message(f"Step {step} / {total_steps} | Loss: {loss.item():.4f}", log_file)
            step += 1
            if step >= total_steps:
                break
    
    # 7. Save the model after training.
    log_message("\nSaving trained model...", log_file)
    save_model(model)
    
    log_message("\nTraining completed!", log_file)
    return model

def train_cyclic_bits(total_steps=1000, steps_per_config=100):
    """Train with cyclic bit configurations (4->8->16->4->8->16)"""
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Create log file with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = f'logs/training_cyclic_{timestamp}.txt'
    
    log_message("Starting training with cyclic bit configurations...", log_file)
    log_message("==========================================", log_file)
    
    # 1. Load dataset and tokenizer.
    log_message("\n1. Loading dataset and tokenizer...", log_file)
    train_data, _ = load_squad()
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    # 2. Prepare data loader.
    log_message("\n2. Preparing data loader...", log_file)
    train_loader = DataLoader(
        train_data,
        batch_size=16,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, tokenizer)
    )
    
    # 3. Initialize model and configurations.
    log_message("\n3. Initializing model...", log_file)
    base_config = GPT2Config()
    num_layers = base_config.n_layer
    layer_bit_config = [[8, 8]] * num_layers
    
    candidate_w_bits = [4, 8, 16]
    candidate_a_bits = [4, 8, 16]
    
    # Define two different LoRA configurations.
    lora_configs = [
        {"r": 4, "alpha": 16, "dropout": 0.1, "enable_lora": [True], "fan_in_fan_out": False},  # LoRA 1.
        {"r": 8, "alpha": 32, "dropout": 0.1, "enable_lora": [True], "fan_in_fan_out": False},  # LoRA 2.
    ]
    model = GPT2QuantModel(base_config, layer_bit_config, lora_configs=lora_configs,
                           candidate_w_bits=candidate_w_bits, candidate_a_bits=candidate_a_bits)
    model.train()
    
    # 4. Create LM head with weight tying.
    log_message("\n4. Setting up LM head...", log_file)
    lm_head = torch.nn.Linear(model.embed_dim, base_config.vocab_size, bias=False)
    lm_head.weight = model.wte.weight
    
    # 5. Initialize optimizer.
    log_message("\n5. Setting up optimizer...", log_file)
    optimizer = AdamW(list(model.parameters()) + list(lm_head.parameters()), lr=5e-5)
    
    # 6. Training loop with cyclic configurations
    log_message("\n6. Starting training...", log_file)
    step = 0
    
    # Define the cycle of bit configurations
    cycle = [(4, 4), (8, 8), (16, 16)]
    cycle_length = len(cycle)
    
    while step < total_steps:
        for batch in train_loader:
            # Get current configuration from cycle
            cycle_idx = (step // steps_per_config) % cycle_length
            w_bits, a_bits = cycle[cycle_idx]
            
            # Find indices for current bitwidths
            w_idx = candidate_w_bits.index(w_bits)
            a_idx = candidate_a_bits.index(a_bits)
            
            # Use the same bit configuration for all layers
            w_indices = [w_idx] * num_layers
            a_indices = [a_idx] * num_layers
            
            # Set the current bit configuration
            set_active_quantization(model, w_indices, a_indices)
            log_message(f"\nStep {step}: Using configuration w_bits={w_bits}, a_bits={a_bits}", log_file)
            
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            labels = input_ids.clone()
            
            hidden_states = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = lm_head(hidden_states)
            
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if step % 1 == 0:
                log_message(f"Step {step} / {total_steps} | Loss: {loss.item():.4f}", log_file)
            step += 1
            if step >= total_steps:
                break
    
    # 7. Save the model after training.
    log_message("\nSaving trained model...", log_file)
    save_model(model)
    
    log_message("\nTraining completed!", log_file)
    return model

if __name__ == "__main__":
    # Example usage:
    # train_fixed_bits(total_steps=1000, w_bits=8, a_bits=8)  # Train with fixed 8-bit configuration
    # train_cyclic_bits(total_steps=1000)  # Train with cyclic configurations
    train(total_steps=5)  # Original random configuration training
