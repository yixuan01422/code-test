import os
import json
import torch
import re
import string
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer, GPT2Config
from new_quantized_gpt2 import GPT2QuantModel, set_active_lora, set_active_quantization
from load_squad import load_squad
from data_utils import collate_fn
import random
from collections import Counter

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def f1_score(prediction, ground_truth):
    """Calculate token-level F1 score between prediction and ground truth."""
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def evaluate_squad(model, dataloader, tokenizer, device, num_samples=10):
    """Evaluate model on SQuAD validation set using F1 score and loss"""
    model.eval()
    total_f1 = 0.0
    total_loss = 0.0
    total = 0
    
    # Take a random subset of samples
    all_samples = list(dataloader)
    if len(all_samples) > num_samples:
        samples = random.sample(all_samples, num_samples)
    else:
        samples = all_samples
    
    print(f"\nEvaluating on {len(samples)} samples...")
    
    # Create LM head with weight tying (same as in training)
    lm_head = torch.nn.Linear(model.embed_dim, model.config.vocab_size, bias=False)
    lm_head.weight = model.wte.weight
    lm_head = lm_head.to(device)
    
    with torch.no_grad():
        for batch in samples:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = input_ids.clone()
            
            # Get model's prediction and apply LM head (same as training)
            hidden_states = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = lm_head(hidden_states)
            
            # Calculate loss (same as training)
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            total_loss += loss.item()
            
            # Get the predicted answer (greedy decoding)
            predicted_ids = torch.argmax(logits, dim=-1)
            predicted_text = tokenizer.decode(predicted_ids[0], skip_special_tokens=True)
            
            # Get the actual answer from the batch
            actual_text = batch['answer_text'][0]
            
            # Calculate F1 score
            f1 = f1_score(predicted_text, actual_text)
            total_f1 += f1
            total += 1
            
            # Print some examples
            if total <= 3:  # Print first 3 examples
                print(f"\nExample {total}:")
                print(f"Actual Answer: {actual_text}")
                print(f"Predicted Answer: {predicted_text[:100]}...")
                print(f"F1 Score: {f1:.4f}")
                print(f"Loss: {loss.item():.4f}")
    
    if total == 0:
        print("No samples were successfully evaluated")
        return 0.0, float('inf')
    
    avg_f1 = total_f1 / total
    avg_loss = total_loss / total
    print(f"\nEvaluation Results:")
    print(f"Average F1 Score: {avg_f1:.4f}")
    print(f"Average Loss: {avg_loss:.4f}")
    
    return avg_f1, avg_loss

def test_quantization_configs(model):
    print("Testing different quantization configurations...")
    print("=============================================")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # 1. Load validation data
    print("\n1. Loading validation data...")
    _, val_data = load_squad()
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    val_loader = DataLoader(
        val_data,
        batch_size=1,  # Process one example at a time
        shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, tokenizer)
    )
    
    # 2. Test different quantization configurations
    print("\n2. Testing different configurations...")
    results = []
    
    # Test all combinations of weight and activation bitwidths
    for w_idx in range(len(model.candidate_w_bits)):
        for a_idx in range(len(model.candidate_a_bits)):
            w_bits = model.candidate_w_bits[w_idx]
            a_bits = model.candidate_a_bits[a_idx]
            print(f"\nTesting w_bits={w_bits}, a_bits={a_bits}")
            
            # Set all layers to use the same bitwidth configuration
            num_layers = model.config.n_layer
            set_active_quantization(model, [w_idx] * num_layers, [a_idx] * num_layers)
            
            # Evaluate on SQuAD task
            f1_score, loss = evaluate_squad(model, val_loader, tokenizer, device)
            
            results.append({
                'w_bits': w_bits,
                'a_bits': a_bits,
                'f1_score': f1_score,
                'loss': loss
            })
    
    # 3. Save results
    print("\n3. Saving results...")
    os.makedirs("results", exist_ok=True)
    with open("results/quantization_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    return results