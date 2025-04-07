import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from load_squad import load_squad
from data_utils import collate_fn
from evaluate import evaluate_squad, f1_score
import random

def test_original_gpt2(num_samples=10):
    """Test the original GPT-2 model on SQuAD dataset"""
    print("Testing original GPT-2 model...")
    print("===============================")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model and tokenizer
    print("\n1. Loading model and tokenizer...")
    model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load validation data
    print("\n2. Loading validation data...")
    _, val_data = load_squad()
    
    val_loader = torch.utils.data.DataLoader(
        val_data,
        batch_size=1,
        shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, tokenizer)
    )
    
    # Evaluate model
    print("\n3. Evaluating model...")
    model.eval()
    with torch.no_grad():
        # Create a wrapper function to handle the original model's output format
        def evaluate_wrapper(model, dataloader, tokenizer, device, num_samples):
            model.eval()
            total_f1 = 0.0
            total = 0
            
            # Take a random subset of samples
            all_samples = list(dataloader)
            if len(all_samples) > num_samples:
                samples = random.sample(all_samples, num_samples)
            else:
                samples = all_samples
            
            print(f"\nEvaluating on {len(samples)} samples...")
            
            with torch.no_grad():
                for batch in samples:
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    
                    # Get model's prediction
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    # Extract logits from the output dictionary
                    logits = outputs.logits
                    
                    # Get the predicted answer (greedy decoding)
                    predicted_ids = torch.argmax(logits, dim=-1)
                    predicted_text = tokenizer.decode(predicted_ids[0], skip_special_tokens=True)
                    
                    # Get the actual answer from the batch
                    actual_text = batch['answer_text'][0]
                    
                    # Calculate F1 score
                    score = f1_score(predicted_text, actual_text)
                    total_f1 += score
                    total += 1
                    
                    # Print some examples
                    if total <= 3:  # Print first 3 examples
                        print(f"\nExample {total}:")
                        print(f"Actual Answer: {actual_text}")
                        print(f"Predicted Answer: {predicted_text[:100]}...")
                        print(f"F1 Score: {score:.4f}")
            
            avg_f1 = total_f1 / total
            print(f"\nEvaluation Results:")
            print(f"Average F1 Score: {avg_f1:.4f}")
            
            return avg_f1
        
        avg_f1 = evaluate_wrapper(model, val_loader, tokenizer, device, num_samples)
    
    print(f"\nOriginal GPT-2 Model F1 Score: {avg_f1:.4f}")
    return avg_f1

if __name__ == "__main__":
    test_original_gpt2() 