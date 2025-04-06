import torch
from transformers import GPT2Tokenizer
from new_quantized_gpt2 import GPT2QuantModel, GPT2Config, set_active_lora
from load_squad import load_squad

def test_quantized_model():
    print("Testing Quantized GPT-2 with SQuAD sample")
    print("=========================================")
    
    # 1. Load SQuAD dataset
    print("\n1. Loading SQuAD dataset...")
    train_data, _ = load_squad()
    
    # 2. Initialize tokenizer
    print("\n2. Initializing tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    # 3. Prepare a sample from SQuAD
    print("\n3. Preparing sample data...")
    sample = train_data[0]
    text = f"Context: {sample['context']} Question: {sample['question']} Answer: {sample['answers']['text'][0]}"
    
    # Tokenize the sample
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True, padding='max_length')
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    
    print(f"Sample text length: {len(text)} characters")
    print(f"Tokenized length: {input_ids.shape[1]} tokens")
    
    # 4. Initialize quantized model
    print("\n4. Initializing quantized model...")
    config = GPT2Config()
    num_layers = config.n_layer
    
    # Configure quantization and LoRA
    layer_bit_config = [[8, 8]] * num_layers  # Use 8-bit for all layers
    
    # Define LoRA configurations
    lora_configs = [
        {"r": 4, "alpha": 16, "dropout": 0.1, "enable_lora": [True], "fan_in_fan_out": False},  # LoRA 1
        {"r": 8, "alpha": 32, "dropout": 0.1, "enable_lora": [True], "fan_in_fan_out": False},  # LoRA 2
    ]
    
    model = GPT2QuantModel(config, layer_bit_config, lora_configs=lora_configs)
    
    # 5. Test base model (no LoRA)
    print("\n5. Testing base model (no LoRA)...")
    with torch.no_grad():
        base_output = model(input_ids=input_ids, attention_mask=attention_mask)
        print(f"Base output shape: {base_output.shape}")
        print(f"Base output stats - Mean: {base_output.mean():.4f}, Std: {base_output.std():.4f}")
    
    # 6. Test with LoRA 1
    print("\n6. Testing with LoRA 1...")
    set_active_lora(model, [0] * num_layers)  # Activate LoRA 1 for all layers
    with torch.no_grad():
        lora1_output = model(input_ids=input_ids, attention_mask=attention_mask)
        diff1 = torch.abs(base_output - lora1_output)
        print(f"LoRA 1 output shape: {lora1_output.shape}")
        print(f"Difference from base - Mean: {diff1.mean():.4f}, Max: {diff1.max():.4f}")
    
    # 7. Test with LoRA 2
    print("\n7. Testing with LoRA 2...")
    set_active_lora(model, [1] * num_layers)  # Activate LoRA 2 for all layers
    with torch.no_grad():
        lora2_output = model(input_ids=input_ids, attention_mask=attention_mask)
        diff2 = torch.abs(base_output - lora2_output)
        print(f"LoRA 2 output shape: {lora2_output.shape}")
        print(f"Difference from base - Mean: {diff2.mean():.4f}, Max: {diff2.max():.4f}")
    
    # 8. Verify outputs are different
    print("\n8. Verifying outputs...")
    assert not torch.allclose(base_output, lora1_output, rtol=1e-5, atol=1e-5), "LoRA 1 did not affect output!"
    assert not torch.allclose(base_output, lora2_output, rtol=1e-5, atol=1e-5), "LoRA 2 did not affect output!"
    assert not torch.allclose(lora1_output, lora2_output, rtol=1e-5, atol=1e-5), "Different LoRAs produced same output!"
    
    print("\nAll tests completed successfully!")

if __name__ == "__main__":
    test_quantized_model() 