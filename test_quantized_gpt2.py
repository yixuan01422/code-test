import torch
import torch.nn as nn
from transformers import GPT2Model
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention
from quantized_gpt2 import QuantizedAttention, QuantizedLayer, KVCacheQuantizer, symmetric_quantize

def test_quantized_attention():
    # Create a small test input
    batch_size, seq_len, hidden_size = 2, 4, 768
    hidden_states = torch.randn(batch_size, seq_len, hidden_size)
    
    # Initialize original and quantized attention
    original_attn = GPT2Attention(GPT2Model.from_pretrained('gpt2').config)
    quantized_attn = QuantizedAttention(original_attn, weight_bits=4, act_bits=4, kv_bits=4)
    
    # Test forward pass
    with torch.no_grad():
        # Run original attention
        orig_output, orig_present = original_attn(hidden_states)
        
        # Run quantized attention
        quant_output, quant_present = quantized_attn(hidden_states)
        
        # Print shapes and sample values
        print("\nShape comparison:")
        print(f"Original output shape: {orig_output.shape}")
        print(f"Quantized output shape: {quant_output.shape}")
        
        print("\nValue comparison:")
        print(f"Original output mean: {orig_output.mean():.4f}")
        print(f"Quantized output mean: {quant_output.mean():.4f}")
        print(f"Original output std: {orig_output.std():.4f}")
        print(f"Quantized output std: {quant_output.std():.4f}")
        
        # Calculate error metrics
        mse = torch.mean((orig_output - quant_output) ** 2)
        max_error = torch.max(torch.abs(orig_output - quant_output))
        print(f"\nError metrics:")
        print(f"MSE: {mse:.6f}")
        print(f"Max absolute error: {max_error:.6f}")
        
        # Test KV cache
        print("\nTesting KV cache:")
        # Create past key and value
        past_key = torch.randn(batch_size, quantized_attn.num_heads, 2, quantized_attn.head_dim)
        past_value = torch.randn(batch_size, quantized_attn.num_heads, 2, quantized_attn.head_dim)
        layer_past = (past_key, past_value)
        
        # Run with past
        quant_output_past, quant_present_past = quantized_attn(hidden_states, layer_past=layer_past)
        print(f"Output with KV cache shape: {quant_output_past.shape}")
        print(f"Present KV cache key shape: {quant_present_past[0].shape}")
        print(f"Present KV cache value shape: {quant_present_past[1].shape}")
        
        # Verify attention mask handling
        print("\nTesting attention mask:")
        attention_mask = torch.ones(batch_size, 1, 1, seq_len)
        attention_mask[:, :, :, -1] = 0  # Mask out last token
        quant_output_masked, _ = quantized_attn(hidden_states, attention_mask=attention_mask)
        print(f"Output with mask shape: {quant_output_masked.shape}")
        
        # Compare quantization levels
        print("\nQuantization analysis:")
        # Get unique values in quantized weights
        c_attn_unique = torch.unique(quantized_attn.c_attn.quantized_weight).numel()
        c_proj_unique = torch.unique(quantized_attn.c_proj.quantized_weight).numel()
        print(f"Number of unique values in c_attn weights: {c_attn_unique}")
        print(f"Number of unique values in c_proj weights: {c_proj_unique}")
        print(f"Expected max unique values for 4-bit: {2**4}")
        
        # Memory usage comparison
        orig_size = orig_output.element_size() * orig_output.nelement()
        quant_size = quant_output.element_size() * quant_output.nelement()
        print(f"\nMemory usage:")
        print(f"Original size: {orig_size / 1024:.2f} KB")
        print(f"Quantized size: {quant_size / 1024:.2f} KB")
        print(f"Compression ratio: {orig_size / quant_size:.2f}x")

def test_quantization():
    # Test per-channel quantization (weights)
    print("\n=== Testing Per-Channel Quantization (Weights) ===")
    # Create a weight matrix [out_features, in_features]
    weight = torch.randn(4, 6)  # [4 output channels, 6 input features]
    print(f"Original weight shape: {weight.shape}")
    
    # Quantize with per-channel
    weight_q = symmetric_quantize(weight, bits=4, per_channel=True)
    
    # Check shape preservation
    assert weight_q.shape == weight.shape, "Shape mismatch after quantization"
    
    # Check value ranges
    print(f"Original weight range: [{weight.min():.4f}, {weight.max():.4f}]")
    print(f"Quantized weight range: [{weight_q.min():.4f}, {weight_q.max():.4f}]")
    
    # Check unique values per channel
    for i in range(weight.shape[0]):
        unique_vals = torch.unique(weight_q[i])
        print(f"Channel {i} unique values: {unique_vals.numel()}")
        assert unique_vals.numel() <= 2**4, f"Channel {i} has too many unique values"
    
    # Test per-token quantization (activations)
    print("\n=== Testing Per-Token Quantization (Activations) ===")
    # Create activation tensor [batch_size, seq_len, hidden_size]
    activation = torch.randn(2, 3, 4)  # [batch=2, seq_len=3, hidden=4]
    print(f"Original activation shape: {activation.shape}")
    
    # Quantize with per-token
    activation_q = symmetric_quantize(activation, bits=4, per_channel=False)
    
    # Check shape preservation
    assert activation_q.shape == activation.shape, "Shape mismatch after quantization"
    
    # Check value ranges
    print(f"Original activation range: [{activation.min():.4f}, {activation.max():.4f}]")
    print(f"Quantized activation range: [{activation_q.min():.4f}, {activation_q.max():.4f}]")
    
    # Check unique values across entire tensor
    unique_vals = torch.unique(activation_q)
    print(f"Total unique values in activation: {unique_vals.numel()}")
    assert unique_vals.numel() <= 2**4, "Too many unique values in activation"
    
    # Test memory savings
    print("\n=== Testing Memory Savings ===")
    original_size = weight.element_size() * weight.nelement()
    quantized_size = weight_q.element_size() * weight_q.nelement()
    print(f"Original weight size: {original_size} bytes")
    print(f"Quantized weight size: {quantized_size} bytes")
    print(f"Compression ratio: {original_size/quantized_size:.2f}x")
    
    # Test forward pass through QuantizedLayer
    print("\n=== Testing QuantizedLayer Forward Pass ===")
    layer = QuantizedLayer(nn.Linear(6, 4), bits=4)
    x = torch.randn(2, 6)  # [batch=2, features=6]
    y = layer(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    assert y.shape == (2, 4), "Output shape mismatch in QuantizedLayer"
    
    # Test KV cache quantization
    print("\n=== Testing KV Cache Quantization ===")
    kv_quantizer = KVCacheQuantizer(bits=4)
    k = torch.randn(2, 3, 4)  # [batch=2, seq_len=3, hidden=4]
    v = torch.randn(2, 3, 4)
    k_q, v_q = kv_quantizer(k, v)
    assert k_q.shape == k.shape, "Key shape mismatch after quantization"
    assert v_q.shape == v.shape, "Value shape mismatch after quantization"
    
    print("\nAll tests passed successfully!")

if __name__ == "__main__":
    test_quantized_attention()
    test_quantization() 