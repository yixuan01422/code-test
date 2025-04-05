import torch
import torch.nn as nn
from transformers import GPT2Model, GPT2Tokenizer
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention, GPT2MLP
from typing import Dict, Optional

def symmetric_quantize(x: torch.Tensor, bits: int = 8, per_channel: bool = False):
    if per_channel:
        # Per-channel quantization for weights
        # x shape: [out_features, in_features]
        max_val = torch.amax(torch.abs(x), dim=1, keepdim=True)
    else:
        # Per-token quantization for activations and KV cache
        # x shape: [batch_size, sequence_length, hidden_size]
        max_val = torch.amax(torch.abs(x), dim=2, keepdim=True)  # max along hidden_size
        max_val = torch.amax(max_val, dim=1, keepdim=True)  # max along sequence_length
    scale = max_val / (2 ** (bits - 1) - 1)
    x_quant = torch.round(x / scale)
    x_quant = torch.clamp(x_quant, -2 ** (bits - 1), 2 ** (bits - 1) - 1)
    return x_quant * scale

class QuantizedLayer(nn.Module):
    def __init__(self, original_layer, bits: int = 8):
        super().__init__()
        self.bits = bits
        self.in_features = original_layer.nx
        self.out_features = original_layer.nf
        self.bias = original_layer.bias
        # Use per-channel quantization for weights
        self.quantized_weight = symmetric_quantize(original_layer.weight, bits, per_channel=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size_out = x.size()[:-1] + (self.out_features,)
        x = x.view(-1, x.size(-1))
        x = nn.functional.linear(x, self.quantized_weight, self.bias)
        return x.view(size_out)

class ActivationQuantizer(nn.Module):
    def __init__(self, bits: int = 8):
        super().__init__()
        self.bits = bits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return symmetric_quantize(x, self.bits, per_channel=False)

class KVCacheQuantizer(nn.Module):
    def __init__(self, bits: int = 8):
        super().__init__()
        self.bits = bits

    def forward(self, k: torch.Tensor, v: torch.Tensor) -> tuple:
        return symmetric_quantize(k, self.bits, per_channel=False), symmetric_quantize(v, self.bits, per_channel=False)

class QuantizedAttention(nn.Module):
    def __init__(self, original_attn, weight_bits: int = 8, act_bits: int = 8, kv_bits: int = 8):
        super().__init__()
        self.c_attn = QuantizedLayer(original_attn.c_attn, weight_bits)
        self.c_proj = QuantizedLayer(original_attn.c_proj, weight_bits)
        self.attn_dropout = original_attn.attn_dropout
        self.resid_dropout = original_attn.resid_dropout
        
        # Quantizers
        self.act_quantizer = ActivationQuantizer(act_bits)
        self.kv_cache_quantizer = KVCacheQuantizer(kv_bits)
        
        # Original attention parameters
        self.num_heads = original_attn.num_heads
        self.head_dim = original_attn.head_dim
        self.scale = 1.0 / (self.head_dim ** 0.5)
        
    def _split_heads(self, x, num_heads, head_dim):
        new_shape = x.size()[:-1] + (num_heads, head_dim)
        x = x.view(new_shape)
        return x.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_dim)

    def forward(self, hidden_states, layer_past=None, attention_mask=None, head_mask=None):
        query, key, value = self.c_attn(hidden_states).split(self.head_dim * self.num_heads, dim=2)
        
        # Split heads
        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)

        if layer_past is not None:
            past_key, past_value = layer_past
            # Quantize past KV cache
            past_key, past_value = self.kv_cache_quantizer(past_key, past_value)
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)
        
        # Quantize current KV
        key, value = self.kv_cache_quantizer(key, value)
        present = (key, value)

        # Compute attention
        attn_output = torch.matmul(query, key.transpose(-1, -2)) * self.scale
        if attention_mask is not None:
            attn_output = attn_output + attention_mask
        attn_output = nn.functional.softmax(attn_output, dim=-1)
        attn_output = self.attn_dropout(attn_output)

        if head_mask is not None:
            attn_output = attn_output * head_mask

        # Combine heads
        attn_output = torch.matmul(attn_output, value)
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous()
        attn_output = attn_output.view(attn_output.size()[:-2] + (self.num_heads * self.head_dim,))

        # Output projection and quantization
        attn_output = self.act_quantizer(attn_output)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        return attn_output, present

class QuantizedGPT2(nn.Module):
    def __init__(self, model_name: str = 'gpt2', weight_bits: int = 8, act_bits: int = 8, kv_bits: int = 8):
        super().__init__()
        self.model = GPT2Model.from_pretrained(model_name)
        self.weight_bits = weight_bits
        self.act_bits = act_bits
        self.kv_bits = kv_bits
        self._quantize_model()

    def _quantize_model(self):
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Module):
                # Handle attention blocks
                if isinstance(module, GPT2Attention):
                    # Replace the entire attention block with quantized version
                    module.attn = QuantizedAttention(
                        module, 
                        self.weight_bits, 
                        self.act_bits, 
                        self.kv_bits
                    )
                
                # Handle MLP blocks
                elif isinstance(module, GPT2MLP):
                    # Quantize the MLP layers
                    quant_c_fc = QuantizedLayer(module.c_fc, self.weight_bits)
                    quant_c_proj = QuantizedLayer(module.c_proj, self.weight_bits)
                    
                    # Add activation quantizers
                    act_quantizer = ActivationQuantizer(self.act_bits)
                    
                    # Replace the layers
                    module.c_fc = quant_c_fc
                    module.c_proj = quant_c_proj
                    
                    # Store the activation quantizer
                    module.act_quantizer = act_quantizer
                    
                    # Modify the forward pass to include activation quantization
                    original_forward = module.forward
                    def new_forward(*args, **kwargs):
                        x = original_forward(*args, **kwargs)
                        return module.act_quantizer(x)
                    module.forward = new_forward


    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

if __name__ == "__main__":
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    
    # Initialize model with uniform bit widths
    model = QuantizedGPT2(
        weight_bits=8,  # All weights use 8 bits
        act_bits=8,     # All activations use 8 bits
        kv_bits=8       # All KV cache use 8 bits
    )
    # model.eval()

    # text = "Hello, I am a language model that has been"
    # inputs = tokenizer(text, return_tensors="pt")
    
    # with torch.no_grad():
    #     outputs = model(**inputs)
    #     hidden_states = outputs.last_hidden_state
    
    # print(f"\nInput text: {text}")
    # print(f"Output shape: {hidden_states.shape}")
    # print(f"Sample output values:\n{hidden_states[0, 0, :10]}")

    # def print_layer_info(name, module):
    #     if hasattr(module, 'bits'):
    #         print(f"\nLayer: {name}")
    #         print(f"Quantization bits: {module.bits}")
    #         if hasattr(module, 'quantized_weight'):
    #             print(f"Weight shape: {module.quantized_weight.shape}")

    # for name, module in model.named_modules():
    #     print_layer_info(name, module)
