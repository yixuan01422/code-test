import torch
import torch.nn as nn
from transformers import GPT2Model, GPT2Tokenizer
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention, GPT2MLP
from typing import Dict, Optional
from quantize import Quantize, calculate_qparams

def symmetric_quantize(x: torch.Tensor, bits: int = 8, per_channel: bool = False):
    if per_channel:
        # Per-channel quantization for weights
        # x shape: [out_features, in_features]
        # Compute stats independently for each output channel
        qparams = calculate_qparams(
            x, 
            num_bits=bits, 
            flatten_dims=(1, -1),  # Flatten all dims except channel dim
            reduce_dim=0,  # Reduce across channel dimension
            reduce_type='extreme'  # Use min/max for better precision
        )
        return Quantize(x, qparams=qparams, dequantize=True, signed=True)
    else:
        # Per-tensor quantization for activations and KV cache
        # x shape: [batch_size, sequence_length, hidden_size]
        # Compute stats across the entire tensor
        qparams = calculate_qparams(
            x, 
            num_bits=bits,
            flatten_dims=(0, -1),  # Flatten entire tensor
            reduce_dim=None,  # Global stats
            reduce_type='extreme'  # Use min/max for better precision
        )
        return Quantize(x, qparams=qparams, dequantize=True, signed=True)

class QuantizedLayer(nn.Module):
    def __init__(self, original_layer, bits: int = 8):
        super().__init__()
        self.bits = bits
        weight = original_layer.weight
        # For c_attn layer, weight shape should be [768, 2304] -> [2304, 768]
        if weight.size(1) == 2304:  # This is the c_attn layer
            weight = weight.transpose(0, 1)
        self.in_features = weight.size(1)
        self.out_features = weight.size(0)
        self.bias = original_layer.bias
        if self.bias is not None and weight.size(1) == 2304:  # For c_attn layer
            self.bias = self.bias.view(self.out_features)
        # Use per-channel quantization for weights
        self.quantized_weight = symmetric_quantize(weight, bits, per_channel=True)

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
        config = original_attn.config
        self.num_heads = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.embed_dim = config.n_embd
        
        # Print dimensions for debugging
        print(f"Attention dimensions:")
        print(f"- Embed dim: {self.embed_dim}")
        print(f"- Num heads: {self.num_heads}")
        print(f"- Head dim: {self.head_dim}")
        print(f"- QKV dim: {3 * self.embed_dim}")
        
        # Initialize quantized layers with correct dimensions
        self.c_attn = QuantizedLayer(original_attn.c_attn, weight_bits)
        self.c_proj = QuantizedLayer(original_attn.c_proj, weight_bits)
        
        # Original attention parameters
        self.scale = 1.0 / (self.head_dim ** 0.5)
        self.attn_dropout = original_attn.attn_dropout
        self.resid_dropout = original_attn.resid_dropout
        
        # Quantizers
        self.act_quantizer = ActivationQuantizer(act_bits)
        self.kv_cache_quantizer = KVCacheQuantizer(kv_bits)

    def _split_heads(self, x, num_heads, head_dim):
        new_shape = x.size()[:-1] + (num_heads, head_dim)
        x = x.view(new_shape)
        return x.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_dim)

    def forward(self, hidden_states, layer_past=None, attention_mask=None, head_mask=None):
        qkv = self.c_attn(hidden_states)
        all_head_size = self.num_heads * self.head_dim
        query, key, value = qkv.split(all_head_size, dim=2)
        
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
        attn_output = attn_output.view(attn_output.size()[:-2] + (self.embed_dim,))

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
    pass
