import torch
import torch.nn as nn
from transformers import GPT2Model, GPT2Tokenizer
from typing import Dict, Optional

def symmetric_quantize(x: torch.Tensor, bits: int = 8):
    max_val = torch.max(torch.abs(x))
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
        self.quantized_weight = symmetric_quantize(original_layer.weight, bits)

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
        return symmetric_quantize(x, self.bits)

class KVCacheQuantizer(nn.Module):
    def __init__(self, bits: int = 8):
        super().__init__()
        self.bits = bits

    def forward(self, k: torch.Tensor, v: torch.Tensor) -> tuple:
        # Quantize K and V cache separately
        k_quant = symmetric_quantize(k, self.bits)
        v_quant = symmetric_quantize(v, self.bits)
        return k_quant, v_quant

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
        self.split_size = original_attn.split_size
        self.scale = original_attn.scale
        
    def _split_heads(self, x, num_heads, head_dim):
        new_shape = x.size()[:-1] + (num_heads, head_dim)
        x = x.view(new_shape)
        return x.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_dim)

    def forward(self, hidden_states, layer_past=None, attention_mask=None, head_mask=None):
        query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
        
        # Split heads
        query = self._split_heads(query, self.num_heads, self.split_size // self.num_heads)
        key = self._split_heads(key, self.num_heads, self.split_size // self.num_heads)
        value = self._split_heads(value, self.num_heads, self.split_size // self.num_heads)

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
        attn_output = attn_output.view(attn_output.size()[:-2] + (self.num_heads * self.split_size // self.num_heads,))

        # Output projection and quantization
        attn_output = self.act_quantizer(attn_output)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        return attn_output, present

class QuantizedGPT2(nn.Module):
    def __init__(self, model_name: str = 'gpt2', bit_config: Optional[Dict] = None):
        super().__init__()
        self.model = GPT2Model.from_pretrained(model_name)
        self.bit_config = bit_config
        self._quantize_model()

    def _quantize_model(self):
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Module):
                if hasattr(module, 'nf'):
                    for pattern, bits in self.bit_config["weight"].items():
                        if name.endswith(pattern):
                            parent = self._get_parent_module(name)
                            if 'attn' in pattern and hasattr(parent, 'attn'):
                                # Replace entire attention block
                                weight_bits = self.bit_config["weight"][pattern]
                                act_bits = self.bit_config["activation"].get(pattern, 8)
                                kv_bits = self.bit_config.get("kv_cache", {}).get(pattern, 8)
                                quant_attn = QuantizedAttention(parent.attn, weight_bits, act_bits, kv_bits)
                                setattr(parent, 'attn', quant_attn)
                            else:
                                # Regular layer quantization
                                quant_layer = QuantizedLayer(module, bits)
                                setattr(parent, name.split('.')[-1], quant_layer)
                                
                                if pattern in self.bit_config.get("activation", {}):
                                    act_bits = self.bit_config["activation"][pattern]
                                    quantizer = ActivationQuantizer(act_bits)
                                    setattr(parent, f"{name.split('.')[-1]}_act_quant", quantizer)
                                    self._modify_forward_pass(parent, name.split('.')[-1])
                            break

    def _modify_forward_pass(self, parent_module, layer_name):
        original_forward = parent_module.forward
        quantizer = getattr(parent_module, f"{layer_name}_act_quant")
        
        def new_forward(*args, **kwargs):
            outputs = original_forward(*args, **kwargs)
            if isinstance(outputs, tuple):
                return (quantizer(outputs[0]),) + outputs[1:]
            return quantizer(outputs)
            
        parent_module.forward = new_forward

    def _get_parent_module(self, name: str) -> nn.Module:
        if not name:
            return self.model
        parts = name.split('.')
        current = self.model
        for part in parts:
            current = getattr(current, part)
        return current

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

if __name__ == "__main__":
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    bit_config = {
        "weight": {
            'attn.c_attn': 4,
            'attn.c_proj': 8,
            'mlp.c_fc': 6,
            'mlp.c_proj': 8
        },
        "activation": {
            'attn.c_attn': 8,
            'attn.c_proj': 8,
            'mlp.c_fc': 8,
            'mlp.c_proj': 8
        },
        "kv_cache": {
            'attn.c_attn': 8  # 8-bit quantization for KV cache
        }
    }
    
    model = QuantizedGPT2(bit_config=bit_config)
    model.eval()

    text = "Hello, I am a language model that has been"
    inputs = tokenizer(text, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
        hidden_states = outputs.last_hidden_state
    
    print(f"\nInput text: {text}")
    print(f"Output shape: {hidden_states.shape}")
    print(f"Sample output values:\n{hidden_states[0, 0, :10]}")

    def print_layer_info(name, module):
        if hasattr(module, 'bits'):
            print(f"\nLayer: {name}")
            print(f"Quantization bits: {module.bits}")
            if hasattr(module, 'quantized_weight'):
                print(f"Weight shape: {module.quantized_weight.shape}")

    for name, module in model.named_modules():
        print_layer_info(name, module)
