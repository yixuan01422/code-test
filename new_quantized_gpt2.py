import math
import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2PreTrainedModel
import loralib as lora  # using loralib for LoRA modules

# Import the base quantization utilities from utils_quant.py
from utils_quant import QuantizeLinear, SymQuantizer

###############################################################################
# 1. QuantizeLinearSwitchable: Wrap the base QuantizeLinear to support multiple candidate bit‑widths.
###############################################################################
class QuantizeLinearSwitchable(nn.Module):
    def __init__(self, in_features, out_features, candidate_w_bits, candidate_a_bits, **kwargs):
        """
        candidate_w_bits: list of candidate weight bit‑widths.
        candidate_a_bits: list of candidate activation bit‑widths.
        """
        super().__init__()
        # Instantiate the base quantized linear using the first candidate.
        self.base_linear = QuantizeLinear(in_features, out_features,
                                          w_bits=candidate_w_bits[0],
                                          a_bits=candidate_a_bits[0],
                                          **kwargs)
        self.candidate_w_bits = candidate_w_bits
        self.candidate_a_bits = candidate_a_bits
        self.active_quant_idx = 0  # default candidate index

    def set_active_bitwidth(self, idx):
        """Switch to the candidate bit‑width at index idx."""
        self.active_quant_idx = idx
        self.base_linear.w_bits = self.candidate_w_bits[idx]
        self.base_linear.a_bits = self.candidate_a_bits[idx]

    def forward(self, input_):
        return self.base_linear(input_)

###############################################################################
# 2. QuantizeLinearWithLoRASwitchable: Wrap QuantizeLinearSwitchable and add multiple LoRA modules.
###############################################################################
class QuantizeLinearWithLoRASwitchable(nn.Module):
    def __init__(self, in_features, out_features, candidate_w_bits, candidate_a_bits, 
                 lora_configs=None, **kwargs):
        """
        lora_configs: a list of dictionaries, each containing LoRA parameters
                      for one LoRA adapter (e.g., keys 'r', 'alpha', 'dropout', etc.)
        """
        super().__init__()
        # Create the switchable quantization module.
        self.switchable = QuantizeLinearSwitchable(in_features, out_features, 
                                                   candidate_w_bits, candidate_a_bits, **kwargs)
        self.lora_modules = nn.ModuleList()
        if lora_configs is not None:
            for cfg in lora_configs:
                self.lora_modules.append(
                    lora.MergedLinear(
                        in_features, out_features,
                        r=cfg.get('r', 0),
                        lora_alpha=cfg.get('alpha', 1),
                        lora_dropout=cfg.get('dropout', 0.0),
                        # Ensure enable_lora is a list
                        enable_lora=cfg.get('enable_lora', [True]),
                        fan_in_fan_out=cfg.get('fan_in_fan_out', False),
                        merge_weights=False
                    )
                )
        # By default, no LoRA adapter is active.
        self.active_lora_idx = None

    def set_active_lora(self, idx):
        self.active_lora_idx = idx

    def set_active_bitwidth(self, idx):
        self.switchable.set_active_bitwidth(idx)

    def forward(self, input_):
        base_out = self.switchable(input_)
        if self.active_lora_idx is not None:
            lora_out = self.lora_modules[self.active_lora_idx](input_)
            return base_out + lora_out
        return base_out

###############################################################################
# 3. GPT2QuantAttention using our new switchable module with LoRA support
###############################################################################
class GPT2QuantAttention(nn.Module):
    def __init__(self, config: GPT2Config, w_bits, a_bits, 
                 candidate_w_bits=None, candidate_a_bits=None, lora_configs=None):
        super().__init__()
        self.embed_dim = config.n_embd
        self.num_heads = config.n_head
        self.head_dim = self.embed_dim // self.num_heads
        self.w_bits = w_bits
        self.a_bits = a_bits
        
        # If candidate lists are not provided, default to a single candidate.
        candidate_w_bits = candidate_w_bits if candidate_w_bits is not None else [w_bits]
        candidate_a_bits = candidate_a_bits if candidate_a_bits is not None else [a_bits]
        
        self.q_proj = QuantizeLinearWithLoRASwitchable(
            self.embed_dim, self.embed_dim,
            candidate_w_bits=candidate_w_bits,
            candidate_a_bits=candidate_a_bits,
            bias=True,
            lora_configs=lora_configs
        )
        self.k_proj = QuantizeLinearWithLoRASwitchable(
            self.embed_dim, self.embed_dim,
            candidate_w_bits=candidate_w_bits,
            candidate_a_bits=candidate_a_bits,
            bias=True,
            lora_configs=lora_configs
        )
        self.v_proj = QuantizeLinearWithLoRASwitchable(
            self.embed_dim, self.embed_dim,
            candidate_w_bits=candidate_w_bits,
            candidate_a_bits=candidate_a_bits,
            bias=True,
            lora_configs=lora_configs
        )
        self.out_proj = QuantizeLinearWithLoRASwitchable(
            self.embed_dim, self.embed_dim,
            candidate_w_bits=candidate_w_bits,
            candidate_a_bits=candidate_a_bits,
            bias=True,
            lora_configs=lora_configs
        )
        
    def _shape(self, x, seq_len, bsz):
        return x.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
    
    def forward(self, hidden_states, attention_mask=None):
        bsz, seq_len, _ = hidden_states.size()
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        
        q = self._shape(q, seq_len, bsz)
        k = self._shape(k, seq_len, bsz)
        v = self._shape(v, seq_len, bsz)
        
        attn_weights = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            attention_mask = attention_mask.view(bsz, 1, 1, seq_len)
            attention_mask = (1.0 - attention_mask) * -10000.0
            attn_weights = attn_weights + attention_mask
        attn_probs = nn.functional.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_probs, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, seq_len, self.embed_dim)
        attn_output = self.out_proj(attn_output)
        return attn_output

###############################################################################
# 4. GPT2QuantMLP using our new switchable module with LoRA support
###############################################################################
class GPT2QuantMLP(nn.Module):
    def __init__(self, config: GPT2Config, w_bits, a_bits, 
                 candidate_w_bits=None, candidate_a_bits=None, lora_configs=None):
        super().__init__()
        self.w_bits = w_bits
        self.a_bits = a_bits
        inner_dim = config.n_inner if config.n_inner is not None else 4 * config.n_embd
        self.fc_in = QuantizeLinearWithLoRASwitchable(
            config.n_embd, inner_dim,
            candidate_w_bits=candidate_w_bits, candidate_a_bits=candidate_a_bits,
            bias=True,
            lora_configs=lora_configs
        )
        self.fc_out = QuantizeLinearWithLoRASwitchable(
            inner_dim, config.n_embd,
            candidate_w_bits=candidate_w_bits, candidate_a_bits=candidate_a_bits,
            bias=True,
            lora_configs=lora_configs
        )
        self.act = nn.GELU()
        
    def forward(self, hidden_states):
        hidden_states = self.fc_in(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.fc_out(hidden_states)
        return hidden_states

###############################################################################
# 5. GPT2QuantBlock: Transformer Block using our modules
###############################################################################
class GPT2QuantBlock(nn.Module):
    def __init__(self, config: GPT2Config, layer_bit_pair, 
                 candidate_w_bits=None, candidate_a_bits=None, lora_configs=None):
        super().__init__()
        w_bits, a_bits = layer_bit_pair
        self.ln_1 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.attn = GPT2QuantAttention(
            config, w_bits=w_bits, a_bits=a_bits, lora_configs=lora_configs,
            candidate_w_bits=candidate_w_bits, candidate_a_bits=candidate_a_bits
        )
        self.ln_2 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.mlp = GPT2QuantMLP(
            config, w_bits=w_bits, a_bits=a_bits, lora_configs=lora_configs,
            candidate_w_bits=candidate_w_bits, candidate_a_bits=candidate_a_bits
        )
        
    def forward(self, hidden_states, attention_mask=None):
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_output = self.attn(hidden_states, attention_mask=attention_mask)
        hidden_states = residual + attn_output
        
        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        mlp_output = self.mlp(hidden_states)
        hidden_states = residual + mlp_output
        
        return hidden_states

###############################################################################
# 6. GPT2QuantModel: Stack Blocks and Embeddings
###############################################################################
class GPT2QuantModel(GPT2PreTrainedModel):
    def __init__(self, config: GPT2Config, layer_bit_config, 
                 candidate_w_bits=None, candidate_a_bits=None, lora_configs=None):
        """
        Args:
            config: A standard GPT2Config.
            layer_bit_config: A list of [w_bits, a_bits] pairs for each transformer block.
            candidate_w_bits: A list of candidate weight bit‑widths.
            candidate_a_bits: A list of candidate activation bit‑widths.
            lora_configs: A list of LoRA parameter dictionaries to be applied in each linear layer.
        """
        super().__init__(config)
        if len(layer_bit_config) != config.n_layer:
            raise ValueError("Length of layer_bit_config must equal n_layer.")
        self.config = config
        self.layer_bit_config = layer_bit_config
        self.candidate_w_bits = candidate_w_bits
        self.candidate_a_bits = candidate_a_bits
        self.lora_configs = lora_configs
        self.embed_dim = config.n_embd
        
        self.wte = nn.Embedding(config.vocab_size, self.embed_dim)
        self.wpe = nn.Embedding(config.n_positions, self.embed_dim)
        self.h = nn.ModuleList([
            GPT2QuantBlock(
                config, layer_bit_config[i],
                candidate_w_bits=candidate_w_bits, candidate_a_bits=candidate_a_bits,
                lora_configs=lora_configs
            ) for i in range(config.n_layer)
        ])
        self.ln_f = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)
        self.register_buffer("position_ids", torch.arange(config.n_positions).expand((1, -1)))
        self.post_init()
        
    def forward(self, input_ids, attention_mask=None):
        bsz, seq_len = input_ids.size()
        position_ids = self.position_ids[:, :seq_len].to(input_ids.device)
        inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds
        
        for block in self.h:
            hidden_states = block(hidden_states, attention_mask=attention_mask)
        hidden_states = self.ln_f(hidden_states)
        return hidden_states

###############################################################################
# 7. Helper Functions to Adaptively Activate LoRA and Quantization Bit-Widths
###############################################################################
def set_active_lora(model, active_config):
    """
    active_config: a list of integers with length equal to the number of layers.
                   Each element specifies the active LoRA module index for that layer.
    """
    for layer_idx, block in enumerate(model.h):
        active_idx = active_config[layer_idx]
        for module in [block.attn.q_proj, block.attn.k_proj, block.attn.v_proj, block.attn.out_proj,
                       block.mlp.fc_in, block.mlp.fc_out]:
            if hasattr(module, "set_active_lora"):
                module.set_active_lora(active_idx)

def set_active_quantization(model, active_quant_config):
    """
    active_quant_config: a list of integers with length equal to the number of layers.
                         Each element specifies the active candidate index for the quantization
                         parameters for that layer.
    """
    for layer_idx, block in enumerate(model.h):
        active_idx = active_quant_config[layer_idx]
        for module in [block.attn.q_proj, block.attn.k_proj, block.attn.v_proj, block.attn.out_proj,
                       block.mlp.fc_in, block.mlp.fc_out]:
            if hasattr(module, "set_active_bitwidth"):
                module.set_active_bitwidth(active_idx)

###############################################################################
# 8. Testing LoRA and Switchable Quantization Functionality
###############################################################################
if __name__ == "__main__":
    import torch
    from transformers import GPT2Config, GPT2Tokenizer

    print("Testing GPT2 with LoRA and Switchable Quantization")
    print("====================================================")
    
    # 1. Initialize models and tokenizer using default GPT2Config.
    print("\n1. Initializing Models...")
    base_config = GPT2Config()  # Using default configuration.
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    
    # Configure quantization and LoRA.
    num_layers = base_config.n_layer
    layer_bit_config = [[8, 8]] * num_layers  # Default bit configuration for all layers.
    
    # Define candidate quantization bit-widths.
    candidate_w_bits = [4, 8]  # e.g., candidate weight bit-widths: 4-bit and 8-bit.
    candidate_a_bits = [4, 8]  # Candidate activation bit-widths.
    
    # Define two different LoRA configurations.
    lora_configs = [
        {"r": 4, "alpha": 16, "dropout": 0.1, "enable_lora": [True], "fan_in_fan_out": False},  # LoRA 1.
        {"r": 8, "alpha": 32, "dropout": 0.1, "enable_lora": [True], "fan_in_fan_out": False},  # LoRA 2.
    ]
    
    from new_quantized_gpt2 import GPT2QuantModel, set_active_lora, set_active_quantization  # adjust import as needed
    
    model = GPT2QuantModel(base_config, layer_bit_config, lora_configs=lora_configs,
                           candidate_w_bits=candidate_w_bits, candidate_a_bits=candidate_a_bits)
    
    # 2. Prepare test input.
    print("\n2. Preparing Test Input...")
    text = "Hello, how are you today?"
    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    
    # 3. Get base output (no LoRA, default quantization candidate index 0).
    print("\n3. Testing Base Model Output...")
    model.zero_grad()
    set_active_quantization(model, [0] * num_layers)
    with torch.no_grad():
        base_output = model(input_ids=input_ids, attention_mask=attention_mask)
        print(f"Base output shape: {base_output.shape}")
        print(f"Base output stats - Mean: {base_output.mean():.4f}, Std: {base_output.std():.4f}")
    
    # 4. Test LoRA 1 with quantization candidate index 0.
    print("\n4. Testing LoRA 1 with quantization candidate index 0...")
    set_active_lora(model, [0] * num_layers)
    with torch.no_grad():
        lora1_output = model(input_ids=input_ids, attention_mask=attention_mask)
        diff1 = torch.abs(base_output - lora1_output)
        print(f"LoRA 1 output shape: {lora1_output.shape}")
        print(f"Difference from base - Mean: {diff1.mean():.4f}, Max: {diff1.max():.4f}")
    
    # 5. Test switching quantization candidate to index 1 (e.g., using 8-bit instead of 4-bit).
    print("\n5. Testing switching quantization candidate to index 1...")
    set_active_quantization(model, [1] * num_layers)
    with torch.no_grad():
        switched_output = model(input_ids=input_ids, attention_mask=attention_mask)
        diff_switched = torch.abs(base_output - switched_output)
        print(f"Switched quantization output shape: {switched_output.shape}")
        print(f"Difference from base - Mean: {diff_switched.mean():.4f}, Max: {diff_switched.max():.4f}")
    
    # 6. Test mixed LoRA configuration.
    print("\n6. Testing Mixed LoRA Configuration...")
    mixed_config = [0 if i % 2 == 0 else 1 for i in range(num_layers)]
    set_active_lora(model, mixed_config)
    with torch.no_grad():
        mixed_output = model(input_ids=input_ids, attention_mask=attention_mask)
        diff_mixed = torch.abs(base_output - mixed_output)
        print(f"Mixed LoRA output shape: {mixed_output.shape}")
        print(f"Difference from base - Mean: {diff_mixed.mean():.4f}, Max: {diff_mixed.max():.4f}")
    
    # 7. Verify that LoRA and quantization switching are affecting the outputs.
    print("\n7. Verifying LoRA and Quantization Switching Functionality...")
    assert not torch.allclose(base_output, lora1_output, rtol=1e-5, atol=1e-5), "LoRA 1 did not affect output!"
    assert not torch.allclose(base_output, switched_output, rtol=1e-5, atol=1e-5), "Quantization switching did not affect output!"
    print("\nAll tests completed successfully!")
