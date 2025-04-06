# quantized_gpt2_with_lora.py
import math
import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2PreTrainedModel
import loralib as lora  # using loralib for LoRA modules

# Import the quantization utilities (your existing implementation)
from utils_quant import QuantizeLinear, SymQuantizer

###############################################################################
# 1. Extend QuantizeLinear to support multiple LoRA modules
###############################################################################
class QuantizeLinearWithLoRA(QuantizeLinear):
    def __init__(self, *args, lora_configs=None, **kwargs):
        """
        lora_configs: a list of dictionaries, each dictionary contains the LoRA parameters
                      for one LoRA adapter (for example, keys 'r', 'alpha', 'dropout', etc.)
        """
        super().__init__(*args, **kwargs)
        self.lora_modules = nn.ModuleList()
        if lora_configs is not None:
            for cfg in lora_configs:
                # Create a LoRA module using loralib.MergedLinear.
                # We use the same in_features and out_features as the base layer.
                self.lora_modules.append(
                    lora.MergedLinear(
                        self.in_features, self.out_features,
                        r=cfg.get('r', 0),
                        lora_alpha=cfg.get('alpha', 1),
                        lora_dropout=cfg.get('dropout', 0.0),
                        enable_lora=cfg.get('enable_lora', True),
                        fan_in_fan_out=cfg.get('fan_in_fan_out', False),
                        merge_weights=False
                    )
                )
        # By default, no LoRA adapter is active.
        self.active_lora_idx = None

    def set_active_lora(self, idx):
        """Set the active LoRA adapter index (or None to disable)."""
        self.active_lora_idx = idx

    def forward(self, input_):
        # Compute base output with quantization as usual.
        base_out = super().forward(input_)
        # If a LoRA module is active, add its output.
        if self.active_lora_idx is not None:
            lora_out = self.lora_modules[self.active_lora_idx](input_)
            return base_out + lora_out
        return base_out

###############################################################################
# 2. Replace QuantizeLinear with QuantizeLinearWithLoRA in our modules.
# (For brevity, only key modules are shown.)
###############################################################################

class GPT2QuantAttention(nn.Module):
    def __init__(self, config: GPT2Config, w_bits, a_bits, lora_configs=None):
        super().__init__()
        self.embed_dim = config.n_embd
        self.num_heads = config.n_head
        self.head_dim = self.embed_dim // self.num_heads
        self.w_bits = w_bits
        self.a_bits = a_bits

        # Use our new QuantizeLinearWithLoRA for all projections.
        self.q_proj = QuantizeLinearWithLoRA(
            self.embed_dim, self.embed_dim,
            w_bits=self.w_bits, a_bits=self.a_bits, bias=True,
            lora_configs=lora_configs
        )
        self.k_proj = QuantizeLinearWithLoRA(
            self.embed_dim, self.embed_dim,
            w_bits=self.w_bits, a_bits=self.a_bits, bias=True,
            lora_configs=lora_configs
        )
        self.v_proj = QuantizeLinearWithLoRA(
            self.embed_dim, self.embed_dim,
            w_bits=self.w_bits, a_bits=self.a_bits, bias=True,
            lora_configs=lora_configs
        )
        self.out_proj = QuantizeLinearWithLoRA(
            self.embed_dim, self.embed_dim,
            w_bits=self.w_bits, a_bits=self.a_bits, bias=True,
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

class GPT2QuantMLP(nn.Module):
    def __init__(self, config: GPT2Config, w_bits, a_bits, lora_configs=None):
        super().__init__()
        self.w_bits = w_bits
        self.a_bits = a_bits
        inner_dim = config.n_inner if config.n_inner is not None else 4 * config.n_embd
        self.fc_in = QuantizeLinearWithLoRA(
            config.n_embd, inner_dim,
            w_bits=self.w_bits, a_bits=self.a_bits, bias=True,
            lora_configs=lora_configs
        )
        self.fc_out = QuantizeLinearWithLoRA(
            inner_dim, config.n_embd,
            w_bits=self.w_bits, a_bits=self.a_bits, bias=True,
            lora_configs=lora_configs
        )
        self.act = nn.GELU()

    def forward(self, hidden_states):
        hidden_states = self.fc_in(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.fc_out(hidden_states)
        return hidden_states

class GPT2QuantBlock(nn.Module):
    def __init__(self, config: GPT2Config, layer_bit_pair, lora_configs=None):
        super().__init__()
        w_bits, a_bits = layer_bit_pair
        self.ln_1 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.attn = GPT2QuantAttention(config, w_bits=w_bits, a_bits=a_bits, lora_configs=lora_configs)
        self.ln_2 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.mlp = GPT2QuantMLP(config, w_bits=w_bits, a_bits=a_bits, lora_configs=lora_configs)

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
# 3. GPT2QuantModel: Stack Blocks and Embeddings
###############################################################################
class GPT2QuantModel(GPT2PreTrainedModel):
    def __init__(self, config: GPT2Config, layer_bit_config, lora_configs=None):
        """
        Args:
            config: A standard GPT2Config.
            layer_bit_config: A list of [w_bits, a_bits] pairs for each transformer block.
            lora_configs: A list (or dict) of LoRA parameters to be applied in each linear layer.
                          For simplicity, the same lora_configs are applied to all linear layers here.
        """
        super().__init__(config)
        if len(layer_bit_config) != config.n_layer:
            raise ValueError("Length of layer_bit_config must equal n_layer.")
        self.layer_bit_config = layer_bit_config
        self.embed_dim = config.n_embd

        self.wte = nn.Embedding(config.vocab_size, self.embed_dim)
        self.wpe = nn.Embedding(config.n_positions, self.embed_dim)
        self.h = nn.ModuleList([
            GPT2QuantBlock(config, layer_bit_config[i], lora_configs=lora_configs)
            for i in range(config.n_layer)
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
# 4. Helper Function to Adaptively Activate LoRA Modules
###############################################################################
def set_active_lora(model, active_config):
    """
    active_config: a list of integers (or None) with length equal to the number of layers.
                   Each element specifies the active LoRA module index for that layer.
                   For example: [0, 1, 0, 2, ...] means layer 0 uses LoRA module 0, layer 1 uses module 1, etc.
    """
    for layer_idx, block in enumerate(model.h):
        active_idx = active_config[layer_idx]
        # Set active LoRA index for all QuantizeLinearWithLoRA modules in the block.
        for module in [block.attn.q_proj, block.attn.k_proj, block.attn.v_proj, block.attn.out_proj,
                       block.mlp.fc_in, block.mlp.fc_out]:
            if hasattr(module, "set_active_lora"):
                module.set_active_lora(active_idx)

###############################################################################
# 5. Testing LoRA Functionality
###############################################################################
if __name__ == "__main__":
    import torch
    from transformers import GPT2Config, GPT2Tokenizer

    print("Testing GPT2 with LoRA")
    print("=====================")
    
    # 1. Initialize models and tokenizer using default GPT2Config.
    print("\n1. Initializing Models...")
    base_config = GPT2Config()  # Using default configuration.
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    
    # Configure quantization and LoRA
    num_layers = base_config.n_layer
    layer_bit_config = [[8, 8]] * num_layers  # Use 8-bit for all layers
    
    # Define two different LoRA configurations.
    # Make sure enable_lora is a list, not a boolean.
    lora_configs = [
        {"r": 4, "alpha": 16, "dropout": 0.1, "enable_lora": [True], "fan_in_fan_out": False},  # LoRA 1
        {"r": 8, "alpha": 32, "dropout": 0.1, "enable_lora": [True], "fan_in_fan_out": False},  # LoRA 2
    ]
    
    from new_quantized_gpt2 import GPT2QuantModel, set_active_lora  # adjust import as needed
    
    model = GPT2QuantModel(base_config, layer_bit_config, lora_configs=lora_configs)
    
    # 2. Prepare test input.
    print("\n2. Preparing Test Input...")
    text = "Hello, how are you today?"
    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    
    # 3. Get base output (no LoRA).
    print("\n3. Testing Base Model Output...")
    model.zero_grad()
    with torch.no_grad():
        base_output = model(input_ids=input_ids, attention_mask=attention_mask)
        print(f"Base output shape: {base_output.shape}")
        print(f"Base output stats - Mean: {base_output.mean():.4f}, Std: {base_output.std():.4f}")
    
    # 4. Test LoRA 1.
    print("\n4. Testing LoRA 1...")
    set_active_lora(model, [0] * num_layers)  # Activate LoRA 1 for all layers.
    with torch.no_grad():
        lora1_output = model(input_ids=input_ids, attention_mask=attention_mask)
        diff1 = torch.abs(base_output - lora1_output)
        print(f"LoRA 1 output shape: {lora1_output.shape}")
        print(f"Difference from base - Mean: {diff1.mean():.4f}, Max: {diff1.max():.4f}")
    
    # 5. Test LoRA 2.
    print("\n5. Testing LoRA 2...")
    set_active_lora(model, [1] * num_layers)  # Activate LoRA 2 for all layers.
    with torch.no_grad():
        lora2_output = model(input_ids=input_ids, attention_mask=attention_mask)
        diff2 = torch.abs(base_output - lora2_output)
        print(f"LoRA 2 output shape: {lora2_output.shape}")
        print(f"Difference from base - Mean: {diff2.mean():.4f}, Max: {diff2.max():.4f}")
    
    # 6. Test mixed LoRA configuration.
    print("\n6. Testing Mixed LoRA Configuration...")
    mixed_config = [0 if i % 2 == 0 else 1 for i in range(num_layers)]  # Alternate between LoRA 1 and 2.
    set_active_lora(model, mixed_config)
    with torch.no_grad():
        mixed_output = model(input_ids=input_ids, attention_mask=attention_mask)
        diff_mixed = torch.abs(base_output - mixed_output)
        print(f"Mixed LoRA output shape: {mixed_output.shape}")
        print(f"Difference from base - Mean: {diff_mixed.mean():.4f}, Max: {diff_mixed.max():.4f}")
    
    # 7. Verify that LoRA is affecting the outputs.
    print("\n7. Verifying LoRA Functionality...")
    assert not torch.allclose(base_output, lora1_output, rtol=1e-5, atol=1e-5), "LoRA 1 did not affect output!"
    assert not torch.allclose(base_output, lora2_output, rtol=1e-5, atol=1e-5), "LoRA 2 did not affect output!"
    assert not torch.allclose(lora1_output, lora2_output, rtol=1e-5, atol=1e-5), "Different LoRAs produced the same output!"
    
    ###############################################################################
    # Additional Tests: Gradient Flow and Behavioral Tests
    ###############################################################################
    print("\n8. Testing Gradient Flow...")
    model.zero_grad()
    # Activate LoRA 1 for gradient test.
    set_active_lora(model, [0] * num_layers)
    output = model(input_ids=input_ids, attention_mask=attention_mask)
    loss = output.mean()
    loss.backward()
    
    # Check that some parameters have nonzero gradients.
    total_grad = 0.0
    count = 0
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.abs().sum().item()
            total_grad += grad_norm
            count += 1
    print(f"Total gradient sum across {count} parameters: {total_grad:.4f}")
    assert total_grad > 0, "No gradients flowed through the model!"
    
    print("\n9. Testing Behavioral Training Step...")
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    model.zero_grad()
    initial_loss = model(input_ids=input_ids, attention_mask=attention_mask).mean().item()
    print(f"Initial loss: {initial_loss:.6f}")
    for step in range(10):
        optimizer.zero_grad()
        output = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = output.mean()
        loss.backward()
        optimizer.step()
    final_loss = loss.item()
    print(f"Final loss after 10 steps: {final_loss:.6f}")
    assert final_loss < initial_loss, "Loss did not decrease after training steps!"
    
    print("\nAll gradient flow and behavioral tests completed successfully!")
