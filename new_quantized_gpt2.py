# quantized_gpt2.py
import math
import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2PreTrainedModel

# Import the quantization utilities from utils_quant.py
from utils_quant import QuantizeLinear, SymQuantizer  # Ensure utils_quant.py is in your PYTHONPATH

###############################################################################
# 1. Quantized Self-Attention Block (without KV cache quantization)
###############################################################################
class GPT2QuantAttention(nn.Module):
    def __init__(self, config: GPT2Config, w_bits, a_bits):
        super().__init__()
        self.embed_dim = config.n_embd      # hidden size
        self.num_heads = config.n_head       # number of attention heads
        self.head_dim = self.embed_dim // self.num_heads
        self.w_bits = w_bits
        self.a_bits = a_bits

        # Replace standard linear layers with quantized versions.
        self.q_proj = QuantizeLinear(self.embed_dim, self.embed_dim, w_bits=self.w_bits, a_bits=self.a_bits, bias=True)
        self.k_proj = QuantizeLinear(self.embed_dim, self.embed_dim, w_bits=self.w_bits, a_bits=self.a_bits, bias=True)
        self.v_proj = QuantizeLinear(self.embed_dim, self.embed_dim, w_bits=self.w_bits, a_bits=self.a_bits, bias=True)
        self.out_proj = QuantizeLinear(self.embed_dim, self.embed_dim, w_bits=self.w_bits, a_bits=self.a_bits, bias=True)

    def _shape(self, x, seq_len, bsz):
        # Reshape from [bsz, seq_len, hidden_dim] to [bsz, num_heads, seq_len, head_dim]
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
            # Convert attention mask to proper shape and type
            attention_mask = attention_mask.view(bsz, 1, 1, seq_len)
            attention_mask = attention_mask.expand(-1, self.num_heads, seq_len, -1)
            attention_mask = (1.0 - attention_mask) * -10000.0
            attn_weights = attn_weights + attention_mask
            
        attn_probs = nn.functional.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_probs, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, seq_len, self.embed_dim)
        attn_output = self.out_proj(attn_output)
        return attn_output

###############################################################################
# 2. Quantized MLP (Feed-Forward) Block
###############################################################################
class GPT2QuantMLP(nn.Module):
    def __init__(self, config: GPT2Config, w_bits, a_bits):
        super().__init__()
        self.w_bits = w_bits
        self.a_bits = a_bits
        inner_dim = config.n_inner if config.n_inner is not None else 4 * config.n_embd
        self.fc_in = QuantizeLinear(config.n_embd, inner_dim, w_bits=self.w_bits, a_bits=self.a_bits, bias=True)
        self.fc_out = QuantizeLinear(inner_dim, config.n_embd, w_bits=self.w_bits, a_bits=self.a_bits, bias=True)
        self.act = nn.GELU()

    def forward(self, hidden_states):
        hidden_states = self.fc_in(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.fc_out(hidden_states)
        return hidden_states

###############################################################################
# 3. Transformer Block with Layer-Specific Quantization Settings
###############################################################################
class GPT2QuantBlock(nn.Module):
    def __init__(self, config: GPT2Config, layer_bit_pair):
        super().__init__()
        # layer_bit_pair is a list [w_bits, a_bits] for this block.
        w_bits, a_bits = layer_bit_pair
        self.ln_1 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.attn = GPT2QuantAttention(config, w_bits=w_bits, a_bits=a_bits)
        self.ln_2 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.mlp = GPT2QuantMLP(config, w_bits=w_bits, a_bits=a_bits)

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
# 4. GPT2QuantModel: Stack Blocks and Embeddings
###############################################################################
class GPT2QuantModel(GPT2PreTrainedModel):
    def __init__(self, config: GPT2Config, layer_bit_config):
        """
        Args:
            config: A standard GPT2Config.
            layer_bit_config: A list of [w_bits, a_bits] pairs for each transformer block.
        """
        super().__init__(config)
        if len(layer_bit_config) != config.n_layer:
            raise ValueError("Length of layer_bit_config must equal n_layer.")
        self.layer_bit_config = layer_bit_config
        self.embed_dim = config.n_embd

        self.wte = nn.Embedding(config.vocab_size, self.embed_dim)
        self.wpe = nn.Embedding(config.n_positions, self.embed_dim)
        self.h = nn.ModuleList([
            GPT2QuantBlock(config, layer_bit_config[i]) for i in range(config.n_layer)
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
# 5. Testing the Quantized Model
###############################################################################
if __name__ == "__main__":
    import torch
    from transformers import GPT2Model
    
    print("Testing GPT2 Quantization Implementation")
    print("=======================================")
    
    # 1. Model Initialization
    print("\n1. Initializing Models...")
    base_config = GPT2Config()
    original_model = GPT2Model(base_config)
    
    # Create mixed precision config (different bits for different layers)
    num_layers = base_config.n_layer
    layer_bit_config = [
        [4, 4],  # First layer: 4-bit weights, 8-bit activations
        *[[4, 4] for _ in range(num_layers-2)],  # Middle layers: 8-bit
        [4, 4]   # Last layer: 4-bit weights, 8-bit activations
    ]
    
    quantized_model = GPT2QuantModel(base_config, layer_bit_config)
    
    # 2. Structure Comparison
    print("\n2. Comparing Model Structures...")
    print(f"Number of layers: Original={base_config.n_layer}, Quantized={len(quantized_model.h)}")
    print(f"Hidden size: Original={base_config.n_embd}, Quantized={quantized_model.embed_dim}")
    print(f"Number of attention heads: Original={base_config.n_head}, Quantized={quantized_model.h[0].attn.num_heads}")
    
    # 3. Test Forward Pass
    print("\n3. Testing Forward Pass...")
    input_ids = torch.randint(0, base_config.vocab_size, (2, 10))  # Batch size 2, sequence length 10
    attention_mask = torch.ones_like(input_ids)
    
    # Run both models
    with torch.no_grad():
        original_output = original_model(input_ids, attention_mask=attention_mask).last_hidden_state
        quantized_output = quantized_model(input_ids, attention_mask=attention_mask)
    
    print(f"Input shape: {input_ids.shape}")
    print(f"Original output shape: {original_output.shape}")
    print(f"Quantized output shape: {quantized_output.shape}")
    
    # 4. Check Quantization Effects
    print("\n4. Checking Quantization Effects...")
    
    def count_unique_values(tensor):
        return len(torch.unique(tensor))
    
    # Check first layer (4-bit quantization)
    first_layer_orig = original_model.h[0].attn.c_attn.weight
    first_layer_quant = quantized_model.h[0].attn.q_proj.weight
    
    print("\nFirst Layer (4-bit):")
    print(f"Original unique values: {count_unique_values(first_layer_orig)}")
    print(f"Quantized unique values: {count_unique_values(first_layer_quant)}")
    
    # Check middle layer (8-bit quantization)
    mid_layer_orig = original_model.h[num_layers//2].attn.c_attn.weight
    mid_layer_quant = quantized_model.h[num_layers//2].attn.q_proj.weight
    
    print("\nMiddle Layer (8-bit):")
    print(f"Original unique values: {count_unique_values(mid_layer_orig)}")
    print(f"Quantized unique values: {count_unique_values(mid_layer_quant)}")
    
    # 5. Output Statistics
    print("\n5. Comparing Output Statistics...")
    print("Original Output:")
    print(f"- Mean: {original_output.mean().item():.4f}")
    print(f"- Std: {original_output.std().item():.4f}")
    print(f"- Min: {original_output.min().item():.4f}")
    print(f"- Max: {original_output.max().item():.4f}")
    
    print("\nQuantized Output:")
    print(f"- Mean: {quantized_output.mean().item():.4f}")
    print(f"- Std: {quantized_output.std().item():.4f}")
    print(f"- Min: {quantized_output.min().item():.4f}")
    print(f"- Max: {quantized_output.max().item():.4f}")
    
    # 6. Memory Usage
    print("\n6. Comparing Model Sizes...")
    def get_model_size(model):
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        size_all_mb = (param_size + buffer_size) / 1024**2
        return size_all_mb
    
    orig_size = get_model_size(original_model)
    quant_size = get_model_size(quantized_model)
    
    print(f"Original model size: {orig_size:.2f} MB")
    print(f"Quantized model size: {quant_size:.2f} MB")
    print(f"Compression ratio: {orig_size/quant_size:.2f}x")
    
    print("\nAll tests completed successfully!")
