import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings


def scaled_dot_product(q, k, v, mask=None):
    d_k = q.size()[-1]
    attn_logits = torch.matmul(q, k.transpose(-2, -1))
    attn_logits = attn_logits / math.sqrt(d_k)
    if mask is not None:
        attn_logits = attn_logits.masked_fill(mask == 0, -9e15)
    attention = F.softmax(attn_logits, dim=-1)
    values = torch.matmul(attention, v)
    return values, attention


class MultiheadAttentionAggregation(nn.Module):
    def __init__(
        self, input_dim, output_dim, embed_dim, num_heads, num_output_tokens, use_flash_attn=False
    ):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be 0 modulo number of heads."

        self.use_flash_attn = use_flash_attn
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.num_output_tokens = num_output_tokens

        self.kv_proj = nn.Linear(input_dim, 2 * embed_dim)
        self.o_proj = nn.Linear(embed_dim, output_dim)

        # This is the big difference to default attention,
        # the Q matric is not the mapping of the queries but is a learnable matrix itself.
        # This lets us pool over the output tokens.
        # Setting
        self.q = nn.Parameter(
            torch.zeros(1, num_heads, num_output_tokens, self.head_dim), requires_grad=True
        )

        self._reset_parameters()

    def _reset_parameters(self):
        # Original Transformer initialization, see PyTorch documentation
        nn.init.xavier_uniform_(self.kv_proj.weight)
        self.kv_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.q)

    def forward(self, x, mask=None, return_attention=False):
        if self.use_flash_attn and return_attention:
            warnings.warn(
                "return_attention=True will disable flash attention because current interface of flash attention does not return attention matrix."
            )

        batch_size, seq_length, *_ = x.size()
        kv = self.kv_proj(x)

        # Separate K, V from linear output
        kv = kv.reshape(batch_size, seq_length, self.num_heads, 2 * self.head_dim)
        kv = kv.permute(0, 2, 1, 3)  # [B, num_heads, L, D]
        k, v = kv.chunk(2, dim=-1)

        # Determine value outputs
        if self.use_flash_attn and not return_attention:
            values = torch.nn.functional.scaled_dot_product_attention(self.q, k, v, attn_mask=mask)
        else:
            values, attention = scaled_dot_product(self.q, k, v, mask=mask)
        values = values.permute(0, 2, 1, 3)  # [B, L, num_heads, D]
        values = values.reshape(batch_size, self.num_output_tokens, self.embed_dim)
        o = self.o_proj(values)

        if return_attention:
            return o, attention
        else:
            return o


class EncoderAggregationBlock(nn.Module):
    def __init__(
        self,
        input_dim,
        embed_dim,
        num_heads,
        num_output_tokens: int = 1,
        add_mlp: bool = False,
        dropout: float = 0.0,
        use_flash_attn=False,
    ):
        super().__init__()

        self.add_mlp = add_mlp
        # Attention layer
        self.self_attn_agg = MultiheadAttentionAggregation(
            input_dim,
            input_dim,
            embed_dim,
            num_heads,
            num_output_tokens,
        )

        if self.add_mlp:
            # Two-layer MLP
            self.linear_net = nn.Sequential(
                nn.Linear(input_dim, 2 * embed_dim),
                nn.Dropout(dropout),
                # probably we will use something else??
                nn.ReLU(inplace=True),
                nn.Linear(2 * embed_dim, input_dim),
            )
            # Layers to apply in between the main layers
            self.norm1 = nn.LayerNorm(input_dim)
            self.norm2 = nn.LayerNorm(input_dim)
            self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Attention part
        x = self.self_attn_agg(x, mask=mask)

        if self.add_mlp:
            x = self.norm1(x)
            # MLP part
            linear_out = self.linear_net(x)
            x = x + self.dropout(linear_out)
            x = self.norm2(x)

        return x
