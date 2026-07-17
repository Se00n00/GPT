# Layers:
#   EmbeddingLayer
#   Normallization Layers: LayerNorm, RMSNorm
#   Attention: MHA
#   FeedForward
# Block

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


class EmbeddingLayer(nn.Module):
    def __init__(self, vocab_size: int, block_size: int, d_model: int):
        super().__init__()
        self.token_encoding = nn.Embedding(vocab_size, d_model)
        self.position_encoding = nn.Embedding(block_size, d_model)

    def forward(self, X: torch.Tensor):
        seq_len = X.size(1)
        pos = torch.arange(0, seq_len, dtype=torch.long, device=X.device)
        # [B, L, d_model] <-- [B, L, d_model] + [L, d_model]
        return self.token_encoding(X) + self.position_encoding(pos)


class LayerNorm(nn.Module):
    def __init__(self, d_model: int, device="cpu"):
        super().__init__()

        self.weights = nn.Parameter(torch.ones(d_model))  # requires_grad = True
        self.bias = nn.Parameter(torch.ones(d_model))

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        normallized = (X - X.mean()) / X.std()
        return normallized * self.weights + self.bias


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, epsilon=1e-8):
        super().__init__()

        self.epsilon = epsilon
        self.weights = nn.Parameter(torch.ones(d_model))

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        normallized = X * torch.rsqrt(X.pow(2).mean(-1, keepdim=True) + self.epsilon)

        return normallized.type_as(X) * self.weights


class Attention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, max_len: int, dropout_prob: float):
        super().__init__()

        assert d_model % num_heads == 0, (
            "Number of Heads must be divisible to model dim"
        )
        head_dim = d_model // num_heads
        self.num_heads, self.head_dim = num_heads, head_dim

        self.q_proj = nn.Linear(d_model, head_dim * num_heads, bias=False)
        self.k_proj = nn.Linear(d_model, head_dim * num_heads, bias=False)
        self.v_proj = nn.Linear(d_model, head_dim * num_heads, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

        self.attention_dropout = nn.Dropout(dropout_prob)
        self.output_dropout = nn.Dropout(dropout_prob)

        self.register_buffer(
            "attention_mask",
            torch.triu(torch.full((1, 1, max_len, max_len), float("-inf")), diagonal=1),
        )

    def forward(self, x: torch.Tensor):
        batch, seqlen, d_model = x.shape

        # [B, L, head_dim * num_heads] <-- [B, L, D_model]
        Q, K, V = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        # Reshape <-- View | [B, num_heads, L, head_dim] <-- [B, L, num_heads, head_dim] <-- [B, L, num_heads * head_dim]
        Q = Q.view(batch, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch, seqlen, self.num_heads, self.head_dim).transpose(1, 2)

        # output = F.scaled_dot_product_attention(
        #    Q,
        #    K,
        #    V,
        #    attn_mask=None,
        #    dropout_p=self.attention_dropout.p if self.training else 0.0,
        #    is_causal=True,
        # )
        scores = torch.matmul(Q, K.transpose(2, 3)) / math.sqrt(self.head_dim)
        scores = scores + self.attention_mask[:, :, :seqlen, :seqlen]
        scores = F.softmax(scores, dim=-1)
        scores = self.attention_dropout(scores)
        output = torch.matmul(scores, V)

        # [B, L, D_model] <-- [B, L, num_heads, head_dim] <-- [B, num_heads, L, head_dim]
        output = output.transpose(1, 2).contiguous().view(batch, seqlen, -1)
        return self.output_dropout(self.o_proj(output))


class FeedForward(nn.Module):
    def __init__(
        self,
        d_model: int,
        hidden_d: int | None = None,
        gated: bool = True,
        dropout_prob=0.5,
    ):
        super().__init__()

        if not hidden_d:
            hidden_d = 4 * d_model

        self.is_gated = gated

        self.up_proj = nn.Linear(d_model, hidden_d, bias=False)
        if gated:
            self.gate_proj = nn.Linear(d_model, hidden_d, bias=False)
        self.down_proj = nn.Linear(hidden_d, d_model, bias=False)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if self.is_gated:
            temp = F.silu(self.up_proj(X)) * self.gate_proj(X)
        else:
            temp = F.silu(self.up_proj(X))

        return self.dropout(self.down_proj(temp))


class MoE(nn.Module):
    def __init__(
        self,
        d_model: int,
        top_k: int = 2,
        num_of_experts: int = 8,
        hidden_d: int | None = None,
        gated: bool = True,
        dropout_prob=0.5,
    ) -> None:
        super().__init__()
        assert top_k <=num_of_experts, "Top K experts must be less or equal to number of experts"
        self.top_k = top_k
        
        self.experts = nn.ModuleList(
            [
                FeedForward(d_model, hidden_d, gated, dropout_prob)
                for _ in range(num_of_experts)
            ]
        )
        self.W_expert_gate = nn.Linear(d_model, num_of_experts)
    
    def forward(self, X:torch.Tensor) -> torch.Tensor:
        router = self.W_expert_gate(X) # [B, L, EXPERTS]
        top_k = torch.topk(router, self.top_k, dim=-1) # values, indices: [B, L, TOP_K]
        
        router_logits = F.softmax(top_k.values, dim=-1)
    
    def _get_top_k(self, router:torch.Tensor) -> tuple[torch.Tensor, list]:
        
        return torch.tensor(), []


@dataclass
class Config:
    num_layer: int = 10
    max_len: int = 1024
    vocab_size: int = 30000
    block_size: int = 1024

    d_model: int = 512
    num_heads: int = 8
    dropout_prob: float = 0.1

    # Feedforward
    ff_hidden_d: int | None = 2048
    ff_gated: bool = True

    # RMS Norm
    norm_epsilon: float = 1e-8


class Block(nn.Module):
    def __init__(self, config: Config):
        super().__init__()

        self.norm1 = RMSNorm(config.d_model, config.norm_epsilon)
        self.norm2 = RMSNorm(config.d_model, config.norm_epsilon)
        self.attention = Attention(
            config.d_model, config.num_heads, config.max_len, config.dropout_prob
        )
        self.feedforward = FeedForward(
            config.d_model, config.ff_hidden_d, config.ff_gated, config.dropout_prob
        )

    def forward(self, X: torch.Tensor):
        X = X + self.attention(self.norm1(X))
        X = X + self.feedforward(self.norm2(X))

        return X
