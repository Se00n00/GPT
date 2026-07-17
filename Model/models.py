import torch
import torch.nn as nn
import torch.nn.functional as F

from Model.layers import Block, Config, EmbeddingLayer


class Model(nn.Module):
    def __init__(self, config: Config):
        super().__init__()

        self.embedding = EmbeddingLayer(
            config.vocab_size, config.block_size, config.d_model
        )
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.num_layer)])
        self.head_proj = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, X: torch.Tensor, Y: torch.Tensor, pad_index: int = 0):
        X = self.embedding(X)

        for block in self.blocks:
            X = block(X)

        output = self.head_proj(X)

        if Y is not None:
            loss = F.cross_entropy(
                output.view(-1, output.size(-1)), Y.view(-1), ignore_index=pad_index
            )
            return output, loss
        else:
            # logits = self.output(h[:, -1:, :])
            # return logits, pkv_next

            return output, output[:, [-1], :]  # [B, L, vocab_size], [B, 1, vocab_size]
