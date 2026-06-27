import os

import torch
import torch.nn as nn

from Datasets.tokenizer import BPETokenizer
from model import Config, Model


def expand_vocab_size(old_model):

    # 1 Create a new model
    tokenizer = BPETokenizer(path="Datasets/sft_tokenizer.json")
    new_model = Model(Config(vocab_size=tokenizer.vocab_size, block_size=512))

    # 2. Copy old weights  : everything except embedding/head
    state = old_model.state_dict()
    filtered = {
        k: v
        for k, v in state.items()
        if not (k.startswith("embedding.token_encoding") or k.startswith("head_proj"))
    }
    new_model.load_state_dict(filtered, strict=False)

    # 3. Expand embeddings
    old_vocab = old_model.embedding.token_encoding.num_embeddings
    new_vocab = new_model.embedding.token_encoding.num_embeddings

    with torch.no_grad():
        # copy old rows
        new_model.embedding.token_encoding.weight[:old_vocab] = (
            old_model.embedding.token_encoding.weight
        )

        # initialize new rows
        nn.init.normal_(
            new_model.embedding.token_encoding.weight[old_vocab:], mean=0.0, std=0.02
        )

    # 4. Expand output head
    with torch.no_grad():
        new_model.head_proj.weight[:old_vocab] = old_model.head_proj.weight

        nn.init.normal_(new_model.head_proj.weight[old_vocab:], mean=0.0, std=0.02)

    return new_model
    


if __name__ == "__main__":
    old_tokenizer = BPETokenizer(path="Datasets/tokenizer.json")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_file = "checkpoints/best_GPT.pt"
    checkpoint = torch.load(checkpoint_file, map_location=device, weights_only=False)

    old_model = Model(Config(vocab_size=old_tokenizer.vocab_size, block_size=512))
    old_model.load_state_dict(checkpoint["model_state_dict"])

    # EXPANSION
    new_model = expand_vocab_size(old_model)

    # SAVE NEW MODEL
    checkpoint_path = os.path.join("checkpoints", "expanded_GPT.pt")
    torch.save(
        {"model_state_dict": new_model.state_dict()},
        checkpoint_path,
    )