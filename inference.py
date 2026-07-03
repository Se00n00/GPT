import os
import sys
import argparse
import torch
import torch.nn.functional as F
import time

# from tokenizer import BPETokenizer
# from models import GPT2LMHeadModel, AdvancedLMHeadModel

SYSTEM_PROMPT = """
You are a helpful, conversational AI companion. Keep your tone natural, engaging, and slightly witty. 

* **Be Concise:** Answer directly and avoid unnecessary fluff or repetitive pleasantries.
* **Adapt:** Match the user's energy, style, and technical level. 
* **Structure:** Use clean markdown (bolding, lists) to make responses easy to scan.
* **Stay Grounded:** Never make up facts. If you don't know something, just say so.
"""

@torch.no_grad()
def generate(model, tokenizer, user_prompt, max_new_tokens, temperature=1.0, top_k=50, top_p=0.9, device="cpu"):
    """
    Generates text from a prompt using KV caching and sampling.
    """
    system_prompt = SYSTEM_PROMPT
    model.eval()
    prompt = "<|SYSTEM|>"+system_prompt + "<|USER|>"+ user_prompt + "<|ASSISTANT|>"
    # 1. Encode prompt
    tokenized = tokenizer.encode(prompt)

    input_ids =tokenized.ids
    if input_ids[-1] == tokenizer.eos_id:
        input_ids.pop()

    if not input_ids:
        # If prompt was empty or tokenized to nothing, use endoftext
        input_ids = [tokenizer.bos_id]

    print(tokenizer.decode(input_ids), end="\n\n", flush=True)

    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device) # [1, T]

    generated = list(input_ids)
    past_key_values = None
    curr_input = input_tensor

    end_of_text_id = tokenizer.eos_id

    num_new_tokens = 0
    start_time = time.perf_counter()

    for i in range(max_new_tokens):
        num_new_tokens += 1

        logits, _ = model(curr_input, None)

        # Take only last position
        logits = logits[:, -1, :]

        if temperature == 0.0:
            next_token = torch.argmax(logits, dim=-1, keepdim=True)

        else:
            logits = logits / temperature

            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")

            if top_p is not None and top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(
                    logits,
                    descending=True,
                    dim=-1
                )

                cumulative_probs = torch.cumsum(
                    F.softmax(sorted_logits, dim=-1),
                    dim=-1
                )

                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = \
                    sorted_indices_to_remove[..., :-1].clone()

                sorted_indices_to_remove[..., 0] = False

                indices_to_remove = sorted_indices_to_remove.scatter(
                    -1,
                    sorted_indices,
                    sorted_indices_to_remove
                )

                logits[indices_to_remove] = -float("Inf")

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1)

        next_token_id = next_token.item()

        generated.append(next_token_id)

        if next_token_id == tokenizer.eos_id:
            break

        curr_input = torch.tensor(
            [generated],
            dtype=torch.long,
            device=device
        )
        if len(generated) > 512:
            break
        print(tokenizer.decode([next_token_id]), end=" ", flush=True)

    end_time = time.perf_counter()
    print("\n\n TOKENS /SEC : ", num_new_tokens / (end_time - start_time))
    return tokenizer.decode(generated)

from Datasets.tokenizer import BPETokenizer
from model import Config, Model

def main():
    parser = argparse.ArgumentParser(description="Generate text using trained 10M GPT-2 or Advanced model.")
    parser.add_argument("--model", type=str, choices=["GPT"], default="GPT",
                        help="Model architecture: GPT")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to model checkpoint (.pt). If not specified, looks in checkpoints/ directory.")
    parser.add_argument("--prompt", type=str, default="The wikitext dataset contains articles about",
                        help="Prompt to generate text from")
    parser.add_argument("--max_new_tokens", type=int, default=100, help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="Temperature (0.0 for greedy)")
    parser.add_argument("--top_k", type=int, default=40, help="Top-k filtering (0 to disable)")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p/nucleus sampling (1.0 to disable)")
    
    args = parser.parse_args()

    TOKENIZER_DIR = "Datasets/sft_tokenizer.json"
    # Load tokenizer
    if not os.path.exists(TOKENIZER_DIR):
        print(f"Error: Tokenizer not found in {TOKENIZER_DIR}. Please run train.py first to build the vocabulary!")
        sys.exit(1)
        
    tokenizer = BPETokenizer(path=TOKENIZER_DIR)
    vocab_size = tokenizer.vocab_size
    print(f"Loaded BPE tokenizer. Vocab size = {vocab_size}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Set checkpoint path if not provided
    checkpoint_file = args.checkpoint
    if checkpoint_file is None:
        checkpoint_file = f"checkpoints/{args.model}_it.pt"

    # Initialize model
    match args.model:
        case 'GPT':
            model = Model(Config(vocab_size=tokenizer.vocab_size, block_size=512))
        case _:
            model = Model(Config(vocab_size=tokenizer.vocab_size, block_size=512))

    # Load weights
    if os.path.exists(checkpoint_file):
        print(f"Loading checkpoint weights from {checkpoint_file}...")
        checkpoint = torch.load(checkpoint_file, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        print("Loaded successfully")
    else:
        print(f"WARNING: Checkpoint '{checkpoint_file}' not found. Generating with an UNTRAINED model (random weights)!")

    model.to(device)

    # Generate text
    # print(f"\nGenerating {args.max_new_tokens} tokens with prompt: \"{args.prompt}\"")
    print("-" * 60)
    output_text = generate(
        model=model,
        tokenizer=tokenizer,
        user_prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        device=device
    )
    # print(output_text)
    print("\n")
    print("-" * 60)

if __name__ == "__main__":
    main()
