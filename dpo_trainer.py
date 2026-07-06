import torch
import torch.nn.functional as F
from torch.amp import autocast
import os

from Datasets.tokenizer import BPETokenizer
from model import Config, Model


def sequence_log_probs(model, input_ids, labels, device_type, ptdtype):
    """
    Computes log π(y|x)

    Args:
        input_ids : (B, L)
        labels    : (B, L) with -100 masked tokens

    Returns:
        (B,) sequence log probabilities
    """
    with autocast(device_type=device_type, dtype=ptdtype):
        # print(torch.min(input_ids), ":" ,torch.max(input_ids), ":", input_ids.shape)
        # return
        logits, _ = model(input_ids, Y=None)

    # Shift like normal LM training
    logits = logits[:, :-1]
    labels = labels[:, 1:]

    log_probs = F.log_softmax(logits, dim=-1)

    valid = labels != -100
    labels = labels.masked_fill(~valid, 0)

    token_log_probs = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)

    token_log_probs = token_log_probs * valid

    return token_log_probs.sum(dim=-1)

tokenizer = BPETokenizer(path="Datasets/tokenizer.json")

def dpo_loss(
    policy_model,
    reference_model_checkpoint_dir,
    reference_model_name,
    chosen_input_ids,
    chosen_labels,
    rejected_input_ids,
    rejected_labels,
    device_type,
    dtype,
    beta=0.1,
):
    """
    Computes Direct Preference Optimization loss.

    Returns:
        loss
        reward_chosen
        reward_rejected
    """

    policy_chosen = sequence_log_probs(
        policy_model,
        chosen_input_ids,
        chosen_labels,
        device_type,
        dtype,
    )

    policy_rejected = sequence_log_probs(
        policy_model,
        rejected_input_ids,
        rejected_labels,
        device_type,
        dtype,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reference_model = Model(Config(vocab_size=tokenizer.vocab_size, block_size=512))
    reference_model.to(device)
    reference_resume_path = os.path.join(reference_model_checkpoint_dir, f"{reference_model_name}_IFT.pt")
    
    reference_checkpoint = torch.load(
        reference_resume_path, map_location=device, weights_only=False
    )
    reference_model.load_state_dict(reference_checkpoint["model_state_dict"])

    reference_model.eval()

    for p in reference_model.parameters():
        p.requires_grad = False

    with torch.no_grad():
        ref_chosen = sequence_log_probs(
            reference_model,
            chosen_input_ids,
            chosen_labels,
            device_type,
            dtype,
        )

        ref_rejected = sequence_log_probs(
            reference_model,
            rejected_input_ids,
            rejected_labels,
            device_type,
            dtype,
        )

    # print("BEFORE REFERENCE FORWARD ",torch.cuda.memory_allocated()/1024**2,"\n")
    reference_model.cpu()      # move weights off the GPU
    del reference_model
    torch.cuda.empty_cache()
    # print("AFTER REFERENCE FORWARD ",torch.cuda.memory_allocated()/1024**2,"\n")    
    
    policy_gap = policy_chosen - policy_rejected
    reference_gap = ref_chosen - ref_rejected

    logits = beta * (policy_gap - reference_gap)

    loss = -F.logsigmoid(logits).mean()

    reward_chosen = beta * (policy_chosen - ref_chosen)
    reward_rejected = beta * (policy_rejected - ref_rejected)

    return (
        loss,
        reward_chosen.mean(),
        reward_rejected.mean(),
    )
