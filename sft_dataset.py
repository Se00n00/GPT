import numpy as np

from Datasets.tokenizer import BPETokenizer

tokens = np.memmap("Datasets/Tokens/sft_train_tokens.bin", dtype=np.uint16, mode="r")
labels = np.full(len(tokens), -100, dtype=np.int16)

tokenizer = BPETokenizer(path="Datasets/sft_tokenizer.json")

start_id = tokenizer.bos_id
assistant_id = tokenizer.assistant_id
user_id = tokenizer.user_id
system_id = tokenizer.system_id
end_id = tokenizer.eos_id


state = "IGNORE"

from tqdm import tqdm
BATCH_SIZE = 500000
with open("Datasets/Tokens/sft_train_labels.bin", "wb") as f:
    for start in tqdm(
        range(0, len(tokens), BATCH_SIZE),
        desc="Generating labels",
        unit="batch",
    ):
        end = min(start + BATCH_SIZE, len(tokens))

        batch_tokens = tokens[start:end]
        batch_labels = np.full(len(batch_tokens), 60000, dtype=np.uint16)

        for j, token in enumerate(batch_tokens):
            if token == assistant_id:
                state = "LEARN"
                continue

            if token in (user_id, system_id, start_id):
                state = "IGNORE"
                continue

            if token == end_id:
                if state == "LEARN":
                    batch_labels[j] = token
                state = "IGNORE"
                continue

            if state == "LEARN":
                batch_labels[j] = token

        #print(batch_labels[:10],"\n")
        batch_labels.tofile(f)
