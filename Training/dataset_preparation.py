import os
import numpy as np
import tqdm
from datasets import load_from_disk
from Datasets.tokenizer import BPETokenizer

DATAPATH = {
    "train": {"token_path": "train_tokens.bin", "dataset_path": "Datasets/train"},
    "validation": {
        "token_path": "val_tokens.bin",
        "dataset_path": "Datasets/validation",
    },
    "test": {"token_path": "test_tokens.bin", "dataset_path": "Datasets/test"},
}


def prepare_datasets(DATAPATH, tokenizer_dir="Datasets/tokenizer.json"):
    """
    Tokenizes dataset and saves them as binary uint16 files for fast memmap loading
    """

    if all(os.path.exists(value["token_path"]) for split, value in DATAPATH.items()):
        print("Tokenized binary files found. Skipping tokenization.")
        return

    dataset_files = [v["dataset_path"] for k, v in DATAPATH.items()]

    # Feteches tokenizer.json or trains tokenizer if not trained
    tokenizer = BPETokenizer(path=tokenizer_dir, files=dataset_files)

    if tokenizer.vocab_size > np.iinfo(np.uint16).max:
        raise ValueError("Tokenizer vocab exceeds uint16 capacity [65535].")

    for split, data in DATAPATH.items():
        dataset = load_from_disk(data["dataset_path"])

        ids = []
        batch = 50000
        for start in tqdm.tqdm(
            range(0, dataset.num_rows, batch), desc=f"Tokenizing {split}"
        ):
            batch_data = dataset[start : start + batch]
            texts = batch_data["text"]

            encodings = tokenizer.encode(texts)
            ids.extend(token_id for encoding in encodings for token_id in encoding.ids)

        ids_arr = np.array(ids, dtype=np.uint16)
        ids_arr.tofile(data["token_path"])  # Save Token Binary File

    print("Dataset preparation complete!")
