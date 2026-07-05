from torch.utils.data import Dataset
import torch

class TextDataset(Dataset):
    def __init__(
        self,
        tokenizer,
        token_ids,
        block_size=1024
    ):
        self.token_ids = token_ids
        self.block_size = block_size

    def __len__(self):
        return len(self.token_ids) - self.block_size

    def __getitem__(self, idx):
        x = self.token_ids[idx:idx+self.block_size]
        y = self.token_ids[idx+1:idx+self.block_size+1]

        return (
            torch.tensor(x),
            torch.tensor(y)
        )

# tokens = np.memmap(
#     "train.bin",
#     dtype=np.uint16,
#     mode="r"
# )

# dataset = TextDataset(
#     token_ids=tokens,
#     block_size=512
# )