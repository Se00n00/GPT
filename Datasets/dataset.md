## Download & Save independent set of "wikitext-103-v1" datasets

```bash
pip install datasets
````

```python

from datasets import load_dataset

ds = load_dataset("Salesforce/wikitext", "wikitext-103-v1")
ds.save_to_disk("Dataset/train", ds['train'])
ds.save_to_disk("Dataset/test", ds['test'])
ds.save_to_disk("Dataset/validation", ds["validation"])

````

## Training and Saving Tokenizer

```python
from GPT.tokenizer import BPETokenizer

# Saves the tokenizer
tokenizer = BPETokenizer(path = "Datasets/tokenizer.json", vocab_size = None, files = ["Datasets/train","\
Datasets/test","Datasets/validation"])
````
