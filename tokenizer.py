from tokenizers import Tokenizer

from tokenizers import normalizers
from tokenizers import pre_tokenizers
from tokenizers.models import BPE
from tokenizers.normalizers import NFD, StripAccents
from tokenizers.pre_tokenizers import Whitespace, Digits
from tokenizers.trainers import BpeTrainer

from datasets import load_from_disk

class BPETokenizer:
    def __init__(self, path: string, vocab_size: None|int, files: None|list[str]):
        
        try:
            self.tokenizer = Tokenizer.from_file(path)
        except:
            self._train(vocab_size, files)
            self.tokenizer.save(path)
            self.tokenizer.get_vocab_size()
            print(self.tokenizer.get_vocab_size())

    def _train(self, vocab_size: None|int, files:list[str]):
        
        self.tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

        # Normallization
        self.tokenizer.normalizer = normalizers.Sequence([NFD(), StripAccents()])

        # Pre-tokenization
        self.tokenizer.pre_tokenizer = pre_tokenizers.Sequence([Whitespace(), Digits(individual_digits =True)])
        
        special_tokens = ["[UNK]","<|START|>","<|END|>"]
        if vocab_size:
            trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=special_tokens)
        trainer = BpeTrainer(special_tokens=special_tokens)

        def batch_iterator():
            for path in files:
                ds = load_from_disk(path)
                for example in ds:
                    yield example['text']

        self.tokenizer.train_from_iterator(batch_iterator(), trainer)

    def encode(self, input:list[str] | str):
        if isinstance(input, str):
            return self.tokenizer.encode(input)
        else:
            return self.tokenizer.encode_batch(input)

    def decode(self, input:list[int]):
        return self.tokenizer.decode(input)

