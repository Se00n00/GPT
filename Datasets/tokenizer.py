from datasets import load_from_disk
from tokenizers import Tokenizer, normalizers, pre_tokenizers
from tokenizers.models import BPE
from tokenizers.normalizers import NFD, StripAccents
from tokenizers.pre_tokenizers import Digits, Whitespace
from tokenizers.trainers import BpeTrainer
from tokenizers.processors import TemplateProcessing

class BPETokenizer:
    def __init__(self, path: str, files: None|list[str] = None, vocab_size: int=32768):
        self.path = path
        self.special_tokens_list = ["[PAD]", "[UNK]", "<|START|>", "<|END|>", "<|SYSTEM|>", "<|USER|>", "<|ASSISTANT|>"]

        try:
            self.tokenizer = Tokenizer.from_file(path)
            # Add new special tokens if they don't exist in the loaded tokenizer
            tokens_to_add = [token for token in self.special_tokens_list if token not in self.tokenizer.get_vocab()]
            if tokens_to_add:
                print(f"Adding new special tokens to existing tokenizer: {tokens_to_add}")
                self.tokenizer.add_special_tokens(tokens_to_add)
                # Save the tokenizer after adding new tokens
                self.tokenizer.save(path)
        except Exception as e:
            print(f"Tokenizer file '{path}' not found or error loading: {e}. Training new tokenizer...")
            if not files:
                raise FileNotFoundError(
                    f"Tokenizer file '{path}' not found and no training files provided."
                )
            self._train(files, vocab_size)
            self.tokenizer.save(path) # Save after training

        self._set_special_token_ids()
        self._setup_post_processor() # Call post-processor setup here for loaded and new tokenizers

        print("\n--------------------------------------------------------------------------")
        print(f"VOCAB SIZE: {self.vocab_size} | SPECIAL TOKENS: {' '.join(self.special_tokens_list)}")
        print("--------------------------------------------------------------------------\n")

    def _set_special_token_ids(self):
        self.vocab_size = self.tokenizer.get_vocab_size()
        self.pad_id = self.tokenizer.token_to_id("[PAD]")
        self.unk_id = self.tokenizer.token_to_id("[UNK]")
        self.bos_id = self.tokenizer.token_to_id("<|START|>")
        self.eos_id = self.tokenizer.token_to_id("<|END|>")
        self.system_id = self.tokenizer.token_to_id("<|SYSTEM|>" if "<|SYSTEM|>" in self.tokenizer.get_vocab() else None)
        self.user_id = self.tokenizer.token_to_id("<|USER|>" if "<|USER|>" in self.tokenizer.get_vocab() else None)
        self.assistant_id = self.tokenizer.token_to_id("<|ASSISTANT|>" if "<|ASSISTANT|>" in self.tokenizer.get_vocab() else None)

    def _setup_post_processor(self):
        special_tokens_for_template = []
        for token in self.special_tokens_list:
            token_id = self.tokenizer.token_to_id(token)
            if token_id is not None:
                special_tokens_for_template.append((token, token_id))

        self.tokenizer.post_processor = TemplateProcessing(
            single="<|START|> $A <|END|>",
            special_tokens=special_tokens_for_template,
        )

    def _train(self, files: None | list[str], vocab_size: int):
        self.tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

        # Normalization
        self.tokenizer.normalizer = normalizers.Sequence([NFD(), StripAccents()])

        # Pre-tokenization
        self.tokenizer.pre_tokenizer = pre_tokenizers.Sequence([Whitespace()])
        trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=self.special_tokens_list) # Use the full list

        def batch_iterator():
            for path in files:
                ds_loaded = load_from_disk(path)
                for batch in ds_loaded.iter(batch_size=1000):
                    for message_list in batch["messages"]:
                        for message_dict in message_list:
                            yield message_dict["content"]

        self.tokenizer.train_from_iterator(batch_iterator(), trainer)

    def encode(self, input: list[str] | str):
        if isinstance(input, str):
            return self.tokenizer.encode(input)
        else:
            return self.tokenizer.encode_batch(input)

    def decode(self, input: list[int]):
        return self.tokenizer.decode(input, skip_special_tokens=False)