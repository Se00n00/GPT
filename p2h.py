import torch

def convert(tokenizer_dir, model_path):

    # Save Tokenizer
    tokenizer = BPETokenizer(path = tokenizer_dir)
    tokenizer.save(model_path)


if __name__ == '__main__':
    convert(
        tokenizer_dir = '',
        model_path = ''
    )
