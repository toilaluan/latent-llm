from transformers import AutoTokenizer


def get_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
    return tokenizer
