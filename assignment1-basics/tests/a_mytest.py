import regex as re

def split_by_special_tokens(text: str, special_tokens: list[str]) -> list[str]:
    special_tokens_sorted = sorted(special_tokens, key=lambda x: -len(x))
    if not special_tokens_sorted:
        parts = [text]
    else:
        pattern = "|".join(re.escape(token) for token in special_tokens_sorted)
        parts = re.split('(' + pattern + ')', text)

    return parts

def pretokenize(text: str, special_tokens: list[str], drop_special_token: bool = True) -> list[bytes]:
    parts = split_by_special_tokens(text, special_tokens)

    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    tokens_list = []
    for part in parts:
        if part in special_tokens:
            if not drop_special_token:
                special_tokens_bytes = part.encode('utf-8')
                tokens_list.append([special_tokens_bytes])
        else:
            str_tokens = re.findall(PAT, part)
            part_tokens = [s.encode('utf-8') for s in str_tokens]
            tokens_list.append(part_tokens)
    tokens = [token for part_tokens in tokens_list for token in part_tokens]
    return tokens

    


text = "Hello <|uk|> world! <|endoftext|> Great!"
text2 = "Hello world"
special_tokens = ["<|endoftext|>", "<|uk|>"]
special_tokens2 = []

# print(split_by_special_tokens(text=text, special_tokens=special_tokens))
print(pretokenize(text, special_tokens))


