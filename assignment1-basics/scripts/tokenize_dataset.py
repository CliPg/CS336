import pickle
import os
import pathlib
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from cs336_basics.BPETokenizer import Tokenizer
import numpy as np
from tqdm import tqdm

DATA_DIR = pathlib.Path(__file__).resolve().parent.parent / "data"
TINY_STORIES_TRAIN = "TinyStoriesV2-GPT4-train.txt"
TINY_STORIES_VALID = "TinyStoriesV2-GPT4-valid.txt"

TRAIN_DATA_PATH = os.path.join(DATA_DIR, TINY_STORIES_TRAIN)
VALID_DATA_PATH = os.path.join(DATA_DIR, TINY_STORIES_VALID)

TRAIN_DATA_SAVE_PATH = os.path.join(DATA_DIR, "tiny_stories_train_tokens.dat")
VALID_DATA_SAVE_PATH = os.path.join(DATA_DIR, "tiny_stories_valid_tokens.dat")

VOCAB_PATH = os.path.join(DATA_DIR, "vocab.pkl")
MERGES_PATH = os.path.join(DATA_DIR, "merges.pkl")

special_tokens = ["<|endoftext|>"]


# 构造tokenizer

tokenizer = Tokenizer.from_files(vocab_filepath=VOCAB_PATH, merges_filepath=MERGES_PATH, special_tokens=special_tokens)

print("=== 测试 Tokenizer ===")
test_texts = [
    "Once upon a time, there was a little robot.",
    "Hello world! <|endoftext|> Some more text.",
    "<|endoftext|>",
    "你好，世界！"
]

for text in test_texts:
    print(f"\n原文: {text}")
    encoded = tokenizer.encode(text)
    print("编码:", encoded)

    byte_tokens = [tokenizer.vocab[token_id] for token_id in encoded]
    str_tokens = [b.decode("utf-8", errors="replace") for b in byte_tokens]
    print("分词（可读）:", str_tokens)

    decoded = tokenizer.decode(encoded)
    print("解码:", decoded)
    print("是否完全还原:", decoded == text)



def encode_txt_as_numpy_array(tokenizer, path_to_txt, save_path):
    with open(path_to_txt, 'r', encoding='utf-8') as f:
        num_lines = sum(1 for _ in f)
    
    # 第一步：统计总token数（需要遍历一遍）
    total_tokens = 0
    with open(path_to_txt, 'r', encoding='utf-8') as f:
        for line in tqdm(f, total=num_lines, desc="Counting tokens"):
            total_tokens += len(tokenizer.encode(line))

    # 第二步：创建memmap
    dtype = np.int32
    tokens_mm = np.memmap(save_path, dtype=dtype, mode='w+', shape=(total_tokens,))

    # 第三步：再次遍历写入
    pos = 0
    with open(path_to_txt, 'r', encoding='utf-8') as f:
        for line in tqdm(f, total=num_lines, desc="Tokenizing"):
            ids = tokenizer.encode(line)
            n = len(ids)
            tokens_mm[pos:pos+n] = ids
            pos += n

    tokens_mm.flush()

# python tokenize_dataset.py
def main():
    encode_txt_as_numpy_array(tokenizer, TRAIN_DATA_PATH, TRAIN_DATA_SAVE_PATH)
    encode_txt_as_numpy_array(tokenizer, VALID_DATA_PATH, VALID_DATA_SAVE_PATH)


if __name__ == "__main__":
    main()