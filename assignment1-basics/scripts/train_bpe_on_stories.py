from ..cs336_basics.BPETokenizer import train_bpe
import pickle

def train_bpe_tinystories():
    input_path = '..\data\TinyStoriesV2-GPT4-train.txt'
    vocab_size = 10000
    special_tokens = ['<|endoftext|>']
    
    vocab, merges = train_bpe(input_path, vocab_size, special_tokens)
    
    with open(r'..\data\vocab.pkl', 'wb') as f:
        pickle.dump(vocab, f)

    with open(r'..\data\merges.pkl', 'wb') as f:
        pickle.dump(merges, f)

train_bpe_tinystories()