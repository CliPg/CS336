# CS336 a1

## Byte-Pair Encoding(BPE) Tokenizer

### Unicode

unicode是基于字节的分词方法，能将字符转换为若干字节整数(0-255)

```python
>>> test_string = '牛'
>>> print(test_string.encode('utf-8'))
b'\xe7\x89\x9b'
>>> list(test_string.encode('utf-8'))
[231, 137, 155]
>>>print(test_string.encode('utf-8').decode('utf-8'))
牛
```



### BPE

词级分词器容易遇到 OOV （out of vacabulary)问题；字节级分词器没有 OOV 问题但输入序列很长，计算开销大。

因此使用子词（subword）分词作为折中，既能减少 OOV，也能压缩输入长度。

**关键算法**：Byte-Pair Encoding（BPE）——一种基于频率合并字节对的压缩算法，用于构建子词词表。

例如，如果字节序列 `b'the'` 在训练语料中经常出现，那么将其加入词汇表可以将原本由三个 token 表示的序列压缩为一个 token。

**优点**：

- 通过合并高频字节序列，实现更短的输入；
- 避免 OOV 问题；
- 保持输入长度在可控范围内。



#### 步骤

#####  词汇表初始化（Vocabulary Initialization）

分词器的词汇表是一个从字节串（bytestring token）到整数 ID 的一一映射。由于我们正在训练的是一个基于字节的 BPE 分词器，初始的词汇表就是所有可能的字节的集合。因为字节的取值范围是 0 到 255，总共有 256 个可能的字节值，因此初始词汇表大小就是 256。

------

#####  预分词（Pre-tokenization）

一旦有了初始词汇表，原则上我们可以开始统计文本中哪些字节对最常出现在一起，然后从最频繁的对开始合并。但这种方式非常耗费计算资源：每合并一次，都必须对整个语料库重新扫描一遍。

此外，直接在语料上逐字节合并还有个问题：某些 token 可能只是因为标点不同而被视为完全不同的 token，比如 `dog!` 和 `dog.`，它们虽然语义非常相近，但会被分配成完全不同的 token ID。

为避免这个问题，我们会先对语料进行 **预分词（pre-tokenization）**。你可以把它看作是一种粗粒度的初步分词，有助于我们更高效地统计字节对的出现频率。例如，如果词语 `'text'` 在语料中出现了 10 次，我们可以将 `'t'` 和 `'e'` 紧邻出现的频次直接加 10，而不需要在语料中每次都去查找。

由于我们是训练一个基于字节的 BPE 模型，每个预分词 token 最终会被表示成一个 UTF-8 字节序列。

---

##### BPE 合并操作（BPE merges）

从宏观角度看，BPE 算法的基本过程如下：

1. **统计**：遍历所有 pre-token，统计所有相邻字节对的频率。
2. **选择**：找到出现频率最高的一对字节（例如 "A" 和 "B"）。
3. **合并**：将所有出现这个字节对 ("A", "B") 的地方替换成一个新的 token："AB"。
4. **更新词表**：把新 token 加入词汇表中。

这样不断迭代下去，最终得到的词汇表就包括：

- 初始的 256 个单字节（byte）token；
- 加上每次合并产生的新 token。

**注意事项：**

- 为了提升效率，我们 **不合并跨 pre-token 边界的字节对**。

- 如果多个字节对频率相同，则选择**按字典序更大的那一对**。
   例如在下列字节对中频率相同的情况下：

  ```
  [("A", "B"), ("A", "C"), ("B", "ZZ"), ("BA", "A")]
  ```

  合并的将是：

  ```
  max(...) = ('BA', 'A')
  ```

------

 特殊 Token（Special Tokens）：

有些字符串，比如 `<|endoftext|>`，在编码时作为**元数据**使用（如表示文档之间的边界）。这些特殊 token 应该被**保留为一个整体**，永远不被拆分成多个子 token。

例如 `<|endoftext|>` 应始终是一个完整的 token（一个整数 ID），这样语言模型才能知道何时终止生成。
 所以这些特殊 token 必须手动加入词汇表中，且赋予固定的 token ID。



 Example (bpe_example): BPE training example 

Here is a stylized example from Sennrich et al. [2016]. Consider a corpus consisting of the following text 

low low low low low 

lower lower 

widest widest widest 

newest newest newest newest newest newest 

and the vocabulary has a special token <|endoftext|>. Vocabulary We initialize our vocabulary with our special token <|endoftext|> and the 256 byte values. Pre-tokenization For simplicity and to focus on the merge procedure, we assume in this example that pretokenization simply splits on whitespace. When we pretokenize and count, we end up with the frequency table. {low: 5, lower: 2, widest: 3, newest: 6} 

It is convenient to represent this as a dict[tuple[bytes], int], e.g. {(l,o,w): 5 …}. Note that even a single byte is a bytes object in Python. There is no byte type in Python to represent a single byte, just as there is no char type in Python to represent a single character. 

Merges Wefirst look at every successive pair of bytes and sum the frequency of the words where they appear {lo: 7, ow: 7, we: 8, er: 2, wi: 3, id: 3, de: 3, es: 9, st: 9, ne: 6, ew: 6}. The pair ('es') and ('st') are tied, so we take the lexicographically greater pair, ('st'). We would then merge the pre-tokens so that we end up with {(l,o,w): 5, (l,o,w,e,r): 2, (w,i,d,e,st): 3, (n,e,w,e,st): 6}. 

In the second round, we see that (e, st) is the most common pair (with a count of 9) and we would merge into {(l,o,w): 5, (l,o,w,e,r): 2, (w,i,d,est): 3, (n,e,w,est): 6}. Continuing this, the sequence of merges we get in the end will be ['s t', 'e st', 'o w', 'l ow', 'w est', 'n e', 'ne west', 'w i', 'wi d', 'wid est', 'low e', 'lowe r']. 

If we take 6 merges, we have ['s t', 'e st', 'o w', 'l ow', 'w est', 'n e'] and our vocab ulary elements would be [<|endoftext|>, [...256 BYTE CHARS], st, est, ow, low, west, ne]. 

With this vocabulary and set of merges, the word newest would tokenize as [ne, west].





##### 构建初始词表（vocab）

- 初始 vocab = 所有单独出现的 byte（0~255），每个 byte 是一个 `bytes` 类型：

  ```python
  from collections import defaultdict
  vocab = {i: bytes([i]) for i in range(256)}  # 0~255 的字节
  ```

- 加入 special tokens（比如 `"<pad>"`, `"<unk>"`）：

  ```
  for token in special_tokens:
      vocab[next_id] = token.encode('utf-8')
      next_id += 1
  ```

------

##### 训练 BPE 合并

**主循环**（直到 `len(vocab) >= vocab_size`）：

1. **统计所有 pair 出现次数**：

   ```
   counts[(token1, token2)] += 1
   ```

2. 找到出现最多的 pair：

   ```
   most_common = max(counts.items(), key=lambda x: x[1])
   ```

3. **合并这个 pair**，更新语料：
    把 `('a', 'b')` → `('ab')`，你可以用简单的贪心合并策略更新 corpus。

4. 将合并得到的新 token 加入 vocab，同时记录到 merges：

   ```python
   merges.append((token1, token2))
   vocab[next_id] = token1 + token2
   ```

------

##### 最终返回

```python
return vocab, merges
```


### BPE实现
**首先初始化vocab**

```py
vocab: Dict[int, bytes] = {i: bytes([i]) for i in range(256)}

next_id = 256
special_token_bytes = [token.encode("utf-8") for token inspecial_tokens]
for token_bytes in special_token_bytes:
    if token_bytes not in vocab.values():
        vocab[next_id] = token_bytes
        next_id += 1
```

将所有可能的单字节值（256个）和特殊字符进行编码

```py
bytes([65])  # 结果是 b'A'，也就是 ASCII 字母 A
bytes([97])  # 结果是 b'a'，也就是 ASCII 字母 a
bytes([255]) # 结果是 b'\xff'，即十六进制的 FF 字节
```


**创建一个字典用于统计频率**

```py
pre_tokens_cnt = defaultdict(int)
```

```py
from collections import defaultdict

pre_tokens_cnt = defaultdict(int)

tokens = ['a', 'b', 'a', 'c', 'b', 'a']

for token in tokens:
    pre_tokens_cnt[token] += 1

print(pre_tokens_cnt)
# 输出: {'a': 3, 'b': 2, 'c': 1}

```

然后
```py
chunks = re.split("|".join(map(re.escape, special_tokens)), text)
```


1. special_tokens
这是一个 特殊符号的列表，比如：

```python
special_tokens = ["<pad>", "<unk>", "<s>", "</s>", "<mask>"]
```
这些 token 通常在分词过程中不应该被合并，所以我们先从文本中把它们单独提取出来。

2. map(re.escape, special_tokens)
这个部分会对每个 special_token 进行 正则转义，以防特殊字符被正则当作元字符处理。

例如：

```python
re.escape("<pad>") => "<pad>"
re.escape("</s>")  => "<\/s>"
```

3. "|".join(...)
这会把多个转义后的特殊 token 用 | 连接成一个正则表达式：

```python
"<pad>|<unk>|<s>|<\/s>|<mask>"
| 是正则中的“或”，表示 匹配任意一个特殊 token。
```

4. re.split(...)
这个函数会用指定正则表达式去切割文本，将匹配的部分作为分隔符丢掉，其他部分保留。

例如：

```python
text = "hello<mask>world</s>again"
special_tokens = ["<mask>", "</s>"]


# 正则是 "<mask>|</s>"
chunks = re.split("|".join(map(re.escape, special_tokens)), text)

# 结果是 ['hello', 'world', 'again']
```

数据集中的chunks是文本列表
```
['\nOnce upon a time, there was a little girl named Sally. Sally loved to go to the theater with her mom and dad. One day, Sally\'s dad said, "Sally, do you remember the last time we went to the theater? We had so much fun!"\nSally thought for a moment and said, "Yes, I remember! We saw a show with singing and dancing. It was so much fun!"\nThe next day, Sally and her friends wanted to put on a show for their moms and dads. But they had a problem. Their friend Timmy was feeling weak and couldn\'t sing or dance. They all felt sad because they wanted Timmy to be in the show too.\nSally had an idea. She said, "Timmy, you can still be in the show! You can be the one who tells the story. That way, you don\'t have to sing or dance."\nTimmy liked the idea and they all worked together to put on a great show. In the end, everyone was happy, and they all remembered the fun they had at the theater.\n', '\nOnce upon a time, there was a messy giant named Bob. Bob lived in a big forest with many trees. Bob was always making a mess because he was so big and clumsy. One day, Bob decided he wanted to sell his big rocks to the people in the town.\nBob walked to the town with a big bag of rocks. He met a little girl named Sue. Sue looked at the rocks and said, "These rocks are very messy, but I can help you clean them." Bob was happy and thanked Sue. They cleaned the rocks together and made them shiny.\nAfter cleaning the rocks, Bob an']
```

```py
def to_bytes_tuple(word: str) -> Tuple[bytes]:
   # word: text
   # word.encode("utf-8"): b'text'
   # list(word.encode("utf-8")): [116, 101, 120, 116]
   # [bytes([x]) for x in l]: [b't', b'e', b'x', b't']
   # tuple(l): (b't', b'e', b'x', b't')
   l = list(word.encode("utf-8"))
   l = [bytes([x]) for x in l]
   return tuple(l)
```

在第一遍阅读词库的时候先对单词计数，可以使后面合并的过程更快

```py
        for token, cnt in pre_tokens_cnt.items():
            # Find all occurrences of the `best_pair` in `token`
            indices = [i for i in range(len(token) - 1) if token[i:i + 2] == best_pair]
            if indices:
                # Replace each occurrence with `new_token`
                new_pre_token = []
                i = 0
                while i < len(token):
                    if i in indices:
                        new_pre_token.append(new_token)
                        i += 2
                    else:
                        new_pre_token.append(token[i])
                        i += 1
                new_pre_token = tuple(new_pre_token)
                changes.append((token, new_pre_token, cnt))

```
再一次遍历词库，一对一对遍历，如果是该对是`new_token`，就加入，反之就加入一个


## 3 Transformer Language Model Architecture

语言模型的输入是一个批量化的整数 token ID 序列（即形状为 (batch_size, sequence_length) 的 torch.Tensor）。它的输出是一个（批量化的）针对词汇表的归一化概率分布（即形状为 (batch_size, sequence_length, vocab_size) 的 PyTorch 张量），其中每个位置的预测分布表示该位置的下一个词的概率。

在训练语言模型时，我们会使用这些“下一个词”的预测结果，与真实的下一个词进行比较，计算交叉熵损失。

在推理阶段（生成文本时），我们会取序列最后一个时间步（即序列的最后一个位置）的预测分布，用来生成下一个 token（例如，可以选择概率最高的 token，或者按概率分布进行采样）。然后，将生成的 token 添加到输入序列的末尾，并重复这一过程。

### 3.1 Transformer 语言模型
给定一个 token ID 序列，Transformer 语言模型会先通过输入嵌入层（input embedding）将 token ID 转换为稠密向量，然后将嵌入后的 token 序列传入 num_layers 个 Transformer 块中进行处理，最后通过一个可学习的线性映射（称为“输出嵌入”或 “LM head”）生成预测的下一个 token 的 logits。图 1 给出了该过程的示意图。

#### 3.1.1 Token Embeddings（词嵌入）
在第一步，Transformer 会将（批量化的）token ID 序列嵌入为向量序列，这些向量包含了 token 的身份信息（见图 1 中的红色方块）。

更具体地说，给定一个 token ID 序列，Transformer 语言模型会使用一个 token embedding 层来生成向量序列。每个嵌入层的输入是一个形状为 (batch_size, sequence_length) 的整数张量，输出是一个形状为 (batch_size, sequence_length, d_model) 的向量序列。(d_model表示embedding的维度)

#### 3.1.2 Pre-norm Transformer Block（预归一化 Transformer 块）
在嵌入之后，这些激活值会被送入若干个结构完全相同的神经网络层中处理。标准的仅解码（decoder-only）Transformer 语言模型由 num_layers 个相同的层（通常称为 Transformer “块”）组成。

每个 Transformer 块的输入形状为 (batch_size, sequence_length, d_model)，输出形状同样为 (batch_size, sequence_length, d_model)。每个块会通过自注意力机制（self-attention）在序列中聚合信息，并通过前馈网络（feed-forward layers）进行非线性变换。

### 3.2 输出归一化与嵌入
在经过 num_layers 个 Transformer 块之后，我们会将最终的激活值转换为针对词汇表的概率分布。
我们将实现“预归一化”（pre-norm）的 Transformer 块（详见 §3.5），这要求在最后一个 Transformer 块之后进行层归一化（layer normalization），以确保输出的数值范围合适。
完成归一化后，我们会使用一个标准的、可学习的线性变换，将 Transformer 块的输出转换为预测的下一个 token 的 logits。

### 3.3 Remark: Batching, Einsum, and Efficient Computation

### 3.4 Basic Building Blocks: Linear and Embedding Modules

#### 3.4.1
在训练神经网络时，通常需要对模型参数进行**精心的初始化**——糟糕的初始化可能会导致一些不良现象，例如梯度消失（vanishing gradients）或梯度爆炸（exploding gradients）。**预归一化（Pre-norm）Transformer** 对初始化的鲁棒性（robustness）比一般模型要高，但初始化方式仍然会显著影响训练速度和收敛效果。  

由于本次作业内容已经比较多，我们会把具体细节留到作业 3 中讲解，这里先给出一些在大多数情况下都能表现良好的近似初始化方法：  

- **线性层（Linear）权重**：  
  从均值 μ = 0、方差 σ² = 2 / (d_in + d_out) 的正态分布中采样，并截断在区间 [−3σ, 3σ] 内。  
- **嵌入层（Embedding）**：  
  从均值 μ = 0、方差 σ² = 1 的正态分布中采样，并截断在区间 [−3, 3] 内。  
- **RMSNorm 层**：  
  初始化为 1。  

并使用 `torch.nn.init.trunc_normal_` 来初始化这些截断正态分布的权重。
#### 3.4.2 Linear

```py
class Linear(nn.Module):
    """
    Given the weights of a Linear layer, compute the transformation of a batched input.

    Args:
        in_dim (int): The size of the input dimension
        out_dim (int): The size of the output dimension
        weights (Float[Tensor, "d_out d_in"]): The linear weights to use
        in_features (Float[Tensor, "... d_in"]): The output tensor to apply the function to

    Returns:
        Float[Tensor, "... d_out"]: The transformed output of your linear module.
    """
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype

        # 线性层的投影矩阵，输出维度在前，输入维度在后
        W = torch.empty(out_features, in_features)
        mean = 0
        std = np.sqrt(2 / in_features + out_features)
        nn.init.trunc_normal_(W, mean, std, -3*std, 3*std)
        self.W = nn.Parameter(W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.W.T
```


#### 3.4.3 Embedding

```py
import torch
import torch.nn as nn

class Embedding(nn.Module):

    """
    Given the weights of an Embedding layer, get the embeddings for a batch of token ids.

    Args:
        vocab_size (int): The number of embeddings in the vocabulary
        d_model (int): The size of the embedding dimension
        weights (Float[Tensor, "vocab_size d_model"]): The embedding vectors to fetch from
        token_ids (Int[Tensor, "..."]): The set of token ids to fetch from the Embedding layer

    Returns:
        Float[Tensor, "... d_model"]: Batch of embeddings returned by your Embedding layer.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, 
                device: torch.device | None = None, 
                dtype: torch.dtype | None = None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.device = device
        self.dtype = dtype

        embedding_matrix = torch.empty(num_embeddings, embedding_dim)
        mean = 0
        std = 1
        nn.init.trunc_normal_(embedding_matrix, mean, std, -3*std, 3*std)
        self.embedding_matrix = nn.Parameter(embedding_matrix)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = token_ids.shape
        out_put = torch.empty(batch_size, seq_len, self.embedding_dim)

        # enumerate 默认会从第一维开始遍历，每一个batch是一个seq
        for i, seq in enumerate(token_ids):
            for j, token_id in enumerate(seq):
                out_put[i][j] = self.embedding_matrix[token_id]
        
        return out_put

```