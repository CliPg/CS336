## 1 作业概述

在本次作业中，你将亲身体验如何训练语言模型去**进行推理以解决数学问题**。

### 你需要实现的内容

1. **零样本提示（Zero-shot prompting）**：
   在 MATH 数据集（竞赛数学题，Hendrycks 等人，2021）上测试零样本推理能力。

2. **有监督微调（Supervised Finetuning, SFT）**：
   使用来自更强推理模型（DeepSeek R1，DeepSeek-AI 等人，2025）的推理轨迹进行微调。

3. **专家迭代（Expert Iteration）**：
   使用经过验证的奖励（verified rewards）提升模型推理性能。

4. **组相对策略优化（Group-Relative Policy Optimization, GRPO）**：
   同样使用经过验证的奖励来进一步提升模型的推理性能。

> 对感兴趣的同学，我们还会在接下来几天发布一个**可选部分**，内容为如何让语言模型**对齐人类偏好（human preference alignment）**。

---

### 你需要运行的实验

1. **测量基线性能**：
   测试 Qwen 2.5 Math 1.5B 模型在 MATH 数据集上的零样本提示表现。

2. **SFT 微调**：
   使用 DeepSeek R1 的推理轨迹对 Qwen 2.5 Math 1.5B 进行有监督微调。

3. **专家迭代（Expert Iteration）**：
   在 Qwen 2.5 Math 1.5B 上运行专家迭代算法，并使用验证奖励。

4. **GRPO 训练**：
   在 Qwen 2.5 Math 1.5B 上运行组相对策略优化（GRPO）算法，并使用验证奖励。

---

### 代码结构

所有作业代码和说明都在 GitHub 上：
👉 **[github.com/stanford-cs336/assignment5-alignment](https://github.com/stanford-cs336/assignment5-alignment)**

请先 **git clone** 该仓库。若有更新，我们会通知你执行 **git pull** 以获取最新内容。

仓库结构如下：

1. **`cs336_alignment/*`**
   这是你要编写代码的主要目录（作业 5）。
   除了一些起始代码（starter code）外，这个目录基本是空的，你可以从零开始实现。

2. **`cs336_alignment/prompts/*`**
   我们为你准备了文本文件，其中包含提示语（prompts）。
   这样可以避免你从 PDF 复制粘贴提示时可能出现的格式错误。

3. **`tests/*.py`**
   包含所有需要通过的测试。
   你只需要通过以下两个文件中的测试：

   * `tests/test_sft.py`
   * `tests/test_grpo.py`

   其他测试属于非必做部分。
   这些测试会调用 `tests/adapters.py` 中定义的钩子函数（hooks）。
   你需要实现这些 adapter 函数，以便测试能调用你的代码。

   你也可以编写或修改测试代码来调试自己的实现，但最终版本必须通过**原始提供的测试套件**。

4. **`README.md`**
   包含环境配置和运行的基本说明。

---

### 使用限制

我们希望你**从零开始**实现大部分强化学习（RL）相关的组件。

但你可以使用以下工具：

* **vLLM**：用于从语言模型生成文本（见 §3.1）
* **HuggingFace Transformers**：用于加载 Qwen 2.5 Math 1.5B 模型和分词器，并进行前向计算（见 §4.1）

⚠️ 注意：你**不能使用** HuggingFace 的训练工具，例如 `Trainer` 类。

---

### 提交方式

你需要向 Gradescope 提交以下两个文件：

1. **`writeup.pdf`**：
   包含所有书面回答，请使用排版工具（如 LaTeX）进行整洁排版。

2. **`code.zip`**：
   包含你编写的所有源代码。

---


## 2 推理与语言模型

### 2.1 动机

语言模型的一个显著应用是构建能够处理广泛自然语言处理任务的通用系统。在本次作业中，我们将关注语言模型的一个正在发展中的用例：**数学推理**。它将作为我们的试验场，用于搭建评估、进行监督微调，并尝试使用强化学习（RL）教语言模型如何进行推理。

与我们以往的作业相比，会有两点不同：

* 首先，我们不会再使用之前的语言模型代码库和模型。理论上我们希望基于以前作业训练出的基础模型进行微调，但那些模型太弱，无法在复杂的数学推理任务上展现出有意义的能力。因此我们将改用一个现代的、高性能且可访问的基础模型（Qwen 2.5 Math 1.5B Base），并在其上开展大部分工作。

* 其次，我们将引入一个新的评测基准来评估语言模型。到目前为止，我们一直认为交叉熵是许多下游任务的良好替代指标。然而，本次作业的重点是把基础模型与实际下游任务之间的差距弥合，因此我们必须使用与交叉熵分开且更直接的评估方法。我们将使用 Hendrycks 等人 [2021] 的 MATH 12K 数据集（包含具有挑战性的高中竞赛数学题）。评估时，我们将把语言模型的输出与参考答案进行比较来判定正确性。

### 2.2 链式思维（Chain-of-Thought, CoT）推理与基于推理的强化学习

近来一个令人兴奋的趋势是使用**链式思维推理**来提升模型在多种任务上的表现。所谓链式思维，是指在得出最终答案之前，模型通过逐步生成中间推理步骤来一步一步地推理问题。

**链式思维与大型语言模型。** 早期的链式思维方法通过微调语言模型让其在解决简单数学任务（例如算术）时使用“草稿板（scratchpad）”把问题拆成中间步骤 [Nye et al., 2021]。其他工作通过提示强模型“逐步思考（think step by step）”再回答，发现这能显著提高模型在数学推理任务（例如中小学数学题）上的表现 [Wei et al., 2023]。

**用专家迭代学习推理。** Self-Taught Reasoner（STaR）[Zelikman et al., 2022] 将推理过程构建为一个自举（bootstrapping）循环：先用预训练模型采样多样的链式思维（CoTs），仅保留那些最终得出正确答案的推理轨迹作为“专家”示例，然后在这些专家轨迹上做微调。迭代执行该循环可以提升语言模型的推理能力与解题成功率。STaR 展示了这种基于自动化、字符串匹配式答案验证的专家迭代方法可以在没有人工编写推理示例的情况下，自主引导模型学习推理技能。

**使用经验证奖励的推理强化学习（o1 与 R1 等）。** 最近的工作探索了使用更强的强化学习算法结合经验证的奖励来提升推理表现。OpenAI 的 o1（以及后续的 o3/o4）[OpenAI et al., 2024]、DeepSeek 的 R1 [DeepSeek-AI et al., 2025] 和 Moonshot 的 kimi k1.5 [Team et al., 2025] 使用策略梯度方法 [Sutton et al., 1999] 在数学和代码任务上训练模型，其中通过字符串匹配或单元测试来验证答案的正确性，显示出在竞赛数学与编程任务上显著的性能提升。后续工作（例如 Open-R1 [Face, 2025]、SimpleRL-Zoo [Zeng et al., 2025]、TinyZero [Pan et al., 2025]）也证实了：即便是在参数规模只有 1.5B 的模型上，使用经验证奖励的纯强化学习也能改进模型的推理性能。


---

### 我们的设置：模型与数据集

在接下来的章节中，我们将逐步探讨如何通过越来越复杂的方法来训练一个基础语言模型，使其能够**逐步推理（step-by-step reasoning）**来解决数学问题。

在本次作业中，我们将使用 **Qwen 2.5 Math 1.5B Base** 模型。
该模型是在 **Qwen 2.5 1.5B** 模型的基础上，持续预训练（continual pretraining）得到的，预训练数据是**高质量的合成数学数据集** [Yang et al., 2024]。

**MATH 数据集** 在 Together 集群上的路径如下：

```
/data/a5-alignment/MATH
```

---

#### 💡 对开源审阅者的提示：可替代数据集

遗憾的是，由于版权问题，**MATH 数据集无法公开获取**。
如果你是在本地或非 Together 环境下完成本作业，可以使用以下**开源数学推理数据集**作为替代：

* **Countdown** [Pan et al., 2025]
  简单的合成任务，灵感来自英国电视节目 *Countdown*。
  它是小规模推理强化学习（Reasoning RL）的常用测试平台。

* **GSM8K** [Cobbe et al., 2021a]
  小学数学题数据集，难度比 MATH 低。
  适合用于调试代码正确性并熟悉整个推理强化学习的训练流程。

* **Tulu 3 SFT Math** [Lambert et al., 2025]
  由 GPT-4o 和 Claude 3.5 Sonnet 生成的合成数学题。
  由于是自动生成的，因此**部分题目或答案可能不完全正确**。

* **其他数学 SFT 数据集**
  （官方文档中提供了链接，可按需选择使用。）

---

如果这些替代数据集中没有直接提供标准答案（例如仅有 “1/2” 这样的简短标签），
你可以使用 **数学答案解析器（Math-Verify）** 来处理数据集中的 `ground-truth` 列，以提取出正确答案。

---

## 3 测量零样本（Zero-Shot）MATH 数据集性能

我们首先要在 **MATH 数据集的 5K 测试集**上测量基础语言模型的性能。
建立这个基线（baseline）有助于我们理解后续不同训练方法如何影响模型行为。

除非另有说明，在 MATH 实验中，我们将使用来自 **DeepSeek R1-Zero 模型** [DeepSeek-AI et al., 2025] 的以下提示词（prompt），称为 **`r1_zero` prompt**：

```
A conversation between User and Assistant. The User asks a question, and the Assistant
solves it. The Assistant first thinks about the reasoning process in the mind and
then provides the User with the answer. The reasoning process is enclosed within
<think> </think> and answer is enclosed within <answer> </answer> tags, respectively,
i.e., <think> reasoning process here </think> <answer> answer here </answer>.

User: {question}
Assistant: <think>
```

该提示词文件位于：

```
cs336_alignment/prompts/r1_zero.prompt
```

其中 `{question}` 表示要插入的问题，例如：

> Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?
> （娜塔莉亚在四月卖出了 48 个发夹给她的朋友们，五月卖出了一半。她两个月一共卖出了多少个发夹？）

模型需要扮演 “Assistant” 的角色，开始生成推理过程（因为提示中已经包含了左括号 `<think>`），然后用 `</think>` 结束推理部分，并在 `<answer>` 标签中生成最终的符号化答案，例如：

```
<answer> 4x + 10 </answer>
```

这种标签结构的设计有两个目的：

1. 方便我们从输出中自动解析出模型的答案；
2. 允许我们在检测到 `</answer>` 时自动停止生成。

---

### 关于提示词选择的说明

事实证明，`r1_zero` prompt **并不是**强化学习（RL）训练后获得最佳下游性能的最优选择。
原因是：它与 Qwen 2.5 Math 1.5B 的预训练格式存在一定**不匹配**。

Liu 等人 [2025] 发现，仅仅将问题本身作为提示（即所谓的 **`question_only` prompt**），就能在零样本下达到很高的初始准确率。例如，经过 100 多步 RL 训练后，其效果可与 `r1_zero` prompt 相当。

这表明 **Qwen 2.5 Math 1.5B 在预训练阶段已经见过大量问答格式的数据**。

不过，在本次作业中，我们仍然选择使用 `r1_zero` prompt，因为使用它进行强化学习时，模型的准确率在短时间内会有明显提升。这有助于我们：

* 快速验证 RL 算法机制；
* 检查训练是否正常；
* 即使最终性能不是最优，也能快速验证实验流程。

作为一个“现实校验”，你将在作业后面直接对比 `question_only` prompt 的表现。

---

### 3.1 使用 vLLM 进行离线语言模型推理（Offline Inference）

为了评估语言模型的性能，我们需要针对多个提示生成模型回复（continuations）。
虽然你可以像作业 1 那样自己实现生成函数，但 RL 训练需要高效推理（high-performance inference）。
自行实现高性能推理技术超出了本次作业的范围，因此本次作业推荐使用 **vLLM** 来进行离线批量推理（offline batched inference）。

**vLLM** 是一个高吞吐量、内存高效的语言模型推理引擎，它融合了多种优化技术（如优化 CUDA 内核、PagedAttention 高效缓存 KV 等 [Kwon et al., 2023]）。

---

#### 使用 vLLM 生成文本的示例

```python
from vllm import LLM, SamplingParams

# 示例提示词
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]

# 设置采样参数（遇到换行符停止生成）
sampling_params = SamplingParams(
    temperature=1.0, top_p=1.0, max_tokens=1024, stop=["\n"]
)

# 创建 LLM 对象
llm = LLM(model=<path to model>)

# 根据提示词生成文本
outputs = llm.generate(prompts, sampling_params)

# 输出结果
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
```

> 示例摘自：[https://github.com/vllm-project/vllm/blob/main/examples/offline_inference.py](https://github.com/vllm-project/vllm/blob/main/examples/offline_inference.py)

---

#### 模型加载说明

在上述示例中，`LLM` 可以通过以下两种方式初始化：

1. 传入 HuggingFace 模型名称（vLLM 会自动下载并缓存）；
2. 或传入本地 HuggingFace 模型路径。

由于大型模型（如 70B）下载耗时长，且会占用大量磁盘空间，
为了节省 Together 集群资源，**课程已预先下载以下模型**，请不要在集群中重复下载：

| 模型名称                        | 用途         | Together 集群路径                                      |
| --------------------------- | ---------- | -------------------------------------------------- |
| **Qwen 2.5 Math 1.5B Base** | 数学推理实验     | `/data/a5-alignment/models/Qwen2.5-Math-1.5B`      |
| **Llama 3.1 8B Base**       | （可选）指令微调实验 | `/data/a5-alignment/models/Llama-3.1-8B`           |
| **Llama 3.3 70B Instruct**  | （可选）指令微调实验 | `/data/a5-alignment/models/Llama-3.3-70B-Instruct` |

---


以下是 **3.2 Zero-shot MATH Baseline**（零样本数学基线）的完整中文翻译：

---

### 3.2 零样本 MATH 基线（Zero-shot MATH Baseline）

#### 提示设置（Prompting setup）

为了评估模型在 MATH 测试集上的零样本（zero-shot）表现，我们将**直接加载测试样本**，并使用上文提到的 **r1_zero** 提示模板（prompt）来让语言模型回答问题。

---

#### 评估指标（Evaluation metric）

在**多选题或二分类任务**中，评估指标非常直观——我们只需判断模型输出是否**与正确答案完全一致**。

但在**数学题**中，虽然我们有明确的标准答案（例如 `0.5`），却不能仅仅判断模型输出是否是 `0.5`，
因为模型也可能输出 `<answer> 1/2 </answer>`。

因此，我们必须解决一个棘手的问题：
**如何在评估时判断模型输出与标准答案“语义上等价”**。

为此，我们需要设计一个**答案解析函数（answer parsing function）**，
该函数接受模型输出与标准答案作为输入，并返回一个布尔值（True/False）表示模型回答是否正确。

例如，一个奖励函数（reward function）可能收到如下输入：
模型输出：`<answer> She sold 15 clips. </answer>`
标准答案（gold answer）：`72`
此时该函数应返回 **False**（因为模型答案错误）。

---

#### 我们的奖励函数（Reward function）

在本次 MATH 实验中，我们将使用近期推理强化学习（Reasoning RL）工作 [Liu et al., 2025] 中提出的**快速且准确的答案解析器**。

该奖励函数已经在以下模块中实现：

```
cs336_alignment.drgrpo_grader.r1_zero_reward_fn
```

除非另有说明，你应当使用这个函数来评估模型在 MATH 上的表现。

---

#### 生成超参数（Generation hyperparameters）

在生成模型输出时，我们采用如下采样设置：

* 温度（temperature） = 1.0
* top-p = 1.0
* 最大生成长度（max generation length） = 1024

提示模板要求模型在回答结尾输出 `</answer>` 字符串，因此我们可以指示 vLLM 在生成到此字符串时停止：

```python
# Based on Dr. GRPO: stop when the model completes its answer
# https://github.com/sail-sg/understand-r1-zero/blob/
# c18804602b85da9e88b4aeeb6c43e2f08c594fbc/train_zero_math.py#L167
sampling_params.stop = ["</answer>"]
sampling_params.include_stop_str_in_output = True
```

---

#### 📘 题目 (math_baseline)：共 4 分

##### (a) 编写评估脚本

编写一个脚本，用于评估 **Qwen 2.5 Math 1.5B** 在 MATH 数据集上的**零样本性能**。

该脚本应当完成以下任务：

1. 从

   ```
   /data/a5-alignment/MATH/validation.jsonl
   ```

   加载 **MATH 验证集样本**；
2. 使用 **r1_zero 提示模板** 将样本格式化为字符串输入；
3. 让语言模型对每个样本生成输出；
4. 使用奖励函数计算评估指标；
5. 将**样本、模型生成结果、评估得分**序列化保存到磁盘，供后续分析使用。

实现时，可以定义如下函数（方便后续复用）：

```python
def evaluate_vllm(
    vllm_model: LLM,
    reward_fn: Callable[[str, str], dict[str, float]],
    prompts: List[str],
    eval_sampling_params: SamplingParams
) -> None:
    """
    在一组提示上评估语言模型，
    计算评估指标，并将结果序列化保存。
    """
```

📤 **提交内容**：一个脚本，用于评估 Qwen 2.5 Math 1.5B 的零样本基线性能。

---

##### (b) 运行并分析评估结果

运行你的评估脚本，并统计模型生成结果在以下三种类别中的数量：

1. **格式奖励为 1 且答案奖励为 1**（格式正确、答案正确）
2. **格式奖励为 1 但答案奖励为 0**（格式正确、答案错误）
3. **格式奖励为 0 且答案奖励为 0**（格式错误、答案错误）

观察至少 10 个“格式奖励为 0”的样例，
思考问题出在 **模型输出格式** 还是 **解析器（parser）**？为什么？

再观察至少 10 个“格式奖励为 1 但答案奖励为 0”的样例，
分析问题出在 **模型推理错误** 还是 **奖励函数误判**？

📤 **提交内容**：
对模型与奖励函数表现的评论，包括每一类样例的示例与分析。

---

####3 (c) 总体表现

总结 Qwen 2.5 Math 1.5B 在 MATH 数据集上的零样本基线性能。

📤 **提交内容**：
1～2 句简要总结，包括评估指标（例如准确率或奖励得分）。


## 4 MATH 的监督微调（Supervised Finetuning for MATH）

算法 1：监督微调（SFT）

输入：初始策略模型$ \pi_{\theta_\text{init}}$；SFT 数据集 D
1.	将策略模型$ \pi_\theta \gets \pi_{\theta_\text{init}}$
2. for step = 1, …, n_sft_steps do：
3. 从数据集 D 中抽取一个问题-回答对批次 D_b
4. 使用模型$ \pi_\theta $计算回答相对于问题的 交叉熵损失（cross-entropy loss）
5. 对模型参数$ \theta $进行梯度更新（gradient step）
6.	结束循环

输出：微调后的模型 $\pi_\theta$



推理任务的监督微调

在本节中，我们将对基础模型在 MATH 数据集上进行微调（见算法 1）。
由于我们的目标是提升模型的 推理能力，而不是直接预测正确答案，我们会微调模型，使其 先生成思路链（chain-of-thought reasoning trace），再生成答案。

为此，我们提供了一个包含推理链的数据集，数据来源于 DeepSeek R1 DeepSeek-AI 等人 [2025]，存放路径为：

/data/a5-alignment/MATH/sft.jsonl

在实际训练推理模型时，SFT 通常作为第二步 RL 微调（Reinforcement Learning Fine-Tuning）的 warm-start（预热）。原因有两个：
1.	SFT 需要高质量标注数据（即已有推理链的数据），而 RL 只需要正确答案作为反馈。
2.	即便标注数据充足，RL 仍能通过寻找比 SFT 数据更优的策略来提升性能。

不过，本作业中使用的模型规模较小，无法展示 SFT 与 RL 结合的效果，因此本作业将 分别处理 SFT 与 RL 两个阶段。



### 4.1 使用 HuggingFace 模型

加载 HuggingFace 模型和 tokenizer
要从本地目录加载 HuggingFace 模型和 tokenizer（使用 bfloat16 并启用 FlashAttention-2 以节省显存），可以使用以下示例代码：
```
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "/data/a5-alignment/models/Qwen2.5-Math-1.5B",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)

tokenizer = AutoTokenizer.from_pretrained(
    "/data/a5-alignment/models/Qwen2.5-Math-1.5B"
)
```


#### 前向传播（Forward pass）

加载模型后，可以对一批输入 ID 进行前向传播，并获取输出的 logits（通过 .logits 属性）。
然后，可以计算模型预测的 logits 与真实标签之间的损失：
```
input_ids = train_batch["input_ids"].to(device)
labels = train_batch["labels"].to(device)
logits = model(input_ids).logits # logits是未经过softmax的，对输入的下一个token在词表的预测分数。
loss = F.cross_entropy(logits, labels)
```



#### 保存训练好的模型（Saving a trained model）

训练完成后，要将模型保存到某个目录，可以使用 .save_pretrained() 方法，并传入目标输出目录的路径。
注意，由于模型可能比较大，请确保保存路径在 /data/yourusername 下。

我们也建议保存 tokenizer（即使没有修改过），这样模型和 tokenizer 可以一起存放在同一个目录中，方便后续加载。

```
# 保存模型权重
model.save_pretrained(save_directory=output_dir)
tokenizer.save_pretrained(save_directory=output_dir)
```


#### 梯度累积（Gradient accumulation）

即使将模型加载为 bfloat16 并使用 FlashAttention-2，即便使用 80GB GPU，也可能无法支持较大的 batch size。
为了使用更大的 batch size，可以采用 梯度累积（gradient accumulation） 技术。

基本思想：
- 通常，每个 batch 计算完梯度后就更新模型权重（optimizer.step()）。
- 梯度累积则是在 多个 batch 上累积梯度，然后再进行一次梯度更新。
- 直观理解：如果你有更大显存，一次性计算 32 个样本的梯度，效果等同于把 32 个样本拆成 16 个 batch, 每个batch 2 个样本，然后在累积梯度后平均更新一次。



在 PyTorch 中实现梯度累积
- 每个权重张量都有 .grad 属性，用于存储梯度。
- 在调用 loss.backward() 前，.grad 是 None。
- 调用 loss.backward() 后，.grad 中就存储了梯度。
- 通常，我们会执行：
	1.	optimizer.step() 更新权重
	2.	optimizer.zero_grad() 清空梯度，为下一轮计算做准备

示例（普通梯度更新）：
```
for inputs, labels in data_loader:
    # 前向传播
    logits = model(inputs)
    loss = loss_fn(logits, labels)
    
    # 反向传播
    loss.backward()
    
    # 更新权重
    optimizer.step()
    
    # 清空梯度
    optimizer.zero_grad()

```


实现梯度累积
- 每隔 k 个 batch 再执行一次 optimizer.step() 和 optimizer.zero_grad()
- 在调用 loss.backward() 前，将 loss 除以 gradient_accumulation_steps，这样梯度在累积过程中会被平均

示例代码：
```
gradient_accumulation_steps = 4

for idx, (inputs, labels) in enumerate(data_loader):
    # 前向传播
    logits = model(inputs)
    loss = loss_fn(logits, labels) / gradient_accumulation_steps
    
    # 反向传播
    loss.backward()
    
    # 每 `gradient_accumulation_steps` 个 batch 更新一次权重
    if (idx + 1) % gradient_accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```
这样就可以在显存有限的情况下模拟较大的 batch size，提升训练稳定性。



### 4.2 SFT 辅助方法（SFT Helper Methods）

在训练过程中，由于梯度累积，有效的 batch size 会被乘以 k，也就是梯度累积的步数。

接下来，我们将实现一些在 SFT 和后续 RL 实验中会用到的辅助方法。

关于术语说明：在下文中，我们会交替使用“output”、“completion” 或 “response” 来表示模型在给定 prompt 后生成的结果。


#### Tokenizing prompts and outputs

对于每一对问题和目标输出 (q, o)：
1.	我们会分别对问题和输出进行分词（tokenize）
2.	然后将它们 连接在一起（concatenate）

这样，我们就可以用 SFT 模型（或者后续 RL 策略）计算输出部分的 log-probabilities。

此外，我们需要构建一个 response_mask：
- 对输出 tokens（response tokens）对应位置为 True
- 对问题 tokens（prompt tokens）和 padding tokens 对应位置为 False

在训练循环中，这个 mask 会确保 只对 response tokens 计算损失。



##### Problem tokenize_prompt_and_output

>任务：实现 tokenize_prompt_and_output 方法，对问题和输出分别分词，连接起来，并构建 response_mask。

推荐接口：
```
def tokenize_prompt_and_output(prompt_strs, output_strs, tokenizer):
    """
    对 prompt 和 output 字符串进行分词，并构建 response_mask。
    
    Args:
        prompt_strs: list[str]   # prompt 字符串列表
        output_strs: list[str]   # output 字符串列表
        tokenizer: PreTrainedTokenizer  # 用于分词的 tokenizer
    
    Returns:
        dict[str, torch.Tensor]  # 包含 input_ids, labels, response_mask
    """
```
返回字典中应包含以下 key：

|key	| 说明|
|---|---|
|input_ids|	torch.Tensor，形状 (batch_size, max(prompt_and_output_lens) - 1)：分词后的 prompt + output 字符串，去掉最后一个 token
|labels	|torch.Tensor，形状 (batch_size, max(prompt_and_output_lens) - 1)：shifted input ids，即 input_ids 去掉第一个 token
|response_mask|	torch.Tensor，形状 (batch_size, max(prompt_and_output_lens) - 1)：用于 labels 中 response tokens 的 mask

注：prompt_and_output_lens 是每个样本分词后长度的列表。


测试方法

为了测试你的实现：
1.	在 [adapters.run_tokenize_prompt_and_output] 中调用你的函数
2.	然后运行测试：

```
uv run pytest -k test_tokenize_prompt_and_output
```
确保实现通过测试。



#### 记录每个 token 的熵（entropy）：
在强化学习（RL）训练中，记录模型每个 token 的预测熵是非常有用的。
这样可以帮助我们观察模型的预测分布是否变得“过于自信”（即概率分布过于尖锐）。

熵的定义如下：
$$
H(p) = -\sum_{x \in X} p(x) \log p(x)
$$
其中 p(x) 是离散分布上的概率。

在本题中，我们需要计算模型的 每个 token 的预测熵，即针对“下一个 token 的预测分布”计算熵。



##### Problem：计算每个 token 的熵

题目要求：
实现一个函数 compute_entropy，用于计算每个 token 的预测熵。

推荐接口：
```
def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """
    计算下一个 token 的预测熵（在词汇表维度上求熵）
    参数：
        logits: torch.Tensor
            形状为 (batch_size, sequence_length, vocab_size)
            含有模型的未归一化 logits。
    返回：
        torch.Tensor
            形状为 (batch_size, sequence_length)
            每个 token 的预测熵。
    """
```
注意：
为了避免数值溢出（overflow），需要使用数值稳定的算法，例如：
torch.logsumexp。

测试方法：
实现好后，运行以下命令：

uv run pytest -k test_compute_entropy

通过测试即可。


#### 从模型中获取 log-probabilities

在之后的 SFT（监督微调） 和 RL（强化学习） 中，我们需要从语言模型中获取 log-probabilities。

给定：
- 一个前缀 x
- 模型的输出 $logits f_\theta(x) \in \mathbb{R}^{|V|}$
- 标签 token $y \in V$

其 log 概率为：
$$
\log p_\theta(y|x) = \log [\text{softmax}(f_\theta(x))]_y
$$
其中 $[x]_y $表示向量 x 的第 y 个元素。


##### Problem：实现 get_response_log_probs

任务：
实现一个函数 get_response_log_probs，计算因果语言模型（causal LM）中每个 token 的条件 log 概率（即给定前面 tokens 的 log p(y|x)）。

此外：
还可以选择性地计算并返回 每个 token 的预测熵（使用上一个函数 compute_entropy）。

推荐接口：

```
def get_response_log_probs(model, input_ids, attention_mask=None, return_entropy=False):
    """
    获取每个 token 的条件 log 概率（以及可选的熵）

    参数：
        model: 语言模型（如 GPT2）
        input_ids: torch.Tensor，模型输入的 token ids
        labels: torch.Tensor，可选，用于掩码
        return_entropy: bool，是否返回每个 token 的熵

    返回：
        log_probs: torch.Tensor，每个 token 的 log p(y|x)
        entropies: torch.Tensor（可选），每个 token 的预测熵
    """

```


#### SFT microbatch train step.

在 SFT阶段，我们的目标是最小化目标输出在给定 prompt 条件下的负对数似然（Negative Log-Likelihood, NLL）损失。

要计算这个损失：
- 我们需要计算目标输出在 prompt 条件下的对数概率（log-probabilities）；
- 然后对输出序列中所有的 token 求和；
- 但要屏蔽掉 prompt 部分的 token（这些不是我们要预测的）；
- 同时也要屏蔽掉 padding token（填充部分不应参与计算）。


##### Problem: masked_normalize

你需要实现一个函数，用于在张量上进行 带掩码（mask）的求和与归一化操作。
这个函数在 SFT 和后续 RL（强化学习）阶段都会被使用。

函数接口推荐：
```
def masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    normalize_constant: float,
    dim: int | None = None,
) -> torch.Tensor:

```

函数功能说明：

对 tensor 中的元素进行加权求和与归一化，只在 mask == 1 的位置上进行计算。

参数说明：
- tensor: torch.Tensor
要进行求和和归一化的张量。
- mask: torch.Tensor
与 tensor 形状相同。mask == 1 的位置被计入求和，mask == 0 的位置会被忽略。
- normalize_constant: float
用于归一化的常数，即最终结果会除以这个常数。
- dim: int | None
指定在哪个维度上求和。如果为 None，则在所有维度上求和。

返回值：
- 一个 torch.Tensor，表示在掩码下求和后除以 normalize_constant 得到的归一化结果。
（即：掩码为 0 的位置不参与计算）

uv run pytest -k test_masked_normalize

测试文件会调用 adapters.run_masked_normalize 来检查你的函数是否实现正确。


现在，我们准备实现 SFT（Supervised Fine-Tuning） 的单个微批次（microbatch）训练步骤。（回忆一下：在一个训练的 minibatch 中，如gradient_accumulation_steps > 1，我们会循环处理多个 microbatch。）


##### Problem (sft_microbatch_train_step): 微批次训练步骤（3 分）

实现一个 SFT 的单个微批次更新（micro-batch update），包括：
- 计算交叉熵损失（cross-entropy loss）；
- 使用掩码（mask）进行加权求和；
- 进行梯度缩放（gradient scaling）。

推荐函数接口
```
def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
```

参数说明

|参数名|类型|说明|
|---|---|---|
|policy_log_probs|	(batch_size, sequence_length) 的张量|	模型输出的每个 token 的对数概率（log-probabilities），来自正在训练的 SFT 策略模型
response_mask	|(batch_size, sequence_length) 的张量|	掩码，1 表示属于 response 的 token（应计算损失），0 表示 prompt 或 padding（不计算）
gradient_accumulation_steps|	int|	每次优化器更新前累计的微批次数
normalize_constant	|float，默认 1.0|	用于除法归一化的常数，可以保持为 1.0


返回值

tuple[torch.Tensor, dict[str, torch.Tensor]]

|返回项|类型|说明|
|-|-|-|
|loss|	torch.Tensor（标量）|	当前微批次的损失，已经根据梯度累积进行缩放（即除以 gradient_accumulation_steps）
|metadata|	dict[str, torch.Tensor]|	附加信息，比如底层损失计算的中间结果或日志统计指标



实现提示
- 你需要在函数中调用：loss.backward()执行反向传播。
- 别忘了在反向传播前按梯度累积次数进行缩放：loss = loss / gradient_accumulation_steps
- 在计算损失时，需要：
    1.	通过掩码 response_mask 选出响应部分；
    2.	对这些部分的 log 概率取负（因为我们要最小化 -log p）；
    3.	使用 masked_normalize 来求加权平均或归一化。

测试方式

在实现完后，你需要在 adapters.py 中实现以下函数：

adapters.run_sft_microbatch_train_step

然后运行以下命令来测试：

uv run pytest -k test_sft_microbatch_train_step

确保测试全部通过




#### 日志记录模型生成结果（Logging generations in-the-loop）

在训练过程中进行“生成日志记录”是一种非常好的实践，这一点在 推理 SFT/RL（Reasoning SFT/RL） 阶段也不例外。

你需要编写一个函数 log_generations，让模型针对一些给定的 prompt（提示语） 生成响应，并记录相关信息。
这些 prompt 可以从验证集（validation set）中随机抽样获得。

你应该为每个样本至少记录以下内容：
1.	输入 prompt（模型生成的起始文本）
2.	模型生成的响应（response），来自 SFT 或 RL 阶段的模型
3.	真实答案（ground-truth answer）
4.	奖励信息（reward information）
	- 包括：
	- 格式奖励（format reward）
	- 答案奖励（answer reward）
	- 总奖励（total reward）
5.	响应的平均 token 熵（average token entropy）
	- 衡量模型生成时的不确定性
6.	响应长度相关统计信息：
	- 平均响应长度（average response length）
	- 正确响应的平均长度（average response length for correct responses）
	- 错误响应的平均长度（average response length for incorrect responses）


Problem: log_generations

交付内容

实现一个函数：
```
def log_generations(...):
    ...
```
该函数用于在训练过程中（例如每隔若干步）从模型中生成样本输出，并将生成结果及相关统计信息进行日志记录或打印，以便分析模型性能。



### 4.3 SFT 实验（Supervised Fine-Tuning Experiment）

使用你前面实现的各个部分，现在你需要完成整个 SFT（监督微调）流程（对应算法 1），
在 MATH 数据集 上对 Qwen 2.5 Math 1.5B Base 模型 进行微调。


#### 数据集说明
- 数据文件路径为：
/data/a5-alignment/MATH/sft.jsonl
- 每个样本都是一个 JSON 元素，格式如下：

{
    "prompt": "问题文本",
    "response": "模型的目标回答（包含推理过程和最终答案）"
}

也就是说，response 中不仅有最终答案，还包括 chain-of-thought（推理链）。


#### 训练与评估设置

为了在训练过程中跟踪模型性能，你需要定期在 MATH 验证集上进行评估。

你应该在 2 张 GPU 上运行你的脚本：
- 一张 GPU 用于 policy model（SFT 训练模型）；
- 另一张 GPU 用于 vLLM 实例 来进行 评估与推理（rollout）。


#### vLLM 初始化与权重加载

课程给出了一段辅助代码，用于：
1.	在另一张 GPU 上初始化一个 vLLM 推理进程；
2.	将当前 policy 模型的权重加载进该 vLLM 实例中。

示例代码如下
```
from vllm.model_executor import set_random_seed as vllm_set_random_seed

def init_vllm(model_id: str, device: str, seed: int, gpu_memory_utilization: float = 0.85):
    """
    启动 vLLM 推理进程（将其放到单独 GPU 上）
    """
    vllm_set_random_seed(seed)

    # 从 TRL（HuggingFace）Monkeypatch
    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    profiling_patch = patch(
        "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
        return_value=None
    )

    with world_size_patch, profiling_patch:
        return LLM(
            model=model_id,
            device=device,
            dtype=torch.bfloat16,
            enable_prefix_caching=True,
            gpu_memory_utilization=gpu_memory_utilization,
        )


def load_policy_into_vllm_instance(policy: PreTrainedModel, llm: LLM):
    """
    从 TRL 仓库复制的辅助函数：
    将 policy 模型权重加载到 vLLM 推理实例中
    """
    state_dict = policy.state_dict()
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())


```

#### 使用 wandb 记录指标

建议在训练和验证时同时记录指标，这样在后续 RL 实验中也能复用。

可在 wandb 中用以下方式定义指标轴：

**设置 wandb metric 轴**
wandb.define_metric("train_step")  # 训练步数 x 轴
wandb.define_metric("eval_step")   # 验证步数 x 轴

**所有以 train/ 开头的指标绑定到 train_step**
wandb.define_metric("train/*", step_metric="train_step")

**所有以 eval/ 开头的指标绑定到 eval_step**
wandb.define_metric("eval/*", step_metric="eval_step")



#### 梯度裁剪（Gradient Clipping）

为了稳定训练，建议在优化器更新时使用：

torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)


#### Problem: sft_experiment

##### 任务 1：不同数据规模下的 SFT 微调

在 Qwen 2.5 Math 1.5B base 模型 上进行 SFT 微调，
分别使用以下不同数量的样本子集：

{128, 256, 512, 1024} 以及全量数据集

你需要：
- 调整 学习率（learning rate） 和 batch size（批大小）
- 确保在使用全量数据集时，模型在验证集上达到至少 15% 的准确率

交付内容 1：
- 各数据集规模对应的验证准确率曲线（Validation accuracy curves）


##### 任务 2：基于正确样本的过滤实验

将推理 SFT 样本中过滤掉那些模型回答错误的样本，
仅保留能产出正确答案的训练样本。

然后在这个“过滤后的完整数据集”上再次运行 SFT。

交付内容 2：
- 过滤后数据集的大小（样本数）
- 在该数据集上训练后的验证准确率曲线（Validation accuracy curve）


##### 任务 3：结果比较

比较：
- 未过滤数据集上的 SFT 结果；
- 过滤后数据集上的 SFT 结果。

分析过滤是否带来了更好的泛化能力或稳定性。


## 5 MATH 数据集上的专家迭代（Expert Iteration）

在前一节中，我们观察到，通过从 SFT（监督微调）数据中筛除劣质样本，可以提升模型的性能。
在本节中，我们将更进一步：把这种筛选过程应用到**由基础模型自身生成的推理轨迹（reasoning traces）**上。
这种方法在文献中被称为 Expert Iteration（专家迭代） [Anthony et al., 2017]，并已在语言模型领域中得到探索
（例如 Cobbe et al. [2021b], Zelikman et al. [2022], Dohan et al. [2022], Gulcehre et al. [2023]）。


算法 2：专家迭代（EI）

输入：
- 初始策略模型 $\pi_{\theta_{\text{init}}}$
- 奖励函数 R
- 任务问题集 D


算法步骤：

1.	设策略模型 $\pi_\theta \leftarrow \pi_{\theta_{\text{init}}}$
2.	对于每一步 $ \text{step} = 1, …, n_{\text{ei-steps}} $：
3. 从任务集 D 中采样一个问题批次 $D_b$
4. 设旧策略模型 $\pi{\theta_{\text{old}}} \leftarrow \pi_\theta$
5. 对每个问题 $q \in D_b$，从旧模型中采样 G 个输出
$\{o^{(i)}\}_{i=1}^{G} \sim \pi_{\theta_{\text{old}}}(\cdot | q)$
6. 通过运行奖励函数 $R(q, o^{(i)})$，为每个生成的输出计算奖励 $r^{(i)}$
7. 筛除错误输出（即奖励 $r^{(i)} = 0 $的样本），得到一个正确问答对组成的数据集 $D_{\text{sft}}$
8. 使用监督微调（SFT，见算法 1）更新策略模型：
$\pi_\theta \leftarrow \text{SFT}(\pi_\theta, D_{\text{sft}})$
9. 重复以上步骤直到结束
输出最终模型$ \pi_\theta$



提示：
在使用 vLLM 生成样本时，应为 SamplingParams 设置 min_tokens 参数，以确保生成结果不为空字符串（否则在后续计算中可能导致 NaN）。
示例代码如下：
```
sampling_min_tokens = 4
sampling_params = SamplingParams(
    temperature=sampling_temperature,
    max_tokens=sampling_max_tokens,
    min_tokens=sampling_min_tokens,
    n=G,
    seed=seed,
)
```
与 SFT 训练相同，你应当使用 梯度裁剪（gradient clipping），裁剪值设为 1.0。

#### Problem:expert_iteration_experiment：
在 MATH 数据集上运行专家迭代实验（约 6 小时 H100 GPU 运行时间）

任务要求

在 MATH 数据集（路径：/data/a5-alignment/MATH/train.jsonl）上，
使用 Qwen 2.5 Math 1.5B Base 模型 运行专家迭代（Expert Iteration），并进行以下实验设置：
1.	超参数设置：
- 专家迭代轮数：n_ei_steps = 5
- 每轮的批量大小（即 $D_b$ 的大小）：从以下集合中选择若干进行实验
{512, 1024, 2048}
- 变动参数：
- 每个问题生成的样本数（rollouts） G
- SFT 阶段使用的训练轮数（epochs）
- 不需要尝试所有组合，只需足够进行比较和得出结论即可。
2.	采样设置：
- 使用 vLLM 生成时，应在遇到第二个 answer 标签时终止生成（与 SFT 阶段一致）。
- 在整个训练过程中记录模型输出的熵（entropy），以分析模型输出的不确定性变化。


需要提交的结果（Deliverables）
1.	不同 rollout 配置下的验证准确率曲线
- 至少尝试两种不同的 rollout 数量和 SFT epoch 数。
- 绘制验证集准确率随训练步骤变化的曲线。
2.	一个在 MATH 验证集上准确率 ≥ 15% 的模型。
3.	简短分析（约 2 句）
- 比较专家迭代（EI）与监督微调（SFT）的性能；
- 并描述模型在不同 EI 步骤间性能的变化趋势。
4.	模型响应熵随训练变化的曲线图
- 显示训练过程中模型输出分布的不确定性变化。


## 6 策略梯度（Policy Gradients）入门

语言模型研究中的一个令人兴奋的新发现是：
当在强大的基础模型上，对经过验证的奖励信号（verified rewards）执行强化学习（RL）训练时，
模型的推理能力与整体表现会显著提升【OpenAI et al., 2024；DeepSeek-AI et al., 2025】。

目前最强的开源推理模型，如 DeepSeek R1 与 Kimi k1.5【Team et al., 2025】，
都是通过一种称为**策略梯度(Policy Gradient)**的强化学习算法训练得到的。
这种算法能够优化任意形式的奖励函数，是强化学习中的核心方法之一。


我们在下面对语言模型中的策略梯度做一个简要介绍。
该内容主要参考了两份优秀的资料：
- OpenAI 的 Spinning Up in Deep RL【Achiam, 2018a】
- Nathan Lambert 的 Reinforcement Learning from Human Feedback (RLHF) Book【Lambert, 2024】


### 6.1 语言模型作为策略（Policies）

一个带参数 $\theta$ 的因果语言模型（Causal LM），定义了一个概率分布：
给定当前文本前缀 $s_t$（即状态或观察），生成下一个 token $a_t \in V$ 的概率为：

$$a_t \sim \pi_\theta(\cdot | s_t), \quad \pi_\theta(a_t | s_t) = \text{softmax}(f_\theta(s_t))_{a_t} \tag{3}$$

在强化学习的视角下：
- 当前文本前缀 $s_t$ 被看作状态（state）；
- 下一个 token $a_t$ 被看作动作（action）；
- 因此，语言模型实际上是一个离散随机策略（categorical stochastic policy）。

优化该策略时需要两个基本操作：
1.	从策略中采样动作：
从上面的分布中采样下一个 token $a_t$。
2.	计算动作的对数似然得分：
计算 $\log \pi_\theta(a_t | s_t)$，即生成该 token 的对数概率。


在语言模型强化学习中：
- $s_t$：当前已生成的部分回答（partial completion）；
- $a_t$：即将生成的下一个 token；
- 整个过程持续到生成结束标志，如 <|end_of_text|> 或在我们的 r1_zero 提示中为 answer。


### 6.2 轨迹（Trajectories）

一个**有限时域轨迹**（finite-horizon trajectory）是智能体经历的一系列状态与动作的交替序列：

$$\tau = (s_0, a_0, s_1, a_1, \ldots, s_T, a_T) \tag{4}$$

其中：
- T 为轨迹长度；
- 终止条件是：模型生成了结束标志 token（例如 answer）或达到了最大生成长度。


在轨迹开始时：
- 初始状态 $s_0$ 来自起始分布 $\rho_0(s_0)$；
- 在语言模型强化学习中，$\rho_0(s_0)$ 是一个**格式化提示**（formatted prompts）的分布。

在一般强化学习环境中：
$s_{t+1} \sim P(\cdot | s_t, a_t)$，其中 P 是环境的状态转移分布。

而在语言模型中：
- 环境是确定性的（deterministic）；
- 下一状态就是当前前缀拼接上生成的 token：
$s_{t+1} = s_t \Vert a_t$

轨迹也被称为 episodes（回合） 或 rollouts（采样生成），
在本课程或实验中，这些术语可以互换使用。


### 6.3 奖励与回报（Rewards and Return）

一个标量奖励 $r_t = R(s_t, a_t)$ 用于评估在状态 $s_t$ 下执行动作 $a_t$ 的即时质量。在**可验证领域**（verified domains）的强化学习中，通常的做法是：
- 对中间步骤给予 0 奖励；
- 对最终动作（即输出完整答案的那一步）给予验证奖励。

即：
$$r_T = R(s_T, a_T) :=
\begin{cases}
1, & \text{如果轨迹 } s_T \Vert a_T \text{ 与真实答案匹配（由奖励函数判断）} \\
0, & \text{否则}
\end{cases}
$$


轨迹的总回报 $R(\tau)$ 是对整条轨迹的奖励求和。常见的两种定义为：
1.	有限时域非折扣回报（finite-horizon undiscounted return）
$$R(\tau) = \sum_{t=0}^{T} r_t \tag{5}$$
2.	无限时域折扣回报（infinite-horizon discounted return）
$$R(\tau) = \sum_{t=0}^{\infty} \gamma^t r_t, \quad 0 < \gamma < 1 \tag{6}$$

在我们的场景中（语言模型生成任务），每个 episode 都有自然终点（例如生成 <|end_of_text|> 或达到最大长度），
因此我们使用非折扣形式（undiscounted formulation）。



智能体（语言模型）的目标是最大化期望回报（expected return）：

$$J(\theta) = \mathbb{E}{\tau \sim \pi _\theta}[R(\tau)] \tag{7}
$$
因此，优化目标为：

$$\theta^* = \arg \max_\theta J(\theta) \tag{8}
$$


### 6.4 朴素策略梯度（Vanilla Policy Gradient）

接下来，我们尝试通过**梯度上升**（gradient ascent）来优化策略参数$ \theta$，
以最大化期望回报：

$$\theta_{k+1} = \theta_k + \alpha \nabla_\theta J(\theta_k) \tag{9}
$$
其中 $\alpha$ 是学习率。


该方法的核心是著名的 REINFORCE 策略梯度公式（Policy Gradient Theorem）：

$$\nabla_\theta J(\pi_\theta)
= \mathbb{E}{\tau \sim \pi _\theta} \left[ \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t | s_t) \, R(\tau) \right] \tag{10}$$


策略梯度的推导

我们是如何得到上面的式子呢？下面给出一个推导的关键步骤。
1.	轨迹的概率分布
一条轨迹 $\tau$ 的概率为：
$$P(\tau | \theta)
= \rho_0(s_0) \prod_{t=0}^{T} P(s_{t+1} | s_t, a_t)\, \pi_\theta(a_t | s_t) \tag{11}$$
其中：
- $\rho_0(s_0)$：初始状态的分布；
- $P(s_{t+1} | s_t, a_t)$：环境的状态转移概率；
- $\pi_\theta(a_t | s_t)$：策略的动作分布。

对上式取对数，得到：
$$\log P(\tau | \theta)
= \log \rho_0(s_0) + \sum_{t=0}^{T} [\log P(s_{t+1} | s_t, a_t)+ \log \pi_\theta(a_t | s_t)] \tag{12}$$

2. 对数导数技巧（Log-derivative Trick）

$$
\nabla_\theta P = P \nabla_\theta \log P \tag{13}
$$


3. 环境项对 $\theta$ 是常数：

由于 $\rho_0$、$P(\cdot|\cdot)$ 和 $R(\tau)$ 都不依赖于策略参数 $\theta$，我们有：

$$\nabla_\theta \rho_0 = \nabla_\theta P = \nabla_\theta R(\tau) = 0 \tag{14}$$

基于上述事实，我们可以得到策略梯度的推导过程：

$$\nabla_\theta J(\theta) = \nabla_\theta \mathbb{E}{\tau \sim \pi _ \theta} [R(\tau)] \tag{15}$$
$$
= \nabla_\theta \left( \sum_\tau P(\tau | \theta) R(\tau) \right) \tag{16}$$


$$= \sum _ \tau \nabla_\theta P(\tau | \theta) R(\tau) \tag{17}
$$

$$= \sum_\tau P(\tau | \theta) \nabla_\theta \log P(\tau | \theta) R(\tau) \tag{18}$$


$$= \mathbb{E}{\tau \sim \pi _\theta} \left[ \nabla_\theta \log P(\tau | \theta) R(\tau) \right] \tag{17}$$

因此，结合轨迹的对数概率并使用环境项常数的事实，得到朴素的 REINFORCE 策略梯度：

$$\nabla_\theta J(\pi_\theta) = \mathbb{E}{\tau \sim \pi _ \theta} \left[ \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t | s_t) R(\tau) \right] \tag{20}$$


这个梯度会增加那些回报较高的轨迹中每个动作的对数概率，而减少回报较低的轨迹中动作的对数概率。



4. 梯度的样本估计：

给定一个由 N 次采样得到的 rollouts 批次 $D = \{ \tau^{(i)} \}^N_{i=1}$，其中每次采样都从起始状态 $s_0^{(i)} \sim \rho_0(s_0)$ 开始，并在环境中运行策略 $\pi_\theta$，我们可以得到梯度的无偏估计：

$$g = \frac{1}{N} \sum_{i=1}^{N} \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t^{(i)} | s_t^{(i)}) R(\tau^{(i)}) \tag{21}$$

这个梯度向量 g 会用于梯度上升更新：
$\theta \leftarrow \theta + \alpha g
$
，其中 $\alpha$ 是学习率。



### 6.5 策略梯度基线（Policy Gradient Baselines）

在最基本的策略梯度（vanilla policy gradient）算法中，一个主要问题是梯度估计的方差较高。
一种常见的缓解方法是从奖励中减去一个仅依赖于状态的基线函数（baseline function）。这属于一种控制变量（control variate） 技术 [Ross, 2022]，其核心思想是：
通过减去一个与估计量相关的项来降低方差，同时不引入偏差（bias）。

基线化的策略梯度定义如下：

$$B = \mathbb{E}{\tau \sim \pi _ \theta} \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t | s_t) [R(\tau) - b(s_t)] \tag{22}$$

举个例子，一个合理的基线选择是策略的价值函数（on-policy value function）：

$$V^\pi(s) = \mathbb{E}{\tau \sim \pi _ \theta} [R(\tau) | s_t = s]
$$
即：

当我们从状态 $s_t = s$ 开始并按照当前策略 $\pi_\theta$ 执行时，期望获得的回报。

那么，差值 $(R(\tau) - V^\pi(s_t))$ 就可以直观地理解为：
“该次轨迹的实际回报比期望值好多少（或差多少）”。

只要基线函数 仅依赖于状态（不依赖于动作），
则引入基线后的策略梯度仍然是无偏的（unbiased）。

我们可以这样推导：

$$B = \mathbb{E}{\tau \sim \pi_\theta} \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t | s_t) R(\tau)
-	\mathbb{E}{\tau \sim \pi_\theta} \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t | s_t) b(s_t) \tag{23}$$

现在我们专注于第二项（与基线相关的部分）：

$$\mathbb{E}{\tau \sim \pi_\theta} \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t | s_t) b(s_t)
= \sum_{t=0}^{T} \mathbb{E}{s_t}[b(s_t) \, \mathbb{E}{a_t \sim \pi_\theta(\cdot|s_t)}[\nabla_\theta \log \pi_\theta(a_t | s_t)]] \tag{24}$$

而一般来说，得分函数的期望为零：

$$\mathbb{E}{x \sim P\theta}[\nabla_\theta \log P_\theta(x)] = 0$$

因此，上式中的基线项为 0，得：

$$B = \mathbb{E}{\tau \sim \pi\theta} \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t | s_t) R(\tau) - 0 = \nabla_\theta J(\pi_\theta) \tag{25}$$

由此可以得出结论：
加入基线后的策略梯度依然是无偏估计。

稍后我们还会通过实验验证，引入基线是否能改善下游性能。


**关于策略梯度的 “损失函数”（Loss）说明**

在像 PyTorch 这样的框架中实现策略梯度方法时，
我们通常会定义一个所谓的 策略梯度损失（policy gradient loss） —— 记作 pg_loss。

其定义为：


$$pgloss = \frac{1}{N} \sum{i=1}^{N} \sum_{t=0}^{T} \log \pi_\theta(a_t^{(i)} | s_t^{(i)}) \, [R(\tau^{(i)}) - b(s_t^{(i)})] \tag{26}$$


当调用 pg_loss.backward() 时，
框架会自动通过反向传播，将近似的策略梯度 g 填充到模型参数的梯度缓冲区中。

换句话说，pg_loss 只是一个计算梯度的中间标量，
并不是一个可以衡量模型性能的“真正意义上的损失函数”。

注意
- 不应在训练集或验证集上报告 pg_loss 作为性能指标；
- 一个好的验证 pg_loss 值并不意味着模型泛化性良好；
- 在强化学习（RL）中，唯一有意义的评估指标是训练与验证的平均回报（reward）；
- 我们优化策略梯度方法的最终目标，就是最大化这些回报。

### 6.6 离策略（Off-Policy）策略梯度

REINFORCE 是一种在策略（on-policy）算法：
训练数据由我们正在优化的策略生成。我们可以这样写出 REINFORCE 算法的流程：
1.	采样阶段：从当前策略 πθ 中采样一批轨迹（rollouts） {τ(i)}ₙᵢ₌₁。
2.	梯度估计：近似计算策略梯度
$∇_θ J(π_θ) ≈ \hat{g} = \frac{1}{N} \sum_{i=1}^{N} \sum_{t=0}^{T} ∇_θ \log π_θ(a_t^{(i)} | s_t^{(i)}) R(τ^{(i)})$.

3.	参数更新：根据计算出的梯度更新策略参数
θ ← θ + αg.

在这种方法中，我们需要进行大量推理来采样新的轨迹批次，但每次仅执行一次梯度更新。
由于语言模型（LM）的行为在一次更新中通常不会发生显著变化，这种在策略方法效率极低。


**离策略策略梯度**（Off-policy Policy Gradient）

在离策略学习（off-policy learning）中，我们不再使用当前正在优化的策略采样轨迹，而是使用由其他策略生成的轨迹。

在像 PPO 和 GRPO 这样的常用策略梯度算法的离策略变体中，
轨迹由策略的先前版本$π_{θ_{old}}$生成，而我们优化的是当前策略 $π_θ$。

离策略策略梯度的估计公式如下：

$
g_{\text{off-policy}} = \frac{1}{N} \sum_{i=1}^{N} \sum_{t=0}^{T} ∇_θ \log π_θ(a_t^{(i)} | s_t^{(i)}) R(τ^{(i)}) \frac{π_θ(a_t^{(i)} | s_t^{(i)})}{π_{θ_{old}}(a_t^{(i)} | s_t^{(i)})} （27）$.


这看起来像是一个带有重要性采样（importance sampling）加权项的普通策略梯度公式，其中权重为
$\frac{π_θ(a_t^{(i)} | s_t^{(i)})}{π_{θ_{old}}(a_t^{(i)} | s_t^{(i)})}$.

实际上，公式 (27) 可以通过重要性采样原理推导得到。
只要 {πθ} 和 $π_{θ_{old}}$ 差异不大，这种近似就是合理的。
有关更详细的理论推导，可参考 Degris 等人（2013） 的研究。


## 7 群体相对策略优化（Group Relative Policy Optimization, GRPO）

接下来，我们将介绍群体相对策略优化（GRPO），这是一种策略梯度（policy gradient）的变体，你将实现并用它来解决数学问题。

### 7.1 GRPO 算法

优势估计（Advantage estimation）
GRPO 的核心思想是：对于每个问题，从当前策略 $π_θ$ 中采样多个输出，用这些输出来计算一个基线（baseline）。
这种方法的好处是，我们不需要训练神经价值函数 $Vϕ(s)$，因为价值函数往往难以训练，并且在系统层面上也比较繁琐。

设对于一个问题 $q$，从策略 $\pi_\theta$ 中采样得到一组输出：
$\{o^{(i)}\}_{i=1}^G \sim \pi_\theta(\cdot | q)$
每个输出 $o^{(i)} $的奖励为：
$r^{(i)} = R(q, o^{(i)})$

DeepSeekMath [Shao et al., 2024] 和 DeepSeek R1 [DeepSeek-AI et al., 2025] 使用如下公式计算群体归一化奖励（group-normalized reward）：
$A^{(i)} = \frac{r^{(i)} - \text{mean}(r^{(1)}, r^{(2)}, \ldots, r^{(G)})}{\text{std}(r^{(1)}, r^{(2)}, \ldots, r^{(G)}) + \text{advantage\_eps}}$
其中，advantage_eps 是一个防止除以零的小常数。

需要注意的是，该优势值 $A^{(i)}$ 对响应中的所有 token 都相同，即：
$A^{(i)}_t = A^{(i)}, \quad \forall t \in 1, \ldots, |o^{(i)}|
$因此，在后续推导中我们省略下标 $t$。


算法整体流程

在深入研究 GRPO 目标函数之前，我们先了解该算法的总体训练循环。
我们参考 Shao 等人（2024）的描述，将训练流程写成 算法 3 的形式。


GRPO 目标函数（GRPO Objective）

GRPO 的目标函数结合了三个关键思想：
1.	离策略的策略梯度（Off-policy policy gradient） —— 对应公式 (27)。
2.	基于群体归一化的优势估计（Advantage computed by group normalization） —— 对应公式 (28)。
3.	裁剪机制（Clipping mechanism） —— 借鉴自 Schulman 等人（2017）提出的 PPO（Proximal Policy Optimization）算法。


裁剪的目的在于：
当我们对同一批轨迹（rollouts）进行多次梯度更新时，防止策略 $πθ$ 偏离旧策略过远，从而保持训练的稳定性。


算法 3：群体相对策略优化（Group Relative Policy Optimization, GRPO）

输入：初始策略模型 $π_{θ_{init}}$；奖励函数 $R$；任务问题集 $D$

输出：最终策略模型 $π_θ$
1.	令策略模型 $π_θ ← π_{θ_init}$
2.	对每个步骤 step = 1, …, n_grpo_steps 执行：
3. 从任务集合 D 中采样一个批次问题 Db
4. 设旧的策略模型 $π_{θ_{old}} ← π_θ$
5. 对每个问题 $q ∈ D_b$，从旧策略 $π_{θ_{old}}$ 中采样 G 个输出 $\{o(i)\}_{i=1}^G ∼ π_{θ_{old}}(·|q)$
6. 使用奖励函数$ R(q, o(i))$ 计算每个采样输出$ o(i)$ 的奖励 $r(i)$
7. 使用群体归一化（公式 28）计算优势值 $A(i)$
8. 对每个训练步骤 train_step = 1, …, n_train_steps_per_rollout_batch 执行：
9. 通过最大化 GRPO-Clip 目标函数（公式 29）来更新策略模型 $π_θ$
10. 结束 for

输出最终策略模型 $π_θ$


接下来我们写出完整的 GRPO-Clip 目标函数，然后解释剪切（clipping）操作的作用：

$$J_{GRPO-Clip}(θ) =
E_{q∼D, {o^{(i)}}^G_{i=1} ∼ π_{θ_{old}}(·|q)}
\frac{1}{G} \sum_{i=1}^{G} \frac{1}{|o^{(i)}|} \sum_{t=1}^{|o^{(i)}|}
\min\left(
\frac{π_θ(o^{(i)}t | q, o^{(i)}{<t})}{π_{θ_{old}}(o^{(i)}t | q, o^{(i)}{<t})} A^{(i)},
\text{clip}\left(\frac{π_θ(o^{(i)}t | q, o^{(i)}{<t})}{π_{θ_{old}}(o^{(i)}t | q, o^{(i)}{<t})}, 1 - ε, 1 + ε\right) A^{(i)}
\right)
（29）$$

其中超参数 ε > 0 控制策略更新的幅度。为了更直观地理解这个机制，我们参考 Achiam [2018a,b] 定义一个函数：

$$g(ε, A^{(i)}) =
\begin{cases}
(1 + ε)A^{(i)}, & \text{如果 } A^{(i)} ≥ 0 \\
(1 - ε)A^{(i)}, & \text{如果 } A^{(i)} < 0
\end{cases}
（30）$$

于是，每个 token 的目标函数可以改写为：

$$\text{per-token objective} =
\min\left(
\frac{π_θ(o^{(i)}t | q, o^{(i)}{<t})}{π_{θ_{old}}(o^{(i)}t | q, o^{(i)}{<t})} A^{(i)},
g(ε, A^{(i)})
\right)$$


我们可以分情况进行推理：
- 当优势值 A(i) 为正时：
每个 token 的目标函数为
$$\text{per-token objective} =
\min\left(
\frac{π_θ(o^{(i)}t | q, o^{(i)}{<t})}{π_{θ_{old}}(o^{(i)}t | q, o^{(i)}{<t})} A^{(i)},
(1 + ε)A^{(i)}
\right)$$
由于 $A(i) > 0$，当新策略 $π_θ$ 使动作$ o(i)_t $的概率变大时（即 $π_θ(o(i)_t|…)$ 增加），目标值会上升。但剪切min操作会限制增长幅度：
当 $πθ(o(i)_t|q, o(i)<t) > (1 + ε)πθ_old(o(i)_t|q, o(i)<t)$ 时，目标值达到上限 $(1 + ε)A(i)$。
因此，策略 $πθ$ 不会被鼓励过度偏离旧策略$ π_{θ_{old}}$。
- 当优势值 A(i) 为负时：
模型会尝试降低 $π_θ(o(i)_t|q, o(i)<t)$，但剪切机制会阻止它降得过低。
当 $π_θ(o(i)_t|q, o(i)<t) < (1 − ε)πθ_old(o(i)_t|q, o(i)<t)$ 时，继续降低不会再提升目标值（可参见 Achiam [2018b] 的完整推导）。

好的，以下是 第 7.2 节 Implementation（实现） 的完整中文翻译，保持技术准确性和可读性：


### 7.2 实现

现在我们已经对 GRPO 的训练循环和目标函数有了较高层次的理解，接下来我们将开始实现其中的各个部分。
在 SFT（监督微调） 和 EI（Explicit Improvement） 部分中实现的许多组件也会被 GRPO 复用。


#### **计算优势值**（群体归一化奖励）

首先，我们将实现计算每个 rollout 批次样本优势值（advantage）的逻辑，也就是 群体归一化奖励（group-normalized rewards）。
我们将考虑两种不同的方式来获得群体归一化奖励：
1.	前文公式 (28) 所提出的方法；
2.	一种更简化的最新方法。

Dr. GRPO [Liu et al., 2025] 指出：
使用标准差 $std(r(1), r(2), …, r(G))$ 进行归一化的方式，
会使得在回答正确性差异较小的问题上（即所有答案都差不多正确或错误），该问题获得较高奖励，
这种现象可能并不理想。

因此，他们提出了一种简化版本：移除归一化步骤，直接计算：

$$A^{(i)} = r^{(i)} - \text{mean}(r^{(1)}, r^{(2)}, …, r^{(G)})
（31）$$


#### Problem（compute_group_normalized_rewards）：群体归一化

交付内容（Deliverable）：
实现一个名为 compute_group_normalized_rewards 的方法。
该方法应当为每个 rollout 响应（response）计算原始奖励（raw rewards），
并在每个组（group）内部进行归一化（normalization），
最后返回归一化后的奖励、原始奖励，以及你认为有用的任何元数据（metadata）。

推荐的函数接口：

```
def compute_group_normalized_rewards(
    reward_fn,
    rollout_responses,
    repeated_ground_truths,
    group_size,
    advantage_eps,
    normalize_by_std,
):
```

函数功能：

计算每组 rollout 响应的奖励值，并根据组大小（group size）进行归一化处理。



参数说明：
- reward_fn：
类型：Callable[[str, str], dict[str, float]]
一个可调用函数，用于根据真实答案（ground truth）评估 rollout 响应，
返回一个包含以下键的字典：
    - "reward"
    - "format_reward"
    - "answer_reward"

- rollout_responses：
类型：list[str]
策略模型生成的 rollout 响应列表。
其长度为
$rollout\_batch\_size = n\_prompts\_per\_rollout\_batch \times group\_size$
即每个问题生成 group_size 个响应。

- repeated_ground_truths：
类型：list[str]
每个示例对应的真实答案（ground truth）。
该列表的长度与 rollout_responses 相同，
因为每个真实答案会重复 group_size 次（每个组内的所有响应共享同一个 ground truth）。

- group_size：
类型：int
每个问题对应的响应数量（组大小）。

- advantage_eps：
类型：float
在标准差归一化中，为避免除以零而加入的微小常数。

- normalize_by_std：
类型：bool
若为 True，则使用组内标准差进行归一化；
若为 False，则仅减去组内均值（即不除以标准差）。


返回值（Returns）：

一个三元组（tuple）：

tuple[torch.Tensor, torch.Tensor, dict[str, float]]

包含以下内容：
- advantages：
形状为 (rollout_batch_size,) 的 torch.Tensor，
表示每个 rollout 响应的群体归一化奖励（group-normalized rewards）。
- raw_rewards：
形状为 (rollout_batch_size,) 的 torch.Tensor，
表示每个 rollout 响应的原始奖励（未归一化的 rewards）。
- metadata：
一个字典，包含可选的统计信息，
例如奖励的均值、标准差、最大值/最小值等，用于日志记录或调试。



测试方法：

实现函数 adapters.run_compute_group_normalized_rewards，
然后运行以下命令进行测试：
```
uv run pytest -k test_compute_group_normalized_rewards
```
确保你的实现通过测试。


