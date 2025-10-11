以下是该作业说明的中文翻译：

---

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

以下是该部分的中文翻译：

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

---

