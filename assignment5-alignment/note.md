# Supervised Finetuning for MATH

这个部分需要我们从零搭建一个监督微调框架。

测试之前需要修改测试文件`conftest.py`中213行
```
@pytest.fixture
def model_id():
    return "Qwen/Qwen2.5-Math-1.5B"

```
这样会从huggingface上下载模型
## 监督微调算法：

输入：初始策略模型$ \pi_{\theta_\text{init}}$；SFT 数据集 D
1. 将策略模型$ \pi_\theta \gets \pi_{\theta_\text{init}}$
2. for step = 1, …, n_sft_steps do：
3. 从数据集 D 中抽取一个问题-回答对批次 D_b
4. 使用模型$ \pi_\theta $计算回答相对于问题的 交叉熵损失（cross-entropy loss）
5. 对模型参数$ \theta $进行梯度更新（gradient step）
6. 结束循环

输出：微调后的模型 $\pi_\theta$


第一步初始化
- 首先需要加载数据集，每条数据包括prompt（即问题）和response，response是由推理链和问题答案构成的。

第二步
- 从数据集中加载一个批次的数据计算交叉熵。

## 组件（需要实现的方法）

### tokenize_prompt_and_output
这个函数是用来对prompt和output分词的。prompt表示问题，output表示模型的输出，即问题的答案。首先用tokenizer对它们进行分词，tokenizer的返回结果是包含input_ids等的字典，input_ids表示输入（prompt）的分词结果在词表（vocab）的id。然后需要将prompt和output的input_ids合并起来，并据此得到掩码（遮盖问题，保留回答），我们需要的掩码应当遮盖问题（prompt）。此时的full_input和mask还是列表，需要转化为tensor。

这里的batch size就是input_id_list的长度，表示有多少条输入的句子。然后我们需要取出input_ids中取出最长的作为max_len，用来构建tensor的形状(batch_size, max_len)。
对于input_ids_batch，我们用pad来填充tensor，mask_batch用0来填充。然后再把之前的到的input_id和response_mask填入这两个创建的张量。因为我们是根据前一个token预测下一个token，所以lable就是input向后平移一位。

### compute_entropy
简单介绍一下交叉熵

假设有一个样本的真实标签分布 y 和模型预测概率分布 p：
$$
H(y, p) = - \sum_{i} y_i \log p_i
$$
- y_i 是真实分布的第 i 个类别的概率（one-hot 编码中，正确类别为 1，其他为 0）
- p_i 是模型预测该类别的概率（softmax 输出）
- 求和是对所有类别 i 做的
假设
- 有一个分类问题，类别数 C = 3
- 真实标签是类别 2（用 one-hot 表示）：y = [0, 1, 0]
- 模型预测的概率分布（softmax 输出）是：p = [0.2, 0.7, 0.1]
套公式计算
1. i=1： y_1 = 0，所以 $ 0 \cdot \log 0.2 = 0$
2. i=2： y_2 = 1，所以 $ 1 \cdot \log 0.7 = \log 0.7 \approx -0.3567$
3. i=3： y_3 = 0，所以 $ 0 \cdot \log 0.1 = 0$

把负号加上：

H(y, p) = -(-0.3567) = 0.3567

这个损失相对较小，说明我们的预测还算准确。

在强化学习（RL）训练中，记录模型每个 token 的预测熵是非常有用的。
这样可以帮助我们观察模型的预测分布是否变得“过于自信”（即概率分布过于尖锐）。
通过交叉熵，我们类比到这个作业中用到的（经典）熵
$$
H(p) = - \sum_{i} p_i \log p_i
$$
与交叉熵相比，它没有真实标签$y_i$，用于衡量一个分布本身的不确定性。

因为
$p_i = \frac{e^{z_i}}{\sum_{j} e^{z_j}}$
这里有一个 指数运算 $e^{z_i}$，
如果 $z_i$ 很大（比如 1000），$e^{z_i}$ 会 溢出，变成 inf，如果 $z_i$ 很小（比如 -1000），$e^{z_i}$ 会接近 0，$\log(0)$ 会导致 -inf
因此我们用logsumexp来优化计算。

$$
p_i = \frac{e^{z_i}}{\sum_{j=1}^C e^{z_j}} \\

H(\mathbf{p}) = - \sum_{i=1}^C p_i \log p_i 
= - \sum_{i=1}^C p_i \log \frac{e^{z_i}}{\sum_{j=1}^C e^{z_j}} \\
\log \frac{e^{z_i}}{\sum_{j=1}^C e^{z_j}} = \log e^{z_i} - \log \sum_{j=1}^C e^{z_j} = z_i - \log \sum_{j=1}^C e^{z_j} \\
H(\mathbf{p}) = - \sum_{i=1}^C p_i (z_i - \log \sum_{j=1}^C e^{z_j}) \\

H(\mathbf{p}) = - \sum_{i=1}^C p_i z_i + \sum_{i=1}^C p_i \log \sum_{j=1}^C e^{z_j} \\
\sum_{i=1}^C p_i = 1 \\


所以：H(\mathbf{p}) = \log \sum_{i=1}^C e^{z_i} - \sum_{i=1}^C p_i z_i
$$

回到我们的代码，logits是模型输出的预测分词未归一化的分数，形状一般为(batch_size, seq_len, vocab_size)，表示batch_size条句子，seq_len长度各个分词的分数。然后我们在最后一个维度（vocab_size）进行熵的计算。

### get_response_log_probs
这个函数可以得到对labels对预测概率。

根据对数概率公式
$$
\log p_i = \log \frac{e^{z_i}}{\sum_j e^{z_j}}
$$
得到每个token的对数概率，log_probs的形状是(batch_size, seq_len, vocab_size)，lables的形状是(batch_size, seq_len)，为了得到lables每个位置token的对数概率，我们对它再增加一个维度(batch_size, seq_len, 1)，这样张量的每个数字就表示某句某个token的id了，从而从log_probs对应过去相应token的对数概率。最后再压缩张量的最后一维，得到(batch_size, seq_len)，这个张量的每个数字表示模型在当前batch中的第b个样本、第t个token位置上对正确标签token的预测对数概率。

### masked_normalize
这个函数用来对每个样本的有效token做掩码平均。

首先把张量点乘上掩码，这样就可以保留需要的损失。接着默认是对所有维度求平均，然后除以一个归一化常数。

### sft_microbatch_train_step
为了使用更大的batch_size，模型采用了梯度累积的方式。通常每个bacth计算完梯度后就更新权重，梯度累积则是在多个batch上累积梯度，再进行一次梯度更新。
## wandb

Weights & Biases（简称 wandb） 是一个常用的机器学习实验跟踪平台。用来记录各项指标并可以绘制相关图像

|功能|	说明|
|-|-|
|实时记录|	训练 loss、验证 accuracy、学习率、显存等指标|
|可视化	|自动生成曲线图（比如 loss 随 step 下降）
|保存模型|	自动上传和版本化模型 checkpoint

在训练过程中，我们会有不同的指标：
- train/loss（训练损失）
- train/accuracy（训练准确率）
- eval/loss（验证损失）
- eval/accuracy（验证准确率）

这些指标都需要一个“横轴（x 轴）” 来表示它们随什么变化。
最常见的横轴是 训练步数（train_step） 或 验证步数（eval_step）。
```
wandb.define_metric("train_step")
wandb.define_metric("eval_step")
```
这两行定义了两个「主横轴」：
- train_step：训练阶段的步数（通常每一次参数更新算一步）
- eval_step：验证阶段的步数（每次验证算一步）

```
wandb.define_metric("train/*", step_metric="train_step")
```
这表示：

所有以 "train/" 开头的指标，都要以 train_step 作为横轴。

举个例子：
```
wandb.log({"train/loss": loss, "train/accuracy": acc, "train_step": step})
```
wandb 会自动画出：
- x 轴：train_step
- y 轴：train/loss 和 train/accuracy

eg.
```
import wandb

wandb.init(project="sft-math", name="qwen-1.5b-sft")

# 定义横轴
wandb.define_metric("train_step")
wandb.define_metric("eval_step")

# 绑定关系
wandb.define_metric("train/*", step_metric="train_step")
wandb.define_metric("eval/*", step_metric="eval_step")

# 在训练时记录
for step in range(1000):
    loss = train_step(...)
    wandb.log({"train/loss": loss, "train_step": step})

# 在验证时记录
for eval_step in range(10):
    eval_acc = validate(...)
    wandb.log({"eval/accuracy": eval_acc, "eval_step": eval_step})
```


## GRPO

**rollout**

在强化学习中，rollout 指的是：
从环境（environment）中采样得到的一条完整轨迹（trajectory）：
$$\tau = (s_0, a_0, r_0, s_1, a_1, r_1, …, s_T)
$$
### compute_group_normalized_rewards
每个问题会生成多(group_size)个回答，我们需要在同一问题组内计算奖励的相对好坏，这称为群体归一化。

常见做法有两种：
1.	标准化（normalize_by_std=True）
$$A_i = \frac{r_i - \text{mean}(r_1, \dots, r_G)}{\text{std}(r_1, \dots, r_G) + \epsilon}$$
2.	仅减去均值（normalize_by_std=False）
$$A_i = r_i - \text{mean}(r_1, \dots, r_G)
$$