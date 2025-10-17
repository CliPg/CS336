# Supervised Finetuning for MATH

这个部分需要我们从零搭建一个监督微调框架。
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
