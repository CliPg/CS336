# 1 Assignment Overview

## 1.1 Profiling and Benchmarking

### 1.1.1 Setup - Importing your Basics Transformer Model

### 1.1.2 Model Sizing

### 1.1.3 End-to-End Benchmarking

#### Problem(benchmarking_script)

编写一个脚本，对你的模型执行基础的端到端前向与反向传播的性能基准测试（benchmark）。具体来说，该脚本需要支持以下功能：
- 根据给定的超参数（例如层数）初始化一个模型。
- 生成一批随机数据作为输入。
- 先运行 w 次预热步骤（warm-up steps）（这些步骤不计入计时），然后计时执行 n 次（根据参数选择只做前向，或同时做前向和反向传播）。
对于计时，你可以使用 Python 的 timeit 模块（例如使用 timeit 函数，或使用 timeit.default_timer()，它提供系统中分辨率最高的时钟，因此比 time.time() 更适合作基准测试）。
- 在每一步执行后调用 torch.cuda.synchronize()。

交付物（Deliverable）：
编写一个脚本，根据给定超参数初始化一个基础版本的 Transformer 模型，创建一批随机数据，并对前向和反向传播进行计时。