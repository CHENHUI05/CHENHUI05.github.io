---
title: 手撕迷你 GPT（逐行代码）：从"预测下一个字"到会写句子
date: 2026-10-23 20:15:00
tags:
  - 大模型基础
  - GPT
  - Transformer
  - PyTorch
  - 深度学习
categories:
  - 人工智能
description: 上一篇手撕了 Transformer，这一篇把它砍成 Decoder-only，从零写出一个 0.87M 参数的迷你 GPT。在 30 首唐诗上训练 3 分钟，它会写出完整的诗——但也会暴露一个所有大模型教程都回避的问题：它到底是在创作，还是在背诵？
---

## 写在前面

上一篇《手撕 Transformer》的结尾，我留了一句话：

> 进阶挑战：把 Encoder 去掉，只用 Decoder 实现一个迷你 GPT。

这一篇就来还债。

我们要从零写出一个 **0.87M 参数**的字符级 GPT：4 层、128 维、4 个注意力头、上下文 64 个字符。在一台普通 CPU 上训练 3 分钟，它就能写出这样的句子：

```
春眠不觉晓，处处闻啼鸟。
夜来风雨声，花落知多少。

登鹳雀楼
王之涣
白日依山尽，黄河入海流。
欲穷千里目，更上一层楼。
```

看起来很惊艳，对吧？

但我要先剧透：**它不是在写诗，它是在背诗。** 训练 loss 降到 0.03，验证 loss 却高达 12.9——这是教科书级的过拟合。本文会把这个"翻车现场"完整展示给你，因为**理解它为什么翻车，比看它写出诗更重要**。

这一篇你会学到：

- GPT 和上一篇的 Transformer 到底差在哪；
- "预测下一个字"如何变成训练信号；
- 因果自注意力、Pre-LN、残差、权重共享，每个零件为什么存在；
- 一个完整 GPT 的训练和生成代码；
- 为什么小数据上的"惊艳效果"往往是记忆，而不是智能。

完整可运行代码（含 30 首唐诗语料）在文末下载。这是「大模型基础」系列的第一篇，后面还会讲分词、采样、训练、上下文和 RAG。

<!-- more -->

## 一、GPT 就是把 Transformer 砍到只剩 Decoder

上一篇我们实现的是经典的 **Encoder-Decoder** 结构：编码器读输入，解码器生成输出，中间用交叉注意力连接。它适合翻译、摘要这类"输入一段、输出一段"的任务。

GPT 的选择更激进：**Encoder 不要了，交叉注意力也不要了，只留 Decoder。**

| 部件 | 原始 Transformer | GPT |
|---|---|---|
| Encoder | 有 | **没有** |
| Decoder | 有 | 有（就是整个模型） |
| 交叉注意力 | 有 | **没有** |
| 自注意力掩码 | 解码器带因果掩码 | 全程因果掩码 |
| 位置编码 | 正弦函数 | **可学习的位置嵌入** |
| 归一化位置 | Post-LN | **Pre-LN** |
| 前馈激活 | ReLU | **GELU** |
| 训练目标 | 序列到序列 | **预测下一个 token** |

为什么可以砍掉 Encoder？因为 GPT 解决的是一切都当成"续写"的问题：

```
输入：中国的首都是
输出：北京

输入：请把下面这句话翻译成英文：今天天气很好
输出：The weather is nice today.
```

翻译、问答、写代码，在 GPT 眼里都是"给一段前缀，续写后半段"。**结构越简单，越容易堆大、堆深、堆数据**——这就是 Decoder-only 赢下这一轮的原因。

整体结构：

{% mermaid %}
flowchart TD
    IDX["字符编号 (B, T)"] --> EMB["词嵌入 + 位置嵌入"]
    EMB --> B1["Block × 4<br/>LayerNorm → 因果自注意力 → 残差<br/>LayerNorm → 前馈网络 → 残差"]
    B1 --> LN["最终 LayerNorm"]
    LN --> HEAD["线性层（与词嵌入共享权重）"]
    HEAD --> LOGITS["每个位置对 544 个字符打分"]
    LOGITS --> LOSS["训练：和真实下一个字算交叉熵"]
    LOGITS --> SAMPLE["推理：取最后一个位置 → softmax → 采样"]
{% endmermaid %}

数据在模型里始终是 `(B, T, C)`：批次大小、序列长度、隐藏维度。本文 `B=32, T=64, C=128`。

## 二、核心思想：一切只是"预测下一个字"

### 2.1 训练样本是怎么造的

把整本诗集拼成一条长字符串：

```
床前明月光，疑是地上霜。举头望明月，低头思故乡。春眠不觉晓……
```

随机截取 64 个字符作为输入 `x`，把窗口向右移一位作为标签 `y`：

```
x: 床 前 明 月 光 ， 疑 是 地 上 霜 。 举 头 望 明 月
y: 前 明 月 光 ， 疑 是 地 上 霜 。 举 头 望 明 月 ，
```

模型在每个位置都要预测"下一个字"：

- 看到"床"，预测"前"；
- 看到"床前"，预测"明"；
- 看到"床前明月光"，预测"，"；
- ……

**一次前向传播，64 个位置同时得到训练信号**，不需要循环，GPU 可以吃满。这叫 **Teacher Forcing**：训练时永远拿"标准答案"当前缀，而不是拿模型自己上一步的输出。

### 2.2 损失函数

每个位置输出一个 544 维的向量（词表大小），过 softmax 变成概率，再和真实的下一个字符算交叉熵：

```
loss = -(1/T) Σ log P(真实字符 | 前面的字符)
```

整个语料的 loss 取平均，反向传播，更新参数。就这么简单。

### 2.3 为什么这能产生"智能"

因为要预测得准，模型必须学会：

- **语法**："疑是地上"后面接"霜"，不接"雪"；
- **事实**："白日依山尽"下一句是"黄河入海流"；
- **格式**：诗题、作者、正文、空行之间的排列规律；
- **风格**：五言和七言的节奏不同。

**预测下一个字，是一个"强迫模型理解数据规律"的任务。** 规律抓得越准，loss 越低。这也是为什么同一个目标能训练出会写代码、会做数学、会对话的模型——只是数据和规模不同。

但注意：它学的是"什么字最可能接在后面"，不是"什么话是真的"。这一点会在幻觉那篇展开。

## 三、分词：字符级，最简单也最诚实

本文用**字符级分词**：每个汉字、标点、换行符都是一个 token。

```python
chars = sorted(set(text))
stoi = {ch: i for i, ch in enumerate(chars)}   # 字符 -> 编号
itos = {i: ch for ch, i in stoi.items()}       # 编号 -> 字符
data = torch.tensor([stoi[c] for c in text])   # 整篇语料变成一条长整数序列
```

30 首唐诗共 1388 个字符，去重后词表大小 **544**。没有 `<PAD>`、`<BOS>`、`<EOS>`，因为语料是一整条连续文本，不需要补齐和分隔。

为什么不用 BPE？因为字符级最直观：每个 token 就是一个字，你能一眼看懂模型在干什么。真实 GPT 用的是字节级 BPE（几万到二十万词表），原理在 [Tokenizer 那篇](/2026/10/26/tokenizer-bpe-explained/) 里详细拆过。

字符级的代价是序列长、语义粒度细；但对一个 0.87M 参数的玩具模型来说，简单比先进重要。

## 四、零件一：因果自注意力

### 4.1 回顾 Q、K、V

每个 token 的向量会投影成三样东西：

- **Q（Query）**：我在找什么；
- **K（Key）**：我能被什么找到；
- **V（Value）**：我实际携带的信息。

注意力分数是 `Q·Kᵀ / √d`，softmax 后对 V 加权求和。上一篇已经用大量篇幅讲过，这里直接看代码。

### 4.2 代码

```python
class CausalSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)   # 一次算出 Q、K、V
        self.c_proj = nn.Linear(n_embd, n_embd)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        # 下三角矩阵：位置 t 只能看 <= t 的位置
        self.register_buffer(
            'mask',
            torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size)
        )

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(C, dim=2)
        d_head = C // n_head
        q = q.view(B, T, n_head, d_head).transpose(1, 2)   # (B, h, T, d_head)
        k = k.view(B, T, n_head, d_head).transpose(1, 2)
        v = v.view(B, T, n_head, d_head).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(d_head)  # (B, h, T, T)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))
```

### 4.3 掩码为什么必须是下三角

训练时，整个 64 个字符是同时喂进去的。如果没有掩码，位置 5 在预测位置 6 时，就能"看到"位置 6 的真实答案——这是作弊，模型学不到任何东西。

下三角掩码把"未来"的位置分数设成 `-inf`，softmax 后权重为 0：

```
位置:   0  1  2  3
0      ✓  ✗  ✗  ✗
1      ✓  ✓  ✗  ✗
2      ✓  ✓  ✓  ✗
3      ✓  ✓  ✓  ✓
```

这样每个位置只能看自己和左边。训练时并行，推理时自回归，两种模式共享同一套参数。

注意：注意力的计算量是 `O(T²)`，这是长上下文昂贵的根源，后面 KV Cache 那篇会算这笔账。

## 五、零件二：Pre-LN、残差与前馈网络

一个 Block 长这样：

```python
class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention()
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),   # 128 -> 512
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),   # 512 -> 128
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))   # 先归一化，再过注意力，最后加回残差
        x = x + self.mlp(self.ln2(x))
        return x
```

三个细节：

1. **Pre-LN**：归一化放在子层**前面**，残差主干是一条"高速公路"，梯度可以从最后一层直接流到第一层。原始 Transformer 的 Post-LN 在深层容易不稳定，GPT-2 之后普遍改成 Pre-LN。
2. **残差连接**：每个子层只学"增量"，不负责重建整个表示。没有残差，深层网络很难训。
3. **FFN 先扩 4 倍再压回**：给模型一个"逐位置做非线性变换"的空间。注意力负责"token 之间交流"，FFN 负责"每个 token 自己消化"。

GELU 是 ReLU 的平滑版本，在 0 附近不是硬拐点，训练更稳，是 GPT 系列的标配。

## 六、零件三：可学习的位置嵌入

注意力本身不知道顺序。把句子打乱，`Q·Kᵀ` 的结果只是换了个排列。所以必须告诉模型"谁在第几位"。

本文用最简单的方案：**给每个位置学一个向量，加到 token 向量上。**

```python
self.pos_emb = nn.Embedding(block_size, n_embd)   # 64 个位置，每个 128 维

pos = torch.arange(T, device=idx.device)
x = self.tok_emb(idx) + self.pos_emb(pos)
```

`block_size=64` 意味着模型最多只能看 64 个字符的上下文。生成时如果序列超过 64，就要把最前面的截掉（第七节会看到）。

现代大模型（LLaMA、Qwen）普遍改用 **RoPE（旋转位置编码）**，能更好地表达相对位置、支持更长上下文。但"给模型注入位置信息"这个核心思想是一样的。

## 七、拼装：MiniGPT

把零件拼起来：

```python
class MiniGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block() for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight   # 权重共享
        self.apply(self._init_weights)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)                    # (B, T, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
        return logits, loss
```

### 7.1 权重共享（weight tying）

输入嵌入矩阵是 `(vocab, n_embd)`，输出层也是 `(n_embd, vocab)`。它们做的是互为逆运算的事：一个把编号变成向量，一个把向量变回编号。

**让它们共享同一份参数**，既省参数，又能提升效果。GPT-2 就是这么做的，很多现代模型也沿用了这个设计。

### 7.2 初始化

```python
nn.init.normal_(module.weight, mean=0.0, std=0.02)
nn.init.zeros_(module.bias)
# 残差投影再缩小一点，避免深层残差累积过大
nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * n_layer))
```

初始化看起来是小细节，但深层 Transformer 对初值很敏感。GPT-2 论文专门讲了残差缩放，本文照抄。

### 7.3 参数量

| 部分 | 参数量 |
|---|---|
| 字符嵌入 544×128 | 69,632 |
| 位置嵌入 64×128 | 8,192 |
| 每个 Block（注意力 + FFN + LN） | 约 198,000 |
| 4 个 Block | 约 793,000 |
| 最终 LayerNorm | 256 |
| **合计** | **约 0.87M** |

和真正的 GPT 比一下：

| | 本文 MiniGPT | GPT-2 Small | GPT-3 175B |
|---|---|---|---|
| 层数 | 4 | 12 | 96 |
| 隐藏维度 | 128 | 768 | 12,288 |
| 注意力头 | 4 | 12 | 96 |
| 上下文 | 64 | 1,024 | 2,048 |
| 词表 | 544 | 50,257 | 50,257 |
| 参数量 | 0.87M | 124M | 175B |

架构是同一个家族，差的是规模、数据和工程。

## 八、训练：3 分钟，3000 步

### 8.1 数据划分与取样

```python
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]   # 前 90% 训练，后 10% 验证

def get_batch(split):
    d = train_data if split == 'train' else val_data
    ix = torch.randint(len(d) - block_size - 1, (batch_size,))
    x = torch.stack([d[i:i+block_size] for i in ix])
    y = torch.stack([d[i+1:i+1+block_size] for i in ix])
    return x, y
```

验证集取语料最后 10%，和训练集不重叠。注意这是**连续切分**，不是随机打乱——因为语料是一整条诗，随机打乱会破坏上下文。

### 8.2 优化器与学习率

```python
optimizer = torch.optim.AdamW(
    model.parameters(), lr=3e-3, betas=(0.9, 0.99), weight_decay=0.01
)
```

- **AdamW**：Adam + 解耦权重衰减，是训练 Transformer 的默认选择；
- **学习率 3e-3**：对 0.87M 的小模型算比较大，因为模型小、数据少；
- **warmup 100 步 + 余弦退火**：先热身，再慢慢降，避免一开始就把参数带偏；
- **梯度裁剪 1.0**：防止个别 batch 梯度爆炸。

### 8.3 训练循环

```python
for step in range(3000):
    x, y = get_batch('train')
    logits, loss = model(x, y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    scheduler.step()
```

每 250 步在训练集和验证集上各估一次 loss（各 20 个 batch 取平均）。

### 8.4 真实训练曲线

在 CPU 上跑 3000 步，约 3 分 20 秒，输出：

```
step     0 | train 6.314 | val 6.309 |    0.8s
step   250 | train 0.073 | val 9.751 |   17.5s
step   500 | train 0.053 | val 11.203 |   34.1s
step   750 | train 0.045 | val 11.845 |   50.6s
step  1000 | train 0.041 | val 12.786 |   67.1s
step  1250 | train 0.043 | val 13.005 |   84.5s
step  1500 | train 0.033 | val 13.273 |  101.1s
step  1750 | train 0.036 | val 12.779 |  117.9s
step  2000 | train 0.036 | val 12.628 |  134.5s
step  2250 | train 0.033 | val 12.370 |  151.1s
step  2500 | train 0.031 | val 12.808 |  167.7s
step  2750 | train 0.029 | val 12.997 |  184.3s
step  3000 | train 0.029 | val 12.938 |  200.9s
```

初始 loss 约 6.31，接近 `ln(544) ≈ 6.30`——相当于在 544 个字符里均匀瞎猜，符合预期。

训练 loss 一路降到 0.03，看起来完美。但**验证 loss 从 6.3 一路涨到 12.9**，比随机猜还差。这是非常典型的过拟合信号。

先别急，我们看看它生成的效果，再解释这个现象。

## 九、生成：一个字一个字往外蹦

### 9.1 代码

```python
@torch.no_grad()
def generate(model, idx, max_new_tokens, temperature=0.8, top_k=20):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -block_size:]        # 只保留最近 64 个字符
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / max(temperature, 1e-8)   # 只取最后一个位置
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('inf')
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx
```

关键点：

- **只取最后一个位置**：前面位置是在训练时算 loss 用的，生成时只需要"下一个字"的分布；
- **截断到 64**：位置嵌入只有 64 个，超了会越界；真实模型用滑动窗口或 RoPE 处理更长上下文；
- **temperature / top-k**：控制随机性，后面的采样篇会专门讲。

### 9.2 真实生成结果

用不同的开头提示，采样 80 个字符（temperature=0.8, top-k=20）：

```
提示词「春眠」->
春眠不觉晓，处处闻啼鸟。
夜来风雨声，花落知多少。

登鹳雀楼
王之涣
白日依山尽，黄河入海流。
欲穷千里目，更上一层楼。

相思
王维
红豆生南国，春来发几枝。


提示词「明月」->
明月，低头思故乡。

春晓
孟浩然
春眠不觉晓，处处闻啼鸟。
夜来风雨声，花落知多少。

登鹳雀楼
王之涣
白日依山尽，黄河入海流。
欲穷千里目，更上一层楼。

相


提示词「大漠」->
大漠孤烟直，长河落日圆。
萧关逢候骑，都护在燕然。

送元二使安西
王维
渭城朝雨浥轻尘，客舍青青柳色新。
劝君更尽一杯酒，西出阳关无故人。

九月九日忆山东兄弟
```

贪心解码（几乎不吃随机性）的结果也差不多：

```
春眠不觉晓，处处闻啼鸟。
夜来风雨声，花落知多少。

登鹳雀楼
王之涣
白日依山尽，黄河入海流。
欲穷千里目，更上一层楼。
```

### 9.3 看起来很美，但…

你会发现：**它写出的每一句，都是语料里原封不动的句子。**

「春眠」后面接《春晓》，再接下一位诗人、下一首诗；「大漠」后面接《使至塞上》，再接《送元二使安西》。连标题、作者、空行、标点都一模一样。

这不是"学会了写诗"，这是"背下了整本诗集"。

## 十、它到底是在创作，还是在背诵？

### 10.1 三个数字拆穿它

| 指标 | 数值 | 含义 |
|---|---|---|
| 训练 loss | 0.029 | 在训练集上，几乎完美预测下一个字 |
| 验证 loss | 12.938 | 在没见过的诗句上，错得离谱 |
| 训练轮数 | 约 4,400 遍 | 1388 个字符，被反复看了几千遍 |

算一下：

- 训练 loss 0.029 对应困惑度 `e^0.029 ≈ 1.03`，相当于每一步都在 1 个候选里选，完全没有不确定性；
- 验证 loss 12.938 对应困惑度 `e^12.938 ≈ 41 万`，比在 544 个字符里均匀瞎猜（困惑度 544）还差 700 多倍。

为什么验证 loss 会比随机猜还差？因为模型在训练集上变得**过度自信**：它把训练集的每个字符序列都记死了，遇到没见过的诗句时，仍然用极高的置信度预测"训练集里的下一句"，结果全错。**自信地错，比均匀地猜，交叉熵更大。**

### 10.2 为什么会这样

一句话：**模型容量远大于数据量。**

- 0.87M 参数，1388 个字符的语料；
- 3000 步 × 32 批 × 64 字符 = 614 万次预测；
- 相当于把 1388 个字符反复看了约 4400 遍。

一个能记住 87 万个数的小模型，面对 1388 个字符，最省力的策略就是**死记硬背**，而不是"理解诗歌的规律"。梯度下降不会区分"理解"和"记忆"，它只负责把 loss 压低。

这也是为什么很多"小数据 + 小模型"的 demo 看起来很惊艳：它们展示的往往是记忆，不是泛化。**训练 loss 低不代表模型聪明，验证 loss 才是照妖镜。**

### 10.3 怎么让它真的"学会写"

想让验证 loss 也降下来，需要同时做几件事：

1. **加数据**：从几千字加到几 MB、几十 MB，让模型没法靠背；
2. **加正则**：提高 dropout、加权重衰减、早停（验证 loss 开始上升就停）；
3. **匹配容量**：数据少的时候，模型也要小；数据多了，再加大模型；
4. **换分词**：字符级序列长、语义碎，换成 BPE 能提升效率；
5. **加训练技巧**：学习率调度、梯度累积、混合精度。

但即便都做了，0.87M 参数的模型能力也有上限。GPT-3 的参数是它的 **20 万倍**，训练 token 是它的**几亿倍**。**架构只是骨架，规模才是血肉。**

### 10.4 那这个实验还有意义吗？

有，而且很大。

它让你亲眼看到：

- 一个完整的 GPT 前向、训练、生成流程，到底长什么样；
- 每个零件（掩码、残差、LN、位置嵌入、权重共享）在代码里的位置；
- loss 曲线的形状意味着什么；
- **过拟合不是抽象概念，而是"训练 loss 0.03、验证 loss 12.9"的具体数字**；
- 为什么大模型需要海量数据和算力——不是厂商想烧钱，是这条路只有这个走法。

看懂了这个 0.87M 的模型，你再看 GPT-4、DeepSeek、Qwen，就不会觉得它们是魔法了。它们只是把这个模型放大了几万倍，换了更好的分词、位置编码、注意力实现和训练数据。

## 十一、常见疑问

**Q1：为什么不用 `nn.TransformerDecoder`？**
因为我们要看清每个零件。PyTorch 的封装把掩码、多头、残差都藏起来了，调包能跑，但学不到东西。手写一遍之后，再用封装就是查文档的事。

**Q2：为什么用字符级，不用 BPE？**
字符级最直观，词表小，代码短。真实模型必须用 BPE，否则序列太长、词表不可控。替换方法很简单：把 `stoi/itos` 换成 tokenizer 的 `encode/decode`，其余代码不变。

**Q3：CPU 真的能跑吗？**
能。本文 3000 步在 CPU 上约 3 分 20 秒，内存占用不到 1GB。有 GPU 会更快，但这个规模没必要。

**Q4：它能对话吗？**
不能。它是一个**基座模型**，只会续写。你问"你好"，它可能继续写"你好，请介绍一下你自己……"。要让它听指令，需要 SFT 和对齐，见 [训练那篇](/2026/10/30/llm-training-pipeline/)。

**Q5：为什么验证 loss 比随机还高？**
因为过拟合导致过度自信。模型在没见过的数据上给出极端错误的概率分布，交叉熵惩罚很重。这也说明：**不要只看训练 loss，验证集才是判断泛化的依据。**

**Q6：怎么改成词级/BPE？**
把文本先 tokenize 成 id 序列，词表换成 tokenizer 的词表，`block_size` 可以适当加大。其他部分（注意力、Block、训练循环）完全不用改。

**Q7：这个模型和 GPT-2 差在哪？**
架构几乎一样，差在规模（0.87M vs 124M）、数据（1.4KB vs 40GB）、分词（字符 vs BPE）、位置编码（可学习 vs 可学习）、以及工程优化（FlashAttention、混合精度等）。GPT-2 也是 Decoder-only + Pre-LN + GELU + 权重共享。

## 十二、总结

把整个模型压缩成一条线：

```
字符 → 编号 → 词嵌入 + 位置嵌入
     → 4 × [LayerNorm → 因果自注意力 → 残差 → LayerNorm → FFN → 残差]
     → 最终 LayerNorm
     → 线性层（共享词嵌入权重）
     → 每个位置对 544 个字符打分
     → 训练：交叉熵；推理：softmax + 采样
```

记住五句话：

1. **GPT = Decoder-only Transformer + 预测下一个 token**，没有 Encoder，没有交叉注意力。
2. **因果掩码是训练和推理共享同一套参数的关键**：训练时并行，但不能偷看未来。
3. **Pre-LN + 残差 + GELU + 权重共享**，是现代 GPT 的四个标准件。
4. **小数据 + 小模型 = 记忆**；训练 loss 低不代表会泛化，验证 loss 才是照妖镜。
5. **规模是能力的来源**，架构只是让规模能堆得起来。

动手建议：

1. 下载文末代码，跑通训练，亲眼看 loss 曲线；
2. 把语料换成你自己的文本（小说、歌词、代码），观察生成风格；
3. 把 `dropout` 从 0.1 调到 0.3，或者把训练步数减到 500，看验证 loss 是否改善；
4. 把 `n_layer` 从 4 改成 8，观察参数量和过拟合速度；
5. 把提示词设成训练集里没有的句子，看它怎么"露馅"。

下一篇我们聊 **Tokenizer**：为什么模型数不清 `strawberry` 里有几个 `r`？为什么中文比英文更费 token？我们会用一百来行纯 Python 手撕 BPE，把"文字变 token"这件事彻底讲透。

**完整代码下载**：[mini_gpt.py](/downloads/code/mini_gpt.py)（含 30 首唐诗语料，直接 `python mini_gpt.py` 即可运行；也可以把任意 UTF-8 文本保存为 `corpus.txt` 放在同目录，脚本会优先读取它。）

---

**本系列其他文章：**

- {% post_link tokenizer-bpe-explained '大模型眼里的文字：手撕 Tokenizer 与 BPE 分词' %}
- {% post_link llm-sampling-decoding '温度、Top-k 与 Top-p：大模型是怎么"挑词"的' %}
- {% post_link llm-training-pipeline '大模型是怎么炼成的：预训练、SFT 与对齐' %}
- {% post_link context-window-kv-cache '上下文窗口与 KV Cache：大模型的记忆为什么又贵又短' %}
- {% post_link llm-hallucination-rag '大模型为什么会胡说八道：幻觉与 RAG' %}
