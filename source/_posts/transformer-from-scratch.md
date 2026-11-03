---
title: 手撕 Transformer（逐行代码）：零基础也能看懂的完整实现
date: 2026-10-19 19:40:49
tags:
  - Transformer
  - 深度学习
  - NLP
  - PyTorch
  - 注意力机制
categories:
  - 人工智能
description: 不调用任何现成封装，从零逐行写出完整 Transformer，并在数字串反转任务上训练成功。用大量比喻、流程图和真实运行结果，讲透词嵌入、位置编码、QKV 注意力、多头、掩码、残差与 Encoder-Decoder。
---

## 写在前面

今天的大模型——ChatGPT、Gemini、DeepSeek、各种翻译软件——它们的骨架里都有同一个名字：**Transformer**。2017 年谷歌发表论文《Attention Is All You Need》之后，它几乎成了现代深度学习的通用结构。

很多人觉得 Transformer 玄乎其玄，Q、K、V、多头、掩码一堆名词看了就头大。但只要你把它**亲手敲一遍**，就会发现它每一个零件都在解决一个具体的问题，逻辑非常朴素。

这篇文章带你做三件事：

1. 用大白话讲清楚 Transformer 的**每一个零件**是干什么的；
2. **不调用任何现成的 Transformer 封装**，只用 PyTorch 最基础的线性层，从零逐行写出完整模型；
3. 训练它学会一个小任务——**把数字串反转**（输入 `314159`，输出 `951413`），亲眼看到 loss 下降、准确率上升。

**前置知识几乎为零**：你只需要知道"向量是一串数字""矩阵乘法是把表格按规则相乘"就够了。看不懂的地方先跳过，把代码跑起来，再回头看概念，会豁然开朗。

运行环境：装好 Python 和 PyTorch 即可（`pip install torch`）。有 GPU 用 GPU，没 GPU 用 CPU 也能在几分钟内跑完本文的小模型。

<!-- more -->

## 一、先建立直觉：机器怎么读句子？

### 1.1 第一步：把词变成数字

计算机只认识数字，不认识"猫""狗"。所以处理文字的第一步是**分词（Tokenize）**，再给每个词一个编号：

```
句子：我 爱 深度 学习
编号： 7  23  105  88
```

所有编号组成的对照表叫**词表（Vocabulary）**。本文为了最简单，直接用字符做单位，词表只有 `0~9` 十个数字，外加三个特殊符号（`<SOS>` 表示"开始"、`<EOS>` 表示"结束"、`<PAD>` 表示"补齐"）。

### 1.2 第二步：把编号变成向量

直接用编号 `7` 有个问题：数字 7 和 8 之间的"距离"，跟"猫"和"狗"之间的语义关系毫无关系。

于是我们给每个词分配一个**由很多个数字组成的向量**（比如 64 个数字），称为**词嵌入（Embedding）**，并让模型在学习中自动调整这些数字，最终使得：

- 意思相近的词，向量也相近（"猫"和"狗"都是动物，距离近；"猫"和"电脑"距离远）；
- 有某种关系的词，向量之间的方向有规律（"国王−男人+女人≈女王"是最经典的例子）。

Embedding 实现起来就是一张**查找表**：给定编号，去表里取出对应的那一行向量。

### 1.3 我们要解决的问题：序列到序列

很多任务的本质是"输入一串词，输出另一串词"：翻译（中文→英文）、摘要（长文→短标题）、对话（问题→回答）。这叫**序列到序列（Seq2Seq）**。

本文的反转任务就是一个迷你版 Seq2Seq：输入一串数字，输出它的逆序。虽然简单，但它需要模型真正"理解"每个位置和顺序，用来检验 Transformer 再合适不过。

## 二、Transformer 之前：为什么需要注意力？

### 2.1 老方法 RNN 的两个痛点

在 Transformer 之前，序列建模主要靠 RNN（循环神经网络）：像排队传话一样，一个词一个词地读，把前面的信息"记"在隐藏状态里，再传给下一步。它有两个致命问题：

1. **不能并行**：必须读完第 1 个词才能读第 2 个，句子越长越慢，GPU 的并行能力完全用不上；
2. **容易遗忘**：句子很长时，开头的信息传到后面已经"模糊"了，比如英文里主语和谓语可能隔了几十个词，模型常常对不上。

### 2.2 注意力的核心直觉

人在读一句话或做翻译时，**并不是对每个词平均用力**。听到"它"这个代词，你会立刻回头去想前面哪个名词才是"它"；看到"银行"，你会根据后面的"河水"判断它是河岸而不是存钱的地方。

**注意力机制（Attention）**就是把这种"有重点地看"教给模型：处理每个词时，让它去和句子里所有其他词打个"相关度分数"，再按分数高低把信息"取其精华"地汇总过来。

Transformer 的革命性在于：**它干脆抛弃了 RNN 的排队结构，全部用注意力**，于是既能高度并行、又能一眼看到任意两个词之间的关系。

## 三、鸟瞰：Transformer 长什么样？

先看整体架构，建立全局印象，后面我们再逐个零件拆开。

{% mermaid %}
flowchart LR
    SRC["源序列 314159"] --> EA
    subgraph ENC["编码器 Encoder 重复 N 层"]
      EA["词嵌入 + 位置编码"] --> EB["多头自注意力"] --> EC["前馈网络"]
    end
    TGT["已生成的目标序列 SOS 9 5 1"] --> DA
    subgraph DEC["解码器 Decoder 重复 N 层"]
      DA["词嵌入 + 位置编码"] --> DB["掩码多头自注意力"] --> DC["交叉注意力"] --> DD["前馈网络"]
    end
    ENC -->|"K、V（对源句的理解）"| DC
    DD --> OUT["线性层 + Softmax 预测下一个词"]
    OUT --> ANS["...951413"]
{% endmermaid %}

它分成两大块：

- **编码器（Encoder）**：负责"读懂"输入句子。核心是**自注意力**——句子里的词互相看；
- **解码器（Decoder）**：负责"生成"输出句子。它包含两种注意力：
  - **掩码自注意力**：看已经生成的词，但不许偷看未来；
  - **交叉注意力（Cross-Attention）**：一边生成，一边"查阅"编码器对源句的理解。

数据在模型里始终是一个三维张量，记住它能极大帮助你理解代码：

```
(批次大小 B, 序列长度 T, 词向量维度 d_model)
例如 (32, 6, 64)：32 个句子，每句 6 个词，每个词用 64 维向量表示
```

下面开始正式"手撕"，每个零件都给出作用、原理和代码。

## 四、零件一：词嵌入 Embedding

**作用**：把整数编号变成稠密向量。

```python
self.enc_embed = nn.Embedding(VOCAB, d_model)
# VOCAB=13 个词，每个词映射到 d_model=64 维向量
```

输入形状 `(B, T)`（里面是编号），输出形状变成 `(B, T, d_model)`。它内部就是一张 `13×64` 的可学习查找表，`nn.Embedding` 帮我们实现好了。

## 五、零件二：位置编码 Positional Encoding

### 5.1 为什么需要它？

注意力有个特点：它把每个词"一视同仁"地并行处理，**天生不知道词的先后顺序**。把句子"我打你"改成"你打我"，对模型来说输入的词是一样的。但顺序显然决定了语义！

解决办法：在词向量里**人为加上一个"位置信号"**，这就是位置编码。

### 5.2 用正弦波造一串"位置身份证"

论文用一组不同频率的正弦、余弦函数来表示位置，得到的位置向量有一个绝妙性质：**任意两个位置的相对距离可以被模型轻易地算出**。公式（不用背，看代码即可）：

```
PE(位置, 偶数维) = sin(位置 / 10000^(2i/d_model))
PE(位置, 奇数维) = cos(位置 / 10000^(2i/d_model))
```

逐行代码：

```python
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)                       # 先准备一张空表
        position = torch.arange(max_len).float().unsqueeze(1)   # (max_len,1) 位置 0,1,2...
        div = torch.exp(torch.arange(0, d_model, 2).float()
                        * (-math.log(10000.0) / d_model))        # 不同维度的频率
        pe[:, 0::2] = torch.sin(position * div)                 # 偶数维用 sin
        pe[:, 1::2] = torch.cos(position * div)                 # 奇数维用 cos
        self.register_buffer('pe', pe.unsqueeze(0))             # (1,max_len,d_model)，随模型保存但不训练
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])         # 词向量 + 位置向量
```

注意最后是 `x + pe`：**逐元素相加**，把位置信息"揉进"词向量里。加一个固定的位置编码，向量的形状不变，仍然是 `(B, T, d_model)`。

## 六、最核心的零件：注意力机制

这是整篇文章的灵魂，我们多花点篇幅讲透。

### 6.1 Q、K、V 三个角色：一个"图书馆查资料"的比喻

注意力里每个词都会被投影成三个向量，名字有点吓人，但角色非常好懂：

| 角色 | 全称 | 通俗理解 | 类比 |
|---|---|---|---|
| **Q** | Query（查询） | "我现在想找什么信息？" | 你在图书馆输入的**搜索关键词** |
| **K** | Key（键） | "我这段内容的标签/主题是什么？" | 每本书封面上的**书名标签** |
| **V** | Value（值） | "我这段内容真正包含的信息" | 书的**正文内容** |

当处理某个词时，它拿着自己的 **Q**，去和**所有词的 K** 做比对：

- 关键词（Q）和某本书的标签（K）越匹配，相关度分数就越高；
- 再用这些分数做权重，把所有书的正文（V）**加权求和**，得到这个词"看到"的综合信息。

### 6.2 计算三步走

{% mermaid %}
flowchart LR
    Q["查询 Q<br/>我想找什么"] --> SCORE
    K["键 K<br/>各内容的标签"] --> SCORE["第一步 Q·Kᵀ / √d<br/>算相关度分数"]
    SCORE --> SM["第二步 Softmax<br/>分数转成权重，和为 1"]
    V["值 V<br/>各内容的信息"] --> SUM["第三步 权重 × V<br/>加权求和"]
    SM --> SUM
    SUM --> OUT["注意力输出"]
{% endmermaid %}

对应那条著名的公式：

```
Attention(Q, K, V) = softmax( Q · Kᵀ / √d_k ) · V
```

### 6.3 为什么要除以 √d_k？

`Q·Kᵀ` 是很多个数字相乘再相加，向量维度 `d_k` 越大，结果的数值可能越大。数值一旦很大，softmax 会把最大的那个分数推到接近 1、其余接近 0，变成"非黑即白"，**梯度几乎消失，模型学不动**。

除以 `√d_k`（维度的平方根）就是把分数**缩放回温和的范围**，让 softmax 输出的权重更平滑、训练更稳定。这就是"**缩放**点积注意力"里"缩放"的含义。

### 6.4 一个手算的小例子

假设有 2 个词，每个词的 Q、K 是 2 维（数字我刻意凑简单）：

```
词1 的 Q = [1, 0]，词1 的 K = [1, 0]
词2 的 Q = [1, 0]，词2 的 K = [0, 1]

处理词1 时，Q·Kᵀ 得到分数：
  对词1：1×1 + 0×0 = 1
  对词2：1×0 + 0×1 = 0
Softmax 后：[0.73, 0.27]
```

含义：词1 有 73% 的注意力放在自己身上，27% 放在词2 上。**这就是"注意力权重"的本质——一个按重要性分配的比例。**

### 6.5 逐行代码

```python
def scaled_dot_product_attention(q, k, v, mask=None):
    # q,k,v 形状: (B, 头数, 序列长度 T, 每头维度 d_head)
    d_k = q.size(-1)
    scores = q @ k.transpose(-2, -1) / math.sqrt(d_k)   # ① 打分: (B,h,T,T)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))  # ② 掩码：把不该看的位置分数设为负无穷
    attn = torch.softmax(scores, dim=-1)                # ③ 权重，最后一维和为 1
    return attn @ v, attn                               # ④ 加权求和，顺便返回权重供观察
```

`scores` 的形状是 `(B, h, T, T)`——你可以理解为一张 `T×T` 的"相关度表"，第 i 行第 j 列就是"第 i 个词对第 j 个词的关注程度"。

## 七、零件三：多头注意力 Multi-Head Attention

### 7.1 一个脑袋不够用

一个注意力头，每对词之间只能学到**一种**关系。但语言里的关系是多方面的：有的头负责主语和谓语的一致性，有的头负责"它"指代谁，有的头负责并列结构……

**多头注意力**就是并行开 `h` 个注意力头，让它们从不同角度看同一句话，最后把各自的结果拼起来。

### 7.2 实现技巧：先切开，分头算，再拼回

做法很直接：把 64 维的向量切成 4 份，每份 16 维，就是 4 个头，各自独立做注意力，最后再拼回 64 维，过一个线性层融合。

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.h = n_heads
        self.d_head = d_model // n_heads
        self.wq = nn.Linear(d_model, d_model)            # 三个线性层把输入投影成 Q/K/V
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.wo = nn.Linear(d_model, d_model)            # 输出融合层

    def split_heads(self, x):                            # (B,T,D) -> (B,h,T,d_head)
        B, T, D = x.shape
        return x.view(B, T, self.h, self.d_head).transpose(1, 2)

    def forward(self, q, k, v, mask=None):
        B = q.size(0)
        q = self.split_heads(self.wq(q))
        k = self.split_heads(self.wk(k))
        v = self.split_heads(self.wv(v))
        out, attn = scaled_dot_product_attention(q, k, v, mask)
        # 合头：(B,h,T,d_head) -> (B,T,D)
        out = out.transpose(1, 2).contiguous().view(B, -1, self.h * self.d_head)
        return self.wo(out), attn
```

一个特别优雅的点：**同一套代码可以处理三种注意力**——

- 当 `q、k、v` 来自同一个序列（编码器内部），就是**自注意力**；
- 当 `q` 来自解码器、`k` 和 `v` 来自编码器（交叉注意力里），就是在"查阅"源句；
- 当 `q、k、v` 来自解码器自身、并加掩码，就是**掩码自注意力**。

## 八、零件四：掩码 Mask，让模型"不该看的别看"

### 8.1 解码器为什么不能看未来

训练时，我们把目标答案整体喂给解码器（这叫**教师强制 Teacher Forcing**，后面会讲）。但如果让解码器在生成第 1 个词时就看到了后面的全部答案，它什么都不用学，直接"抄"就行。

所以必须加一道**因果掩码（Causal Mask）**：保证第 t 个位置只能看见第 t 个位置及之前的内容。这就是一张下三角矩阵：

```
        能看 词1 词2 词3 词4
词1       ✓    ·    ·    ·
词2       ✓    ✓    ·    ·
词3       ✓    ✓    ✓    ·
词4       ✓    ✓    ✓    ✓
（✓ = 允许看，· = 禁止看，禁止处分数设为 -∞，softmax 后权重为 0）
```

### 8.2 逐行代码

```python
def make_causal_mask(t):
    # torch.tril 取下三角，形状 (1,1,t,t)，方便和 (B,h,T,T) 广播
    mask = torch.tril(torch.ones(t, t)).unsqueeze(0).unsqueeze(0)
    return mask.to(device)
```

`torch.tril` 把矩阵上三角全部变成 0、下三角保留为 1，配合前面注意力代码里的 `masked_fill(..., -inf)`，模型就"看"不到未来了。

## 九、零件五：前馈网络、残差连接、LayerNorm

### 9.1 前馈网络（Feed Forward）

注意力负责"词与词之间的信息交流"，而前馈网络负责在每个位置**内部做一次非线性加工**（升维、ReLU、再降维），增强模型的表达能力：

```python
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)    # 64 -> 128 升维
        self.fc2 = nn.Linear(d_ff, d_model)    # 128 -> 64 降维

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))
```

### 9.2 残差连接（Residual Connection）

每层都做一个"加法"：把这一层的**输入**和**输出**加起来，再往下传。

```
输出 = x + 子层(x)
```

直觉上像给信息留了一条"高速公路"：即使某层没学到有用的东西，信息也能原样抄过去，不会丢失；同时它让深层网络的梯度更容易回传，**训练几十、上百层也不容易崩**。

### 9.3 层归一化（LayerNorm）

对每个样本的特征做归一化，把数值拉回到均值 0、方差 1 附近的稳定范围，让训练更平稳。它和残差通常组合成：

```
输出 = LayerNorm( x + Dropout(子层(x)) )
```

## 十、把零件拼成 Encoder 和 Decoder

### 10.1 一个 Encoder 层

一个编码器层 = 一个多头自注意力 + 一个前馈网络，各自都配残差和 LayerNorm：

```python
class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads)
        self.ff = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(0.1)

    def forward(self, x):
        attn_out, _ = self.self_attn(x, x, x)          # 自注意力：q/k/v 都是自己
        x = self.norm1(x + self.drop(attn_out))        # 残差 + 归一化
        ff_out = self.ff(x)
        x = self.norm2(x + self.drop(ff_out))
        return x
```

### 10.2 一个 Decoder 层（比编码器多一个交叉注意力）

```python
class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads)   # 掩码自注意力
        self.cross_attn = MultiHeadAttention(d_model, n_heads)  # 交叉注意力
        self.ff = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(0.1)

    def forward(self, x, memory, causal_mask):
        attn_out, _ = self.self_attn(x, x, x, mask=causal_mask) # 看自己（戴掩码）
        x = self.norm1(x + self.drop(attn_out))
        # q 来自解码器当前状态，k/v 来自编码器的输出 memory —— "带着问题查资料"
        cross_out, _ = self.cross_attn(x, memory, memory)
        x = self.norm2(x + self.drop(cross_out))
        ff_out = self.ff(x)
        x = self.norm3(x + self.drop(ff_out))
        return x
```

**交叉注意力**是解码器的精髓：`q` 是解码器"我现在想生成什么、需要源句里的哪些信息"，`k、v` 是编码器对源句的全部理解。它让解码器在生成每个词时，都能回头精准定位输入句子中最相关的部分——翻译时这一步决定了"这个英文词对应哪个中文词"。

### 10.3 堆叠 N 层

单个层能学到的关系有限，我们把它堆叠 `n_layers` 层（本文用 2 层，GPT-3 用了 96 层），层层抽象，浅层学语法、深层学语义。

## 十一、完整模型与"教师强制"训练

### 11.1 拼装完整 Transformer

```python
class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc_embed = nn.Embedding(VOCAB, d_model)
        self.dec_embed = nn.Embedding(VOCAB, d_model)
        self.pos = PositionalEncoding(d_model)
        self.encoders = nn.ModuleList([EncoderLayer(d_model, n_heads, d_ff)
                                       for _ in range(n_layers)])
        self.decoders = nn.ModuleList([DecoderLayer(d_model, n_heads, d_ff)
                                       for _ in range(n_layers)])
        self.fc_out = nn.Linear(d_model, VOCAB)          # 输出投影：64 维 -> 词表大小

    def encode(self, src):
        x = self.pos(self.enc_embed(src))
        for layer in self.encoders:
            x = layer(x)
        return x                                          # 编码器对源句的理解 memory

    def decode(self, tgt, memory, causal_mask):
        x = self.pos(self.dec_embed(tgt))
        for layer in self.decoders:
            x = layer(x, memory, causal_mask)
        return self.fc_out(x)                             # 每个位置输出 13 个词的得分

    def forward(self, src, tgt, causal_mask):
        memory = self.encode(src)
        return self.decode(tgt, memory, causal_mask)
```

最后 `fc_out` 把 64 维向量投影回词表大小（13），得到每个位置对"下一个词是哪个"的打分，再接 Softmax 就变成概率。

### 11.2 教师强制：训练时怎么教解码器

训练时不需要一个字一个字地等，而是把**正确答案整体错位**喂给解码器（这叫 Teacher Forcing）：

```
源输入 src     : 3 1 4 1 5 9
解码器输入 tgt_in : <SOS> 9 5 1 4 1 3
要预测的标签 tgt_out: 9 5 1 4 1 3 <EOS>
```

看明白了吗？解码器在 `<SOS>` 位置要学会预测 `9`，在看到 `9` 的位置预测 `5`……每一步都拿"标准答案"作为输入，并行地训练，效率极高。真正自己生成时（推理阶段）才会一个字一个字地往外蹦。

## 十二、玩具任务：反转数字串

### 12.1 数据自动生成

不需要任何外部数据集，随机生成数字串，答案就是它的反转：

```python
def make_batch(batch_size):
    src = torch.randint(0, DIGITS, (batch_size, SRC_LEN))      # 随机 6 位数字
    reversed_src = src.flip(1)                                  # 反转
    tgt_in = torch.cat([torch.full((batch_size, 1), SOS), reversed_src], dim=1)
    tgt_out = torch.cat([reversed_src, torch.full((batch_size, 1), EOS)], dim=1)
    return src.to(device), tgt_in.to(device), tgt_out.to(device)
```

### 12.2 训练循环

```python
model = Transformer().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
criterion = nn.CrossEntropyLoss()

for step in range(1, 1001):
    src, tgt_in, tgt_out = make_batch(128)
    causal_mask = make_causal_mask(TGT_LEN)
    logits = model(src, tgt_in, causal_mask)                 # (B, 7, 13)
    loss = criterion(logits.reshape(-1, VOCAB), tgt_out.reshape(-1))
    optimizer.zero_grad(); loss.backward(); optimizer.step()
```

三步标准套路：**前向算 loss → 反向传播 → 优化器更新参数**，和训练任何神经网络一样。

## 十三、完整代码（一个文件，直接可跑）

把上面所有零件拼起来，保存为 `transformer_demo.py`，运行 `python transformer_demo.py`：

```python
# -*- coding: utf-8 -*-
# 手撕 Transformer：从零逐模块实现，在"数字串反转"任务上训练
# 任务：输入 314159 -> 输出 951413（字符级序列到序列）
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ========== 0. 词表与超参数 ==========
DIGITS = 10                      # 0~9 共 10 个字符
PAD, SOS, EOS = 10, 11, 12       # 三个特殊符号
VOCAB = 13
SRC_LEN = 6                      # 输入数字串长度
TGT_LEN = SRC_LEN + 1            # 目标长度（6 个反转数字 + 1 个 EOS）

d_model = 64                     # 每个词向量的维度
n_heads = 4                      # 多头注意力的"头数"
d_ff = 128                       # 前馈网络隐藏层维度
n_layers = 2                     # Encoder / Decoder 各堆几层
dropout = 0.1


# ========== 1. 位置编码 ==========
class PositionalEncoding(nn.Module):
    """正弦位置编码：告诉模型每个词在句子里的位置"""
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len).float().unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float()
                        * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div)
        self.register_buffer('pe', pe.unsqueeze(0))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])


# ========== 2. 缩放点积注意力 ==========
def scaled_dot_product_attention(q, k, v, mask=None):
    # q,k,v 形状: (B, 头数, 序列长度, 每头维度)
    d_k = q.size(-1)
    scores = q @ k.transpose(-2, -1) / math.sqrt(d_k)         # (B,h,T,T)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    attn = torch.softmax(scores, dim=-1)
    return attn @ v, attn


# ========== 3. 多头注意力 ==========
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.h = n_heads
        self.d_head = d_model // n_heads
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.wo = nn.Linear(d_model, d_model)

    def split_heads(self, x):                                # (B,T,D)->(B,h,T,d_head)
        B, T, D = x.shape
        return x.view(B, T, self.h, self.d_head).transpose(1, 2)

    def forward(self, q, k, v, mask=None):
        B = q.size(0)
        q = self.split_heads(self.wq(q))
        k = self.split_heads(self.wk(k))
        v = self.split_heads(self.wv(v))
        out, attn = scaled_dot_product_attention(q, k, v, mask)
        out = out.transpose(1, 2).contiguous().view(B, -1, self.h * self.d_head)
        return self.wo(out), attn


# ========== 4. 前馈网络 ==========
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


# ========== 5. Encoder 一层 ==========
class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads)
        self.ff = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        attn_out, _ = self.self_attn(x, x, x)
        x = self.norm1(x + self.drop(attn_out))
        x = self.norm2(x + self.drop(self.ff(x)))
        return x


# ========== 6. Decoder 一层 ==========
class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads)
        self.cross_attn = MultiHeadAttention(d_model, n_heads)
        self.ff = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, memory, causal_mask):
        attn_out, _ = self.self_attn(x, x, x, mask=causal_mask)
        x = self.norm1(x + self.drop(attn_out))
        cross_out, _ = self.cross_attn(x, memory, memory)
        x = self.norm2(x + self.drop(cross_out))
        x = self.norm3(x + self.drop(self.ff(x)))
        return x


# ========== 7. 完整 Transformer ==========
class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc_embed = nn.Embedding(VOCAB, d_model)
        self.dec_embed = nn.Embedding(VOCAB, d_model)
        self.pos = PositionalEncoding(d_model)
        self.encoders = nn.ModuleList([EncoderLayer(d_model, n_heads, d_ff)
                                       for _ in range(n_layers)])
        self.decoders = nn.ModuleList([DecoderLayer(d_model, n_heads, d_ff)
                                       for _ in range(n_layers)])
        self.fc_out = nn.Linear(d_model, VOCAB)

    def encode(self, src):
        x = self.pos(self.enc_embed(src))
        for layer in self.encoders:
            x = layer(x)
        return x

    def decode(self, tgt, memory, causal_mask):
        x = self.pos(self.dec_embed(tgt))
        for layer in self.decoders:
            x = layer(x, memory, causal_mask)
        return self.fc_out(x)

    def forward(self, src, tgt, causal_mask):
        memory = self.encode(src)
        return self.decode(tgt, memory, causal_mask)


def make_causal_mask(t):
    """下三角为 1、上三角为 0，保证第 t 个位置只能看到它和之前的位置"""
    mask = torch.tril(torch.ones(t, t)).unsqueeze(0).unsqueeze(0)
    return mask.to(device)


# ========== 8. 数据生成 ==========
def make_batch(batch_size):
    src = torch.randint(0, DIGITS, (batch_size, SRC_LEN))
    reversed_src = src.flip(1)
    tgt_in = torch.cat([torch.full((batch_size, 1), SOS), reversed_src], dim=1)
    tgt_out = torch.cat([reversed_src, torch.full((batch_size, 1), EOS)], dim=1)
    return src.to(device), tgt_in.to(device), tgt_out.to(device)


# ========== 9. 训练 ==========
model = Transformer().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
criterion = nn.CrossEntropyLoss()

print(f"运行设备: {device}")
for step in range(1, 1001):
    src, tgt_in, tgt_out = make_batch(128)
    causal_mask = make_causal_mask(TGT_LEN)
    logits = model(src, tgt_in, causal_mask)
    loss = criterion(logits.reshape(-1, VOCAB), tgt_out.reshape(-1))
    optimizer.zero_grad(); loss.backward(); optimizer.step()

    if step % 100 == 0:
        with torch.no_grad():
            pred = logits.argmax(-1)
            correct = (pred[:, :SRC_LEN] == tgt_out[:, :SRC_LEN]).all(dim=1).float().mean()
        print(f"第 {step:4d} 步  loss {loss.item():.4f}  整句准确率 {correct.item()*100:5.1f}%")


# ========== 10. 贪心解码：一个字一个字地生成 ==========
print("\n训练后效果演示：")
model.eval()
with torch.no_grad():
    src = torch.randint(0, DIGITS, (5, SRC_LEN)).to(device)
    memory = model.encode(src)
    ys = torch.full((5, 1), SOS, dtype=torch.long).to(device)  # 先只放一个 <SOS>
    for _ in range(TGT_LEN):
        logits = model.decode(ys, memory, make_causal_mask(ys.size(1)))
        next_token = logits[:, -1].argmax(-1, keepdim=True)   # 取最后一个位置的预测
        ys = torch.cat([ys, next_token], dim=1)               # 拼回去，继续下一步

    for i in range(5):
        inp = ''.join(str(n) for n in src[i].tolist())
        out = ''.join(str(n) for n in ys[i, 1:1+SRC_LEN].tolist())
        print(f"  输入 {inp}  ->  预测 {out}  {'正确' if out == inp[::-1] else '错误'}")
```

> 说明：为了聚焦核心原理，本文省略了变长序列所需的 **Padding Mask（补齐掩码）** 和 **Beam Search（束搜索解码）** 等进阶细节。它们不影响你理解 Transformer 的主干，等主干吃透后再去补即可。

## 十四、运行结果：它真的"学会"反转了

在笔者的 GPU 上（CPU 也能跑，稍慢），训练过程如下：

```
运行设备: cuda
第  100 步  loss 1.2250  整句准确率   1.6%
第  200 步  loss 0.3171  整句准确率  43.8%
第  300 步  loss 0.1449  整句准确率  80.5%
第  400 步  loss 0.0899  整句准确率  87.5%
第  500 步  loss 0.0776  整句准确率  85.9%
第  600 步  loss 0.0732  整句准确率  86.7%
第  700 步  loss 0.0625  整句准确率  86.7%
第  800 步  loss 0.0378  整句准确率  92.2%
第  900 步  loss 0.0609  整句准确率  88.3%
第 1000 步  loss 0.0545  整句准确率  90.6%

训练后效果演示：
  输入 179985  ->  预测 589971  正确
  输入 197741  ->  预测 147791  正确
  输入 629691  ->  预测 196926  正确
  输入 735289  ->  预测 982537  正确
  输入 295672  ->  预测 276592  正确
```

你可以清晰地看到"学习"的过程：

- **前 100 步**：loss 约 1.2，整句准确率只有 1.6%——模型基本在瞎猜（10 个数字随机猜对的概率本就很低）；
- **100~300 步**：loss 快速下降到 0.14，准确率飙到 80%，它开始摸到规律；
- **300 步之后**：loss 缓慢下降到 0.05，准确率稳定在 90% 左右，曲线有正常波动（每批数据随机，不必追求每一步都更高）。

最终随机抽 5 个全新数字串，**全部正确反转**。一个我们只写了"零件"、从没教过它"怎么反转"的模型，靠自己从海量随机样本里学到了规律——这就是深度学习的魅力。

## 十五、推理时它是怎么"逐字生成"的？

训练时答案是整体喂进去的（教师强制），但真实使用时，模型得**一个字一个字地往外蹦**，这叫**自回归生成**。回看代码最后那段：

```
第 1 步：输入 [<SOS>]                 -> 预测出 9
第 2 步：输入 [<SOS>, 9]             -> 预测出 5
第 3 步：输入 [<SOS>, 9, 5]          -> 预测出 1
……
第 7 步：拼到 6 个数字后，预测 <EOS> -> 结束
```

每一步都把"已经生成的内容"拼回输入，再预测下一个。每一步都取概率最高的那个词，叫**贪心解码（Greedy Decoding）**。ChatGPT 打字时一个个往外蹦，本质就是这个循环，只不过它的词表和模型大了几亿倍。

## 十六、张量形状变化总表（强烈建议收藏）

零基础最容易在形状上迷路，把这张表对照着代码看：

| 阶段 | 张量形状 | 含义 |
|---|---|---|
| 原始输入 | `(B, T)` | B 句话，每句 T 个词编号 |
| 词嵌入后 | `(B, T, d_model)` | 每个词变成 64 维向量 |
| 加位置编码后 | `(B, T, d_model)` | 形状不变，注入了位置信息 |
| 分头后 Q/K/V | `(B, n_heads, T, d_head)` | 每个头独立处理 16 维 |
| 注意力分数 | `(B, n_heads, T, T)` | 词与词之间的相关度表 |
| 注意力输出 | `(B, T, d_model)` | 合头后回到原形状 |
| 最终输出 | `(B, T, VOCAB)` | 每个位置对 13 个词的打分 |

## 十七、零基础常见疑问

**Q1：注意力到底在"注意"什么？**
它学习的是一张张"相关度权重表"。在训练中，模型会自动学会：反转任务里位置 i 应该高度关注位置 `T-1-i`；翻译任务里中文词和它对应的英文词之间权重高。这些都不是人写的规则，而是从数据里学出来的。

**Q2：为什么 Q、K、V 要三个不同的线性层，不能共用一个吗？**
因为它们的"角色诉求"不同：我想表达的（V）、我给别人的标签（K）、我想找的（Q）必须允许模型学出不同的表示，否则表达能力会受限。

**Q3：没有了 RNN，模型为什么还能知道词的顺序？**
靠位置编码（第五节）。它把"绝对位置"和"相对位置"的信号直接加进了向量，注意力又能利用这些信号。

**Q4：Encoder 和 Decoder 一定要同时存在吗？**
不是，这是理解现代大模型的关键：

- **只有 Encoder**：BERT，擅长理解（分类、实体识别、语义匹配）；
- **只有 Decoder**：GPT 系列、DeepSeek，擅长生成（聊天、写作）；
- **Encoder + Decoder**：经典 Transformer、Google T5，擅长翻译、摘要这类有明确输入输出的转换。

本文为了让你看到完整结构，用了最经典的 Encoder-Decoder 版。

**Q5：这个 100 行的小模型和 ChatGPT 是什么关系？**
核心机制（注意力、多头、位置编码、残差、LayerNorm、自回归生成）完全一样。区别在于规模：ChatGPT 有上千亿参数、几十上百层、海量语料和大量工程优化（更稳的归一化、旋转位置编码 RoPE、KV Cache 等）。**看懂了这个小模型，你就拿到了理解大模型的钥匙。**

**Q6：没有 GPU 能跑吗？**
能。本文的模型只有约 2 万参数，CPU 上几分钟就能跑完 1000 步，把代码里的设备设成 `cpu` 即可（代码会自动判断）。

## 十八、总结

回顾一下，我们这一路手撕了 Transformer 的完整链路：

```
词编号
  → 词嵌入 Embedding（编号变向量）
  → 加位置编码（注入顺序信息）
  → 编码器：多头自注意力（词互相看）+ 前馈网络，外面包残差和 LayerNorm，堆 N 层
  → 解码器：掩码自注意力（看过去）+ 交叉注意力（查源句）+ 前馈网络，堆 N 层
  → 线性层投影到词表 + Softmax（预测下一个词）
```

而自注意力的核心，请务必记住这一行：

```
Attention(Q, K, V) = softmax(Q·Kᵀ / √d_k) · V
```

**动手建议（按顺序做，收益最大）：**

1. 跑通本文代码，试着改 `d_model`、`n_heads`、`n_layers`，观察 loss 和速度变化；
2. 把反转任务改成"排序"（把乱序数字排好），验证模型是否一样能学；
3. 打印并可视化某一层的注意力权重（代码里 `attn` 已经返回了），看看它是不是真的在关注镜像位置；
4. 去读进阶资料：原论文《Attention Is All You Need》、Harvard 的 *The Annotated Transformer*（逐行带注释版）；
5. 进阶挑战：把 Encoder 去掉，只用 Decoder 实现一个迷你 GPT（字符级生成文本），你会发现思路几乎一模一样。

当你能对着一张 Transformer 结构图，说出每个张量的形状、每个零件的作用、数据怎么进怎么出时，你就已经真正入门了。下一篇我们真的动手：{% post_link mini-gpt-from-scratch '手撕迷你 GPT：从预测下一个字到会写句子' %}，把生成式大模型也拆给你看。
