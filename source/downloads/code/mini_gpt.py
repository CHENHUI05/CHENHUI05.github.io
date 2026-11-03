# -*- coding: utf-8 -*-
# 手撕迷你 GPT：字符级、4 层、128 维，在唐诗上训练
import math
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1337)

POEMS = """静夜思
李白
床前明月光，疑是地上霜。
举头望明月，低头思故乡。

春晓
孟浩然
春眠不觉晓，处处闻啼鸟。
夜来风雨声，花落知多少。

登鹳雀楼
王之涣
白日依山尽，黄河入海流。
欲穷千里目，更上一层楼。

相思
王维
红豆生南国，春来发几枝。
愿君多采撷，此物最相思。

悯农
李绅
锄禾日当午，汗滴禾下土。
谁知盘中餐，粒粒皆辛苦。

江雪
柳宗元
千山鸟飞绝，万径人踪灭。
孤舟蓑笠翁，独钓寒江雪。

寻隐者不遇
贾岛
松下问童子，言师采药去。
只在此山中，云深不知处。

鹿柴
王维
空山不见人，但闻人语响。
返景入深林，复照青苔上。

登高
杜甫
风急天高猿啸哀，渚清沙白鸟飞回。
无边落木萧萧下，不尽长江滚滚来。
万里悲秋常作客，百年多病独登台。
艰难苦恨繁霜鬓，潦倒新停浊酒杯。

望庐山瀑布
李白
日照香炉生紫烟，遥看瀑布挂前川。
飞流直下三千尺，疑是银河落九天。

早发白帝城
李白
朝辞白帝彩云间，千里江陵一日还。
两岸猿声啼不住，轻舟已过万重山。

绝句
杜甫
两个黄鹂鸣翠柳，一行白鹭上青天。
窗含西岭千秋雪，门泊东吴万里船。

春望
杜甫
国破山河在，城春草木深。
感时花溅泪，恨别鸟惊心。
烽火连三月，家书抵万金。
白头搔更短，浑欲不胜簪。

山居秋暝
王维
空山新雨后，天气晚来秋。
明月松间照，清泉石上流。
竹喧归浣女，莲动下渔舟。
随意春芳歇，王孙自可留。

使至塞上
王维
单车欲问边，属国过居延。
征蓬出汉塞，归雁入胡天。
大漠孤烟直，长河落日圆。
萧关逢候骑，都护在燕然。

送元二使安西
王维
渭城朝雨浥轻尘，客舍青青柳色新。
劝君更尽一杯酒，西出阳关无故人。

九月九日忆山东兄弟
王维
独在异乡为异客，每逢佳节倍思亲。
遥知兄弟登高处，遍插茱萸少一人。

黄鹤楼送孟浩然之广陵
李白
故人西辞黄鹤楼，烟花三月下扬州。
孤帆远影碧空尽，唯见长江天际流。

枫桥夜泊
张继
月落乌啼霜满天，江枫渔火对愁眠。
姑苏城外寒山寺，夜半钟声到客船。

游子吟
孟郊
慈母手中线，游子身上衣。
临行密密缝，意恐迟迟归。
谁言寸草心，报得三春晖。

望岳
杜甫
岱宗夫如何？齐鲁青未了。
造化钟神秀，阴阳割昏晓。
荡胸生曾云，决眦入归鸟。
会当凌绝顶，一览众山小。

饮酒
陶渊明
结庐在人境，而无车马喧。
问君何能尔？心远地自偏。
采菊东篱下，悠然见南山。
山气日夕佳，飞鸟相与还。
此中有真意，欲辨已忘言。

赋得古原草送别
白居易
离离原上草，一岁一枯荣。
野火烧不尽，春风吹又生。
远芳侵古道，晴翠接荒城。
又送王孙去，萋萋满别情。

泊船瓜洲
王安石
京口瓜洲一水间，钟山只隔数重山。
春风又绿江南岸，明月何时照我还。

题西林壁
苏轼
横看成岭侧成峰，远近高低各不同。
不识庐山真面目，只缘身在此山中。

饮湖上初晴后雨
苏轼
水光潋滟晴方好，山色空蒙雨亦奇。
欲把西湖比西子，淡妆浓抹总相宜。

竹石
郑燮
咬定青山不放松，立根原在破岩中。
千磨万击还坚劲，任尔东西南北风。

石灰吟
于谦
千锤万凿出深山，烈火焚烧若等闲。
粉骨碎身浑不怕，要留清白在人间。

己亥杂诗
龚自珍
九州生气恃风雷，万马齐喑究可哀。
我劝天公重抖擞，不拘一格降人才。

凉州词
王翰
葡萄美酒夜光杯，欲饮琵琶马上催。
醉卧沙场君莫笑，古来征战几人回。
"""

# 可选：把任意 UTF-8 文本放到脚本同目录的 corpus.txt，会优先使用它
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CORPUS_PATH = os.path.join(BASE_DIR, 'corpus.txt')
if os.path.exists(CORPUS_PATH):
    with open(CORPUS_PATH, encoding='utf-8') as f:
        text = f.read()
    src_name = 'corpus.txt'
else:
    text = POEMS
    src_name = '内置唐诗集'

chars = sorted(set(text))
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}
vocab_size = len(chars)

data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]

block_size = 64
batch_size = 32
n_embd = 128
n_head = 4
n_layer = 4
dropout = 0.1


def get_batch(split):
    d = train_data if split == 'train' else val_data
    ix = torch.randint(len(d) - block_size - 1, (batch_size,))
    x = torch.stack([d[i:i + block_size] for i in ix])
    y = torch.stack([d[i + 1:i + 1 + block_size] for i in ix])
    return x, y


class CausalSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        assert n_embd % n_head == 0
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.register_buffer(
            'mask',
            torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size)
        )

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(C, dim=2)
        d_head = C // n_head
        q = q.view(B, T, n_head, d_head).transpose(1, 2)
        k = k.view(B, T, n_head, d_head).transpose(1, 2)
        v = v.view(B, T, n_head, d_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(d_head)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention()
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class MiniGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block() for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-8)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


@torch.no_grad()
def estimate_loss(model, iters=20):
    model.eval()
    out = {}
    for split in ['train', 'val']:
        losses = torch.zeros(iters)
        for i in range(iters):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[i] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def main():
    model = MiniGPT()
    n_params = sum(p.numel() for p in model.parameters())
    print(f'语料: {src_name}  字符数: {len(text)}  词表大小: {vocab_size}')
    print(f'参数量: {n_params/1e6:.2f}M  设备: cpu')

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, betas=(0.9, 0.99), weight_decay=0.01)
    max_steps = 3000
    warmup = 100

    def lr_lambda(step):
        if step < warmup:
            return step / warmup
        progress = (step - warmup) / (max_steps - warmup)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    start = time.time()
    for step in range(max_steps + 1):
        if step % 250 == 0:
            losses = estimate_loss(model)
            print(f'step {step:5d} | train {losses["train"]:.3f} | val {losses["val"]:.3f} | '
                  f'{time.time()-start:6.1f}s')
        if step == max_steps:
            break
        x, y = get_batch('train')
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

    prompts = ['春眠', '明月', '山', '大漠']
    for p in prompts:
        idx = torch.tensor([[stoi[c] for c in p]])
        out = model.generate(idx, max_new_tokens=80, temperature=0.8, top_k=20)
        gen = ''.join(itos[i] for i in out[0].tolist())
        print('=' * 40)
        print(f'提示词「{p}」->')
        print(gen)

    # 贪心解码（temperature 极低）对比
    idx = torch.tensor([[stoi[c] for c in '春眠']])
    out = model.generate(idx, max_new_tokens=60, temperature=1e-6)
    print('=' * 40)
    print('贪心解码（几乎不吃随机性）->')
    print(''.join(itos[i] for i in out[0].tolist()))


if __name__ == '__main__':
    main()
