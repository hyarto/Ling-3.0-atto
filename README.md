# Ling-3.0-atto

轻量版 [Ling-3.0-tiny](https://huggingface.co/inclusionAI/Ling-3.0-tiny)（BailingMoeV3）：**~0.7B 骨干 + 1.2B Engram**，
词表沿用 [LFM3](https://huggingface.co/LiquidAI/LFM3-1.7B) 的 tokenizer。纯 PyTorch，无自定义 CUDA 依赖（可选 `flash-linear-attention`）。

| | Ling-3.0-tiny | **Ling-3.0-atto** |
|---|---|---|
| 层数 / hidden | 24 / 1536 | 20 / 1024 |
| KDA : MLA | 3 : 1 (`layer_group_size=4`) | **4 : 1** (`layer_group_size=5`，每组最后一层为 MLA → 16 KDA + 4 MLA) |
| MoE | 128 专家, top-8, 8 组/选 4 组, sigmoid + 偏置 (noaux_tc), 1 共享专家 | 32 专家, top-4, 4 组/选 2 组, 同样的路由；首层 dense |
| MLA | q_lora 256, kv_lora 512, 128+64 | q_lora 256, kv_lora 192, 96+32 (v 96), head-wise sigmoid gate |
| KDA | 16×128, short conv 4, safe gate (下界 −5) | 8×128, short conv 4, safe gate (下界 −5) |
| 词表 | 157184 | LFM3 125033 token（embedding 对齐 128000，tied） |
| Engram | – | **2 层 (1, 11)，共 1.2B 参数**（见下） |

参数量（`build_model(cfg, meta=True)` 在 meta 设备上精确统计，不占内存）：

```
atto_full  backbone 710.3M  (embedding 131.1M | KDA mixers 101M | MLA 7.4M | FFN/MoE 471M)   激活参数/token ≈ 318M
           engram   1.200B  (tables 1.198B + projections 2.1M)
nano       backbone 491K    (embedding 131K)     engram 301K     <- 冒烟配置：词表 2048
```

## Engram：1.2B 参数下如何对抗哈希碰撞

DeepSeek 式 Engram 每个 bucket 存一个模型宽度的向量。1.2B 参数、宽 1024 的行只有 ~1.2M 个桶，而真实语料里不同的 3-gram 数以亿计，
平均每桶 >100 个 n-gram。本实现（`ling_atto/engram.py`）：

1. **低秩 bucket 编码**：每行只存 `rank=32` 维的码，再由一个作用在“所有头的码拼接”上的 `wkv`（`cols·rank → 2·hidden`）上投影到 key / value。
   同样的参数预算桶数多 `hidden/rank = 32×`（等价于 `E[bucket] = U[bucket]·W`，W 共享）。全量配置共 37.5M 行。
2. **每阶 K=8 个独立哈希头**（8 个互异素数模数，hash-embedding 思路）：两个 n-gram 只有在（几乎）所有头里同时撞才会被混淆；部分碰撞只是拼接码里几个坐标的小噪声，`wkv` 可平均掉。
3. **带符号哈希（count-sketch）**：每个 (头, n-gram) 抽 ±1 符号，碰撞行叠加是零均值噪声而不是偏置。
4. **按阶分配预算**（2-gram : 3-gram = 25% : 75%）：二元组数量少且高频，三元组稀疏且量大。
5. **上下文门控**（沿用 DeepSeek）：取回的 value 乘以 `sigmoid(signed_sqrt(⟨h, key⟩))`，碰撞带来的“不相关记忆”与当前隐状态匹配度低，门被关掉；之后再接 Engram 论文里的因果深度卷积（初始化为 0，起步恒等）。
6. **token 归一化压缩表**（`build_compressed_token_map`，大小写/重音/空白归并：125033 → 90501）：哈希前缩小字母表。
7. n-gram **不跨 BOS**（packed 文档边界），序列开头用 pad 填充；哈希乘数奇数且有界，`token×mult` 不会 int64 溢出。

`scripts/collision_report.py` 在**相同参数预算**下对比布局（TinyStories 20 万 token，nano 预算 30 万参数/层，`logs/collision_nano.txt`）：

| 布局 | 每头桶数 (3-gram) | 每桶平均 3-gram 数 | 全部头都相同 (不可区分) |
|---|---|---|---|
| 宽行 1 头（DeepSeek 式） | 2,221 | 41.9 | 100% |
| 宽行 4 头 | 487 | 191 | 0% |
| 低秩 1 头 | 18,637 | 5.0 | 99.3% |
| **低秩 4 头 + 符号 + 25/75（本项目）** | 6,829 | 13.6 | **0%** |

“不可区分”是理想化的可辨识性下界（模型仍需从 K 个带噪声的码中解码，这正是门控与 `wkv` 的作用）；“每桶平均 n-gram 数”则是单个头里的噪声量。
低秩行同时把二者拉到可接受区间，宽行要么单头噪声大，要么多头后桶太少。

### Engram 的训练显存（全量 1.2B）

| | |
|---|---|
| bf16 表 | 2.40 GB |
| 若用 AdamW（fp32 m,v + 稠密梯度） | 9.6 + 4.8 GB ← **避免** |
| `RowwiseAdagrad`（稀疏梯度，每行 1 个 fp32 累加器，bf16 权重 + **随机舍入**） | 0.15 GB + 仅被触及的行 |

小学习率更新会被 bf16 的就近舍入吞掉，所以更新用随机舍入（无偏，有单测）。

## 速度 / 显存优化

* **KDA 分块算法**（`kda.py`，纯 torch）：WY/UT 变换推广到逐通道衰减；与块内无关的部分（A、(I+A)⁻¹、Aqk、衰减）对**所有 chunk 并行**计算，串行循环里只剩几个小 matmul。
  指数项以 chunk 中点重新居中并截断，配合有界门控（≥ −5）在 fp32 下数值安全（`kda_chunk_size×|下界|/2 ≤ 60`，违反时直接报错；默认 16）。与朴素递推逐元素对拍（含强衰减、非整块长度、状态续接、反向梯度）。
* **KDA 核心重计算**（`kda_core_checkpoint`，默认开）：归一化后的 q/k、门控、每块衰减、A、Aqk、U、W 等 fp32 中间量全部只存在于一个被 checkpoint 的函数里，反向时重算，只保存 bf16 输入。
  实测已保存激活 **277 MB → 181 MB（−35%）**，耗时 +7%（CPU nano，B=8,T=512）。再叠加层级 `--grad_ckpt`：**8.7 MB**，耗时 +28%。
* 融合投影：KDA 的 q/k/v/f/g/β 一次 matmul；q/k/v 共用一个深度卷积；MLA 的 q-down 与 kv-down 合并。
* MLA 用 SDPA（v 零填充到 qk 维，走 flash 内核）。
* **MoE**：专家权重堆叠成 `[E,H,2I]`/`[E,I,H]`，按专家 id 排序分发，没有 HF 参考实现里的 `repeat_interleave`（K 倍激活拷贝）和 ModuleList；aux-loss-free 偏置 + 极小的序列级 aux loss。
* **分块 + checkpoint 的 LM head 交叉熵**：128k 词表下 `[B·T, V]` logits 永远不会整块存在。
* 稠密参数 fused AdamW，仅对 ≥2 维权重做 weight decay；TF32；bf16 autocast。
* 注意：沙箱没有 GPU，只验证了 CPU 路径；`flash-linear-attention` 的 KDA 内核是**可选且带异常回退**的（未在 GPU 上验证），默认走纯 torch 实现。

## 使用

```bash
pip install -e .[dev]
python -c "from ling_atto.tokenizer import fetch_lfm3_tokenizer as f; f()"   # 下载 LFM3 tokenizer 到 assets/lfm3
pytest -q                                                                      # 24 项测试

# 冒烟训练：LFM3 BPE + 缩减词表(2048) + 49 万参数骨干 + 30 万参数 Engram
python -m ling_atto.train --config nano --data synthetic   --steps 200
python -m ling_atto.train --config nano --data tinystories --steps 600
python -m ling_atto.train --config nano --data tinystories --steps 600 --no_engram   # 对照

PYTHONPATH=. python scripts/bench.py --config nano --batch 8 --seq 512        # 速度 / 激活显存
PYTHONPATH=. python scripts/collision_report.py --data tinystories --vocab 2048 --budget 300000 --hidden 64

# 全量配置（0.71B + 1.2B Engram）：需要 GPU
python -m ling_atto.train --config atto_full --data <text file> --batch_size 4 --seq_len 2048 --grad_ckpt
```

缩减词表（`ReducedVocab`）：用真实的 LFM3 BPE 切词，保留语料里最高频的 N 个 token（+4 个特殊 id），其余映射到 `<unk>`；Engram 的归一化压缩表仍按每个保留 token 的**原始 LFM3 字符串**构建。

## 冒烟结果（CPU，TinyStories 前 8MB，2M token，600 步，B=16,T=128）

| | 参数 | val loss | val ppl |
|---|---|---|---|
| 骨干 only | 491K | 3.2995 | 27.10 |
| 骨干 + Engram（300K 参数，两层） | 792K | **3.0414** | **20.94** |

（Engram 对照不是参数等价的：多了 300K 参数。日志见 `logs/`。）synthetic 语料上 200 步后能正确续写 “The capital of France is Paris.”。

## 布局

```
ling_atto/{config,tokenizer,kda,layers,engram,model,optim,data,train}.py
configs/{atto_full,nano}.json     scripts/{bench.py,collision_report.py,gh_push.sh}     tests/
```

## 已知限制

* 生成不带缓存（每步重算前缀），仅用于冒烟；KDA 的递推形式 (`kda_recurrent`) 已有，可用于 O(1) 解码。
* MoE 分发有一次 `counts.tolist()` 的 host 同步；大规模训练可换 grouped-GEMM。
* 全量 `atto_full` 只做了 meta 设备的结构 / 参数量验证，未实际在 GPU 上训练。
