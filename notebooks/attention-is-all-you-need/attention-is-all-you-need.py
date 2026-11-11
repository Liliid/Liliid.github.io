# %% [markdown]
# # Attention Is All You Need — 미니 구현과 재현 실험
#
# - 블로그 글: https://Liliid.github.io/posts/attention-is-all-you-need/
# - 논문: https://arxiv.org/abs/1706.03762
#
# 이 노트북 하나로 Transformer 구현 → 저차원 예시 검증 → Multi30k 영어→독일어 번역 ablation 실험까지 실행된다.
# Colab에서는 `런타임 > 런타임 유형 변경 > T4 GPU`를 선택한 뒤 실행한다. T4에서는 전체 실험에 1시간 정도 걸린다.

# %%
# Colab 기본 환경에 없는 패키지만 설치한다 (로컬에서 .py로 실행하면 주석이라 무시된다)
# !pip install -q datasets tokenizers sacrebleu

# %%
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SMOKE = os.environ.get("SMOKE") == "1"  # 1이면 전체 흐름만 빠르게 확인
OUT = Path(os.environ.get("OUTDIR", "outputs"))
OUT.mkdir(exist_ok=True)
DATA = Path("data")
DATA.mkdir(exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device, torch.cuda.get_device_name(0) if device.type == "cuda" else "")
torch.backends.cuda.matmul.allow_tf32 = True


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class Config:
    # 모델 (논문 base: N=6, d_model=512, d_ff=2048, h=8 → 여기서는 절반 규모)
    N: int = 3
    d_model: int = 256
    d_ff: int = 1024
    h: int = 8
    p_drop: float = 0.1
    eps_ls: float = 0.1  # label smoothing
    scale: bool = True  # QK^T를 sqrt(d_k)로 나누는지
    pos: str = "sin"  # sin | learned | none
    # 학습
    vocab_size: int = 8000
    max_len: int = 64
    batch_size: int = 128
    epochs: int = 20
    warmup_steps: int = 2000
    lr_factor: float = 1.0
    seed: int = 0


cfg = Config()
if SMOKE:
    cfg = replace(cfg, epochs=1, vocab_size=2000)
print(cfg)

# %% [markdown]
# ## 1. 핵심 모듈
#
# ### Scaled dot-product attention
#
# $$ \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V $$
#
# `scale=False`로 두면 $$\sqrt{d_k}$$로 나누지 않는다 (ablation용).

# %%
def scaled_dot_product_attention(Q, K, V, mask=None, scale=True):
    """Q: (..., n_q, d_k), K: (..., n_k, d_k), V: (..., n_k, d_v), mask: n_q×n_k로 broadcast되는 bool (True=가림)."""
    d_k = Q.size(-1)
    scores = Q @ K.transpose(-2, -1)  # (..., n_q, n_k)
    if scale:
        scores = scores / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask, float("-inf"))
    attn = scores.softmax(dim=-1)  # (..., n_q, n_k), 행마다 합이 1
    return attn @ V, attn  # (..., n_q, d_v)


# %% [markdown]
# ### Multi-head attention
#
# $$ \text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h)W^O, \quad \text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V) $$
#
# $$W_i^Q$$ h개를 따로 두는 대신, $$d_\text{model} \times d_\text{model}$$ 행렬 하나로 한 번에 projection한 뒤 head 축으로 나눈다. 수학적으로 같다.

# %%
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, h, p_drop=0.1, scale=True):
        super().__init__()
        assert d_model % h == 0
        self.h, self.d_k, self.scale = h, d_model // h, scale
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(p_drop)
        self.attn = None  # 시각화용으로 마지막 attention 가중치를 보관

    def split_heads(self, x):  # (B, n, d_model) -> (B, h, n, d_k)
        B, n, _ = x.shape
        return x.view(B, n, self.h, self.d_k).transpose(1, 2)

    def forward(self, q, k, v, mask=None):
        Q = self.split_heads(self.W_Q(q))  # (B, h, n_q, d_k)
        K = self.split_heads(self.W_K(k))  # (B, h, n_k, d_k)
        V = self.split_heads(self.W_V(v))  # (B, h, n_k, d_k)
        out, attn = scaled_dot_product_attention(Q, K, V, mask, self.scale)  # (B, h, n_q, d_k)
        self.attn = attn.detach()
        B, _, n_q, _ = out.shape
        out = out.transpose(1, 2).reshape(B, n_q, self.h * self.d_k)  # concat: (B, n_q, d_model)
        return self.W_O(self.drop(out))


# %% [markdown]
# ### Position-wise FFN과 positional encoding
#
# $$ \text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2 $$
#
# $$ PE_{(pos, 2i)} = \sin(pos / 10000^{2i/d_\text{model}}), \quad PE_{(pos, 2i+1)} = \cos(pos / 10000^{2i/d_\text{model}}) $$

# %%
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, p_drop=0.1):
        super().__init__()
        self.W_1 = nn.Linear(d_model, d_ff)
        self.W_2 = nn.Linear(d_ff, d_model)
        self.drop = nn.Dropout(p_drop)

    def forward(self, x):  # (B, n, d_model)
        return self.W_2(self.drop(F.relu(self.W_1(x))))  # (B, n, d_model)


def sinusoidal_pe(max_len, d_model):
    pos = torch.arange(max_len).unsqueeze(1)  # (max_len, 1)
    i2 = torch.arange(0, d_model, 2)  # 2i = 0, 2, 4, ...
    freq = torch.exp(-math.log(10000.0) * i2 / d_model)  # 1 / 10000^(2i/d_model)
    pe = torch.zeros(max_len, d_model)
    pe[:, 0::2] = torch.sin(pos * freq)
    pe[:, 1::2] = torch.cos(pos * freq)
    return pe  # (max_len, d_model)


# %% [markdown]
# ### Encoder / Decoder layer (Post-LN)
#
# 논문은 각 sub-layer 출력을 $$\text{LayerNorm}(x + \text{Dropout}(\text{Sublayer}(x)))$$로 만든다 (Post-LN).

# %%
class EncoderLayer(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.self_attn = MultiHeadAttention(c.d_model, c.h, c.p_drop, c.scale)
        self.ffn = FeedForward(c.d_model, c.d_ff, c.p_drop)
        self.ln1, self.ln2 = nn.LayerNorm(c.d_model), nn.LayerNorm(c.d_model)
        self.drop = nn.Dropout(c.p_drop)

    def forward(self, x, src_mask):  # x: (B, n_src, d_model)
        x = self.ln1(x + self.drop(self.self_attn(x, x, x, src_mask)))
        return self.ln2(x + self.drop(self.ffn(x)))


class DecoderLayer(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.self_attn = MultiHeadAttention(c.d_model, c.h, c.p_drop, c.scale)
        self.cross_attn = MultiHeadAttention(c.d_model, c.h, c.p_drop, c.scale)
        self.ffn = FeedForward(c.d_model, c.d_ff, c.p_drop)
        self.ln1, self.ln2, self.ln3 = (nn.LayerNorm(c.d_model) for _ in range(3))
        self.drop = nn.Dropout(c.p_drop)

    def forward(self, y, memory, tgt_mask, src_mask):  # y: (B, n_tgt, d_model), memory: (B, n_src, d_model)
        y = self.ln1(y + self.drop(self.self_attn(y, y, y, tgt_mask)))  # masked self-attention
        y = self.ln2(y + self.drop(self.cross_attn(y, memory, memory, src_mask)))  # Q: decoder, K/V: encoder
        return self.ln3(y + self.drop(self.ffn(y)))


# %% [markdown]
# ### 전체 Transformer
#
# - 인코더 입력 임베딩, 디코더 입력 임베딩, 출력 projection이 **같은 가중치 행렬 하나**를 공유한다 (논문 3.4절, 공유 BPE 어휘).
# - 임베딩에는 $$\sqrt{d_\text{model}}$$을 곱한다.

# %%
class Transformer(nn.Module):
    def __init__(self, c: Config, pad_id: int):
        super().__init__()
        self.c, self.pad_id = c, pad_id
        self.emb = nn.Embedding(c.vocab_size, c.d_model, padding_idx=pad_id)
        if c.pos == "sin":
            self.register_buffer("pe", sinusoidal_pe(c.max_len + 2, c.d_model), persistent=False)
        elif c.pos == "learned":
            self.pe = nn.Parameter(torch.randn(c.max_len + 2, c.d_model) * 0.02)
        else:
            self.pe = None
        self.encoder = nn.ModuleList(EncoderLayer(c) for _ in range(c.N))
        self.decoder = nn.ModuleList(DecoderLayer(c) for _ in range(c.N))
        self.drop = nn.Dropout(c.p_drop)
        for name, p in self.named_parameters():
            if p.dim() > 1 and "emb" not in name and name != "pe":
                nn.init.xavier_uniform_(p)
        nn.init.normal_(self.emb.weight, std=c.d_model**-0.5)  # sqrt(d_model)를 곱하면 분산 1

    def embed(self, ids):  # (B, n) -> (B, n, d_model)
        x = self.emb(ids) * math.sqrt(self.c.d_model)
        if self.pe is not None:
            x = x + self.pe[: ids.size(1)]
        return self.drop(x)

    def encode(self, src):
        src_mask = (src == self.pad_id)[:, None, None, :]  # (B, 1, 1, n_src): pad key를 가림
        x = self.embed(src)
        for layer in self.encoder:
            x = layer(x, src_mask)
        return x, src_mask

    def decode(self, tgt_in, memory, src_mask):
        n = tgt_in.size(1)
        causal = torch.triu(torch.ones(n, n, dtype=torch.bool, device=tgt_in.device), diagonal=1)  # 미래 위치 True
        tgt_mask = causal[None, None] | (tgt_in == self.pad_id)[:, None, None, :]  # (B, 1, n, n)
        y = self.embed(tgt_in)
        for layer in self.decoder:
            y = layer(y, memory, tgt_mask, src_mask)
        return y @ self.emb.weight.T  # 출력 projection도 임베딩 가중치 공유: (B, n, vocab)

    def forward(self, src, tgt_in):
        memory, src_mask = self.encode(src)
        return self.decode(tgt_in, memory, src_mask)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# %% [markdown]
# ## 2. 저차원 예시로 검증
#
# 블로그 글의 "숫자로 확인하기"와 같은 값이다. 글에 적힌 숫자는 이 셀의 출력이다.

# %%
torch.set_printoptions(precision=3, sci_mode=False)
toy = {}

# (1) n=3 토큰, d_k=2
Q = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
K = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
V = torch.tensor([[1.0, 0.0], [0.0, 2.0], [3.0, 3.0]])
scores = Q @ K.T
out, attn = scaled_dot_product_attention(Q, K, V)
print("QK^T =\n", scores)
print("QK^T / sqrt(2) =\n", scores / math.sqrt(2))
print("attention =\n", attn)
print("output =\n", out)
assert torch.allclose(attn.sum(-1), torch.ones(3))
toy["attention"] = {"scores": scores.tolist(), "scaled": (scores / math.sqrt(2)).tolist(),
                    "attn": attn.tolist(), "out": out.tolist()}

# (2) 왜 sqrt(d_k)로 나누나: q, k 성분이 평균 0, 분산 1이면 q·k의 분산은 d_k
g = torch.Generator().manual_seed(0)
toy["scaling"] = []
for d_k in [4, 64, 512]:
    q = torch.randn(10000, d_k, generator=g)
    k = torch.randn(10000, 10, d_k, generator=g)  # query마다 key 10개
    dots = torch.einsum("bd,bnd->bn", q, k)  # (10000, 10)
    p_raw, p_scaled = dots.softmax(-1), (dots / math.sqrt(d_k)).softmax(-1)
    row = {"d_k": d_k, "var_dot": dots.var().item(),
           "max_p_unscaled": p_raw.max(-1).values.mean().item(),
           "max_p_scaled": p_scaled.max(-1).values.mean().item()}
    # softmax의 Jacobian 크기: diag(p) - pp^T 의 Frobenius norm 평균 (gradient가 얼마나 흐르는지)
    for name, p in [("unscaled", p_raw[:2000]), ("scaled", p_scaled[:2000])]:
        J = torch.diag_embed(p) - p.unsqueeze(-1) * p.unsqueeze(-2)
        row[f"jac_norm_{name}"] = J.flatten(1).norm(dim=-1).mean().item()
    toy["scaling"].append(row)
    print({k_: round(v, 4) if isinstance(v, float) else v for k_, v in row.items()})

# (3) positional encoding: d_model=4, pos=0..3
pe4 = sinusoidal_pe(4, 4)
print("PE (pos=0..3, d_model=4) =\n", pe4)
# PE(pos+k)는 PE(pos)의 선형 변환: 주파수 w마다 2x2 회전 행렬
d, k_off = 4, 1
M = torch.zeros(d, d)
for i in range(d // 2):
    w = 1 / 10000 ** (2 * i / d)
    c_, s_ = math.cos(w * k_off), math.sin(w * k_off)
    M[2 * i: 2 * i + 2, 2 * i: 2 * i + 2] = torch.tensor([[c_, s_], [-s_, c_]])
pred = pe4[:3] @ M.T  # 각 행 PE(pos)에 M을 곱함
print("M (k=1) =\n", M)
print("M @ PE(pos) =\n", pred, "\nPE(pos+1) =\n", pe4[1:])
assert torch.allclose(pred, pe4[1:], atol=1e-6)
toy["pe"] = {"pe": pe4.tolist(), "M": M.tolist()}

# (4) causal mask: n=4
s4 = torch.tensor([[2.0, 1.0, 0.5, 3.0], [1.0, 2.0, 0.0, 1.0], [0.5, 1.0, 2.0, 0.0], [1.0, 0.0, 1.0, 2.0]])
causal4 = torch.triu(torch.ones(4, 4, dtype=torch.bool), diagonal=1)
p_nomask, p_mask = s4.softmax(-1), s4.masked_fill(causal4, float("-inf")).softmax(-1)
print("mask 없음 =\n", p_nomask, "\ncausal mask =\n", p_mask)
toy["causal"] = {"scores": s4.tolist(), "attn_nomask": p_nomask.tolist(), "attn_mask": p_mask.tolist()}

# (5) multi-head의 shape과 비용: h를 바꿔도 파라미터 수는 같다
for h in [1, 8]:
    mha = MultiHeadAttention(256, h)
    x = torch.randn(2, 5, 256)
    y = mha(x, x, x)
    print(f"h={h}: split {tuple(mha.split_heads(x).shape)}, out {tuple(y.shape)}, params {count_params(mha):,}")
    assert y.shape == x.shape
toy["mha_params"] = count_params(MultiHeadAttention(256, 8))
(OUT / "toy.json").write_text(json.dumps(toy, indent=2), encoding="utf-8")

# %% [markdown]
# ## 3. 데이터: Multi30k 영어→독일어
#
# 논문은 WMT14 En-De(450만 쌍)와 BPE 37k 공유 어휘를 쓴다. 여기서는 Multi30k(2.9만 쌍)에 BPE 8k 공유 어휘를 학습한다.

# %%
from datasets import load_dataset
from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers

ds = load_dataset("bentrevett/multi30k", cache_dir=str(DATA / "hf"))
if SMOKE:
    ds["train"] = ds["train"].select(range(2000))
    ds["test"] = ds["test"].select(range(100))
SPECIALS = ["<pad>", "<unk>", "<bos>", "<eos>"]

tok = Tokenizer(models.BPE(unk_token="<unk>"))
tok.normalizer = normalizers.NFKC()
tok.pre_tokenizer = pre_tokenizers.Metaspace()  # 공백 정보를 보존해 디코딩 후 원문 복원이 가능
tok.decoder = decoders.Metaspace()
trainer = trainers.BpeTrainer(vocab_size=cfg.vocab_size, special_tokens=SPECIALS, min_frequency=2)
tok.train_from_iterator((s for ex in ds["train"] for s in (ex["en"], ex["de"])), trainer)
PAD, UNK, BOS, EOS = (tok.token_to_id(s) for s in SPECIALS)
cfg = replace(cfg, vocab_size=tok.get_vocab_size())
print("vocab:", cfg.vocab_size)


def encode_split(split):
    src = [e.ids[: cfg.max_len] for e in tok.encode_batch(split["en"])]
    tgt = [[BOS] + e.ids[: cfg.max_len - 2] + [EOS] for e in tok.encode_batch(split["de"])]
    return src, tgt


data = {name: encode_split(ds[name]) for name in ["train", "validation", "test"]}
print({k: len(v[0]) for k, v in data.items()}, "예시:", tok.encode(ds["train"][0]["en"]).tokens[:12])


def pad_batch(seqs):
    n = max(len(s) for s in seqs)
    return torch.tensor([s + [PAD] * (n - len(s)) for s in seqs], dtype=torch.long)


def batches(src, tgt, batch_size, shuffle, rng=None):
    idx = list(range(len(src)))
    if shuffle:
        rng.shuffle(idx)
        # 길이가 비슷한 문장끼리 묶어 padding을 줄인다 (논문도 길이로 batch를 구성)
        chunks = [sorted(idx[i: i + batch_size * 50], key=lambda j: len(src[j])) for i in range(0, len(idx), batch_size * 50)]
        bs = [c[i: i + batch_size] for c in chunks for i in range(0, len(c), batch_size)]
        rng.shuffle(bs)
    else:
        bs = [idx[i: i + batch_size] for i in range(0, len(idx), batch_size)]
    for b in bs:
        yield pad_batch([src[j] for j in b]).to(device), pad_batch([tgt[j] for j in b]).to(device)


# %% [markdown]
# ## 4. 학습
#
# 학습률 schedule (논문 식 3): $$ lrate = d_\text{model}^{-0.5} \cdot \min(step^{-0.5},\ step \cdot warmup^{-1.5}) $$
#
# Adam $$\beta_1=0.9, \beta_2=0.98, \epsilon=10^{-9}$$, label smoothing 0.1. 논문과 같다.

# %%
def lr_at(step, c: Config):
    step = max(step, 1)
    return c.lr_factor * c.d_model**-0.5 * min(step**-0.5, step * c.warmup_steps**-1.5)


@torch.no_grad()
def evaluate_loss(model, split):
    model.eval()
    total, count = 0.0, 0
    for src, tgt in batches(*data[split], 256, shuffle=False):
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(src, tgt[:, :-1])
        loss = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), tgt[:, 1:].reshape(-1),
                               ignore_index=PAD, reduction="sum")
        total += loss.item()
        count += (tgt[:, 1:] != PAD).sum().item()
    return total / count


@torch.no_grad()
def greedy_decode(model, src, max_extra=50):
    model.eval()
    memory, src_mask = model.encode(src)
    ys = torch.full((src.size(0), 1), BOS, dtype=torch.long, device=device)
    done = torch.zeros(src.size(0), dtype=torch.bool, device=device)
    for _ in range(min(src.size(1) + max_extra, cfg.max_len + 1)):
        logits = model.decode(ys, memory, src_mask)[:, -1]
        nxt = logits.argmax(-1).masked_fill(done, PAD)
        ys = torch.cat([ys, nxt[:, None]], dim=1)
        done |= nxt == EOS
        if done.all():
            break
    return ys[:, 1:]


def ids_to_text(ids):
    ids = [i for i in ids if i not in (PAD, BOS)]
    if EOS in ids:
        ids = ids[: ids.index(EOS)]
    return tok.decode(ids)


def bleu(model, split="test"):
    import sacrebleu
    hyps = []
    for src, _ in batches(*data[split], 256, shuffle=False):
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            out = greedy_decode(model, src)
        hyps += [ids_to_text(o.tolist()) for o in out]
    refs = list(ds[split]["de"])
    return sacrebleu.corpus_bleu(hyps, [refs]).score, hyps


def train(c: Config, name: str):
    seed_everything(c.seed)
    rng = random.Random(c.seed)
    model = Transformer(c, PAD).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_at(s + 1, c))
    hist = {"step": [], "train_loss": [], "val_loss": [], "lr": []}
    step, t0 = 0, time.time()
    for epoch in range(c.epochs):
        model.train()
        run_loss, run_n = 0.0, 0
        for src, tgt in batches(*data["train"], c.batch_size, shuffle=True, rng=rng):
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = model(src, tgt[:, :-1])  # teacher forcing: 정답을 한 칸 밀어 입력
            loss = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), tgt[:, 1:].reshape(-1),
                                   ignore_index=PAD, label_smoothing=c.eps_ls)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            step += 1
            run_loss, run_n = run_loss + loss.item(), run_n + 1
        val = evaluate_loss(model, "validation")
        hist["step"].append(step)
        hist["train_loss"].append(run_loss / run_n)
        hist["val_loss"].append(val)
        hist["lr"].append(sched.get_last_lr()[0])
        print(f"[{name}] epoch {epoch + 1:2d} step {step:5d} train {run_loss / run_n:.3f} "
              f"val {val:.3f} (ppl {math.exp(val):.2f}) {time.time() - t0:.0f}s", flush=True)
    score, hyps = bleu(model, "test")
    print(f"[{name}] test BLEU {score:.2f} | params {count_params(model):,} | {time.time() - t0:.0f}s", flush=True)
    return model, {"config": asdict(c), "history": hist, "bleu": score, "val_ppl": math.exp(hist["val_loss"][-1]),
                   "params": count_params(model), "train_time_sec": round(time.time() - t0, 1),
                   "samples": [{"src": ds["test"][i]["en"], "ref": ds["test"][i]["de"], "hyp": hyps[i]} for i in range(5)]}


# %% [markdown]
# ## 5. Ablation 실험
#
# 기본 실행은 논문 Table 3의 재현(리뷰 글)만 돌린다. 논문에 없는 실험(심화 글)은 `DEEP_DIVE` 환경 변수로 켠다.
# 예: `DEEP_DIVE=transformer-without-scaling OUTDIR=outputs_dd python attention-is-all-you-need.py`
#
# | 이름 | 바꾼 것 | 쓰는 글 |
# |------|---------|---------|
# | base | h=8, d_k=32, sinusoidal PE, scaling | 리뷰 (Table 3 base) |
# | h1 / h4 / h32 | head 수 (d_k = 256/h, 계산량 고정) | 리뷰 (Table 3 (A)) |
# | learned_pe | 학습되는 위치 임베딩 | 리뷰 (Table 3 (E)) |
# | no_scale, no_scale_h1 | sqrt(d_k) 나눗셈 제거 (d_k=32, 256) | 심화 `transformer-without-scaling` |
# | no_pe | 위치 정보 없음 | 심화 `transformer-without-position` |

# %%
PAPER_VARIANTS = {
    "base": {},
    "h1": {"h": 1},
    "h4": {"h": 4},
    "h32": {"h": 32},
    "learned_pe": {"pos": "learned"},
}
DEEP_DIVE_VARIANTS = {  # 비교용 기준 설정을 함께 돌린다
    "transformer-without-scaling": {"base": {}, "no_scale": {"scale": False},
                                    "h1": {"h": 1}, "no_scale_h1": {"scale": False, "h": 1}},
    "transformer-without-position": {"base": {}, "learned_pe": {"pos": "learned"}, "no_pe": {"pos": "none"}},
}
DEEP_DIVE = os.environ.get("DEEP_DIVE", "")
variants = DEEP_DIVE_VARIANTS[DEEP_DIVE] if DEEP_DIVE else PAPER_VARIANTS
if SMOKE:  # 선택된 묶음에서 앞의 2개만
    variants = dict(list(variants.items())[:2])
SEEDS = [0] if SMOKE else [int(s) for s in os.environ.get("SEEDS", "0,1,2").split(",")]

results, start = {}, time.time()
base_model = None
for name, change in variants.items():
    for seed in SEEDS:
        model, r = train(replace(cfg, seed=seed, **change), f"{name}/s{seed}")
        results.setdefault(name, []).append(r)
        if name == "base" and seed == SEEDS[0]:
            base_model = model
        else:
            del model
        torch.cuda.empty_cache()
total_time = time.time() - start

# %% [markdown]
# ## 6. 결과 저장

# %%
summary = {}
for name, rs in results.items():
    b = [r["bleu"] for r in rs]
    p = [r["val_ppl"] for r in rs]
    summary[name] = {"bleu_mean": float(np.mean(b)), "bleu_std": float(np.std(b)), "bleu": b,
                     "val_ppl_mean": float(np.mean(p)), "val_ppl": p, "params": rs[0]["params"],
                     "train_time_sec_mean": float(np.mean([r["train_time_sec"] for r in rs]))}
metrics = {
    "config": asdict(cfg),
    "seeds": SEEDS,
    "results": summary,
    "history": {name: [r["history"] for r in rs] for name, rs in results.items()},
    "samples": results["base"][0]["samples"],
    "env": {"device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
            "torch": torch.__version__, "total_time_sec": round(total_time, 1), "smoke": SMOKE},
}
(OUT / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
for name, s in summary.items():
    print(f"{name:12s} BLEU {s['bleu_mean']:.2f} ± {s['bleu_std']:.2f}  val ppl {s['val_ppl_mean']:.2f}  params {s['params']:,}")

# %% [markdown]
# ## 7. 학습된 모델의 attention 들여다보기
#
# base 모델의 마지막 decoder layer의 encoder-decoder attention (head 평균)을 예문 하나에 대해 본다.

# %%
sent = "A man in an orange hat is staring at something."
src = torch.tensor([tok.encode(sent).ids], device=device)
out = greedy_decode(base_model, src)
tgt_ids = [BOS] + [i for i in out[0].tolist() if i != PAD]
with torch.no_grad():
    base_model(src, torch.tensor([tgt_ids[:-1]], device=device))
cross = base_model.decoder[-1].cross_attn.attn[0].float().mean(0).cpu().numpy()  # (n_tgt, n_src)
src_toks = tok.encode(sent).tokens
tgt_toks = [tok.id_to_token(i) for i in tgt_ids[1:]]
print("번역:", ids_to_text(out[0].tolist()))
(OUT / "cross_attention.json").write_text(json.dumps(
    {"src": sent, "src_tokens": src_toks, "tgt_tokens": tgt_toks, "attn": cross.tolist(),
     "translation": ids_to_text(out[0].tolist())}, ensure_ascii=False, indent=2), encoding="utf-8")

fig, ax = plt.subplots(figsize=(7, 5))
ax.imshow(cross, cmap="Blues")
ax.set_xticks(range(len(src_toks)), src_toks, rotation=60, fontsize=8)
ax.set_yticks(range(len(tgt_toks)), tgt_toks, fontsize=8)
ax.set_title("encoder-decoder attention (last layer, head mean)")
fig.tight_layout()
fig.savefig(OUT / "cross-attention.png", dpi=150)
plt.show()

# %% [markdown]
# ## 8. 결과 그래프

# %%
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for name, hs in metrics["history"].items():
    h = hs[0]
    axes[0].plot(h["step"], h["val_loss"], label=name)
axes[0].set_xlabel("step")
axes[0].set_ylabel("validation loss")
axes[0].legend(fontsize=8)
names = list(summary)
axes[1].bar(names, [summary[n]["bleu_mean"] for n in names], yerr=[summary[n]["bleu_std"] for n in names])
axes[1].set_ylabel("test BLEU (greedy)")
axes[1].tick_params(axis="x", rotation=45)
fig.tight_layout()
fig.savefig(OUT / "ablation.png", dpi=150)
plt.show()
