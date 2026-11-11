# %% [markdown]
# # ViT (An Image is Worth 16x16 Words) — 미니 구현과 재현 실험
#
# - 블로그 글: https://Liliid.github.io/posts/vit/
# - 논문: https://arxiv.org/abs/2010.11929
# - 앞 글(Transformer): https://Liliid.github.io/posts/attention-is-all-you-need/
#
# 이 노트북 하나로 ViT 구현 → 저차원 예시 검증 → CIFAR-10에서 ViT vs ResNet 데이터 규모 실험, 위치 임베딩 ablation, 내부 분석까지 실행된다.
# Colab에서는 `런타임 > 런타임 유형 변경 > T4 GPU`를 선택한다. T4에서는 전체 실험에 3~4시간 걸리므로 `runs`를 줄여서 돌리는 것을 권한다.

# %%
# Colab 기본 환경에 없는 패키지만 설치한다 (로컬에서 .py로 실행하면 주석이라 무시된다)
# !pip install -q datasets

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

SMOKE = os.environ.get("SMOKE") == "1"
OUT = Path(os.environ.get("OUTDIR", "outputs"))
OUT.mkdir(exist_ok=True)
DATA = Path("data")
DATA.mkdir(exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device, torch.cuda.get_device_name(0) if device.type == "cuda" else "")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class Config:
    model: str = "vit"  # vit | resnet
    # ViT (논문 ViT-Base: L=12, D=768, MLP 3072, heads 12, patch 16 → 로컬 GPU에서 돌릴 수 있는 ViT-Small급)
    img_size: int = 32
    patch: int = 4  # 32/4 = 8 → 8x8 = 64 patch
    D: int = 384
    L: int = 8
    heads: int = 6  # head당 d_k = 64 (논문 ViT-B와 같음)
    mlp: int = 1536
    p_drop: float = 0.0
    pos: str = "1d"  # 1d | 2d | none
    # 학습 (논문처럼 두 모델 모두 같은 Adam 계열 optimizer와 schedule, learning rate만 모델별로)
    data_frac: float = 1.0
    steps: int = 7800  # 전체 데이터 기준 40 epoch (batch 256)
    batch_size: int = 256
    lr: float = 2.5e-4  # ViT: DeiT 규칙 5e-4 × batch/512. ResNet은 실험 셀에서 1e-3으로 덮어쓴다
    weight_decay: float = 0.05
    warmup: int = 500
    label_smoothing: float = 0.1
    eval_every: int = 390
    seed: int = 0


cfg = Config()
if SMOKE:
    cfg = replace(cfg, steps=30, eval_every=15, warmup=5)
print(cfg)

# %% [markdown]
# ## 1. 핵심 모듈
#
# ### Patch embedding (논문 식 1)
#
# $$ z_0 = [x_\text{class};\ x_p^1 E;\ x_p^2 E;\ \cdots;\ x_p^N E] + E_\text{pos}, \quad E \in \mathbb{R}^{(P^2 C) \times D},\ E_\text{pos} \in \mathbb{R}^{(N+1) \times D} $$
#
# 이미지 $$(C, H, W)$$를 $$P \times P$$ patch $$N = HW/P^2$$개로 자르고, 각 patch를 $$P^2 C$$ 벡터로 펼쳐 선형 변환한다.

# %%
def patchify(x, P):
    """x: (B, C, H, W) -> (B, N, P*P*C). patch는 행 우선(raster) 순서, 각 patch 안은 (P, P, C) 순서로 펼친다."""
    B, C, H, W = x.shape
    x = x.reshape(B, C, H // P, P, W // P, P)  # (B, C, H/P, P, W/P, P)
    x = x.permute(0, 2, 4, 3, 5, 1)  # (B, H/P, W/P, P, P, C)
    return x.reshape(B, (H // P) * (W // P), P * P * C)  # (B, N, P²C)


class PatchEmbedding(nn.Module):
    def __init__(self, c: Config, in_ch=3):
        super().__init__()
        self.P = c.patch
        self.grid = c.img_size // c.patch
        self.N = self.grid**2
        self.E = nn.Linear(c.patch * c.patch * in_ch, c.D)  # 식 1의 E
        self.x_class = nn.Parameter(torch.zeros(1, 1, c.D))  # 학습되는 [class] 토큰
        self.pos = c.pos
        if c.pos == "1d":  # 기본: 위치 0..N 각각에 벡터 하나 (2D 구조 정보 없음)
            self.E_pos = nn.Parameter(torch.randn(1, self.N + 1, c.D) * 0.02)
        elif c.pos == "2d":  # 부록 D.4: 행/열 임베딩(각 D/2)을 이어 붙임
            self.E_row = nn.Parameter(torch.randn(self.grid, c.D // 2) * 0.02)
            self.E_col = nn.Parameter(torch.randn(self.grid, c.D // 2) * 0.02)
            self.E_cls = nn.Parameter(torch.randn(1, 1, c.D) * 0.02)

    def pos_embedding(self):  # (1, N+1, D)
        if self.pos == "1d":
            return self.E_pos
        if self.pos == "2d":
            g = self.grid
            rc = torch.cat([self.E_row[:, None, :].expand(g, g, -1), self.E_col[None, :, :].expand(g, g, -1)], -1)
            return torch.cat([self.E_cls, rc.reshape(1, g * g, -1)], dim=1)
        return None

    def forward(self, x):  # x: (B, C, H, W)
        z = self.E(patchify(x, self.P))  # (B, N, D)
        z = torch.cat([self.x_class.expand(z.size(0), -1, -1), z], dim=1)  # (B, N+1, D)
        pe = self.pos_embedding()
        return z if pe is None else z + pe


# %% [markdown]
# ### Encoder block (논문 식 2, 3): Pre-LN
#
# $$ z'_\ell = \text{MSA}(\text{LN}(z_{\ell-1})) + z_{\ell-1}, \qquad z_\ell = \text{MLP}(\text{LN}(z'_\ell)) + z'_\ell $$
#
# 원래 Transformer(Post-LN: `LN(x + Sublayer(x))`)와 달리 LayerNorm이 sub-layer **앞**에 온다. MSA는 앞 글에서 구현한 multi-head attention과 같다.

# %%
class MSA(nn.Module):
    def __init__(self, D, heads, p_drop=0.0):
        super().__init__()
        self.h, self.d_k = heads, D // heads
        self.W_QKV = nn.Linear(D, 3 * D)  # W^Q, W^K, W^V를 한 번에
        self.W_O = nn.Linear(D, D)
        self.drop = nn.Dropout(p_drop)
        self.keep_attn = False
        self.attn = None

    def forward(self, z):  # (B, n, D)
        B, n, D = z.shape
        qkv = self.W_QKV(z).view(B, n, 3, self.h, self.d_k).permute(2, 0, 3, 1, 4)  # (3, B, h, n, d_k)
        Q, K, V = qkv[0], qkv[1], qkv[2]
        if self.keep_attn:  # 분석용: attention 가중치를 직접 계산해 보관
            attn = (Q @ K.transpose(-2, -1) / math.sqrt(self.d_k)).softmax(-1)  # (B, h, n, n)
            self.attn = attn.detach()
            out = attn @ V
        else:  # 학습용: 같은 연산의 빠른 커널
            out = F.scaled_dot_product_attention(Q, K, V)
        out = out.transpose(1, 2).reshape(B, n, D)  # head concat: (B, n, D)
        return self.drop(self.W_O(out))


class EncoderBlock(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(c.D), nn.LayerNorm(c.D)
        self.msa = MSA(c.D, c.heads, c.p_drop)
        self.mlp = nn.Sequential(nn.Linear(c.D, c.mlp), nn.GELU(), nn.Dropout(c.p_drop),
                                 nn.Linear(c.mlp, c.D), nn.Dropout(c.p_drop))

    def forward(self, z):
        z = z + self.msa(self.ln1(z))  # 식 2
        return z + self.mlp(self.ln2(z))  # 식 3


class ViT(nn.Module):
    def __init__(self, c: Config, num_classes=10):
        super().__init__()
        self.embed = PatchEmbedding(c)
        self.blocks = nn.ModuleList(EncoderBlock(c) for _ in range(c.L))
        self.ln = nn.LayerNorm(c.D)
        self.head = nn.Linear(c.D, num_classes)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            nn.init.zeros_(m.bias)

    def forward(self, x):  # (B, 3, 32, 32)
        z = self.embed(x)  # (B, N+1, D)
        for blk in self.blocks:
            z = blk(z)
        y = self.ln(z[:, 0])  # 식 4: [class] 토큰 위치의 출력만 쓴다 (B, D)
        return self.head(y)  # (B, num_classes)


# %% [markdown]
# ### 비교 대상: ResNet-18
#
# 논문은 BiT ResNet(GroupNorm + weight standardization)과 비교한다. 여기서는 표준 CIFAR용 ResNet-18(채널 64-128-256-512, BatchNorm)을 쓴다.
# 파라미터 수를 억지로 맞추지 않고, 두 모델 모두 로컬 GPU(16GB)에서 돌릴 수 있는 대표적인 크기를 고른다.

# %%
class BasicBlock(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.short = nn.Sequential()
        if stride != 1 or cin != cout:
            self.short = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        return F.relu(self.bn2(self.conv2(out)) + self.short(x))


class ResNet18(nn.Module):
    def __init__(self, width=64, num_classes=10):
        super().__init__()
        w = [width, width * 2, width * 4, width * 8]
        self.stem = nn.Sequential(nn.Conv2d(3, w[0], 3, 1, 1, bias=False), nn.BatchNorm2d(w[0]), nn.ReLU())
        layers, cin = [], w[0]
        for i, cout in enumerate(w):
            layers += [BasicBlock(cin, cout, 1 if i == 0 else 2), BasicBlock(cout, cout, 1)]
            cin = cout
        self.layers = nn.Sequential(*layers)
        self.fc = nn.Linear(cin, num_classes)

    def forward(self, x):
        return self.fc(F.adaptive_avg_pool2d(self.layers(self.stem(x)), 1).flatten(1))


def build(c: Config):
    return ViT(c) if c.model == "vit" else ResNet18()


def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


print(f"ViT params: {count_params(ViT(cfg)):,} | ResNet18 params: {count_params(ResNet18()):,}")

# %% [markdown]
# ## 2. 저차원 예시로 검증
#
# 블로그 글의 "숫자로 확인하기"와 같은 값이다.

# %%
torch.set_printoptions(precision=3, sci_mode=False)
toy = {}

# (1) 4x4 흑백 이미지(C=1)를 P=2로 patchify → N=4개 patch, 각 patch는 P²C=4차원
img = torch.arange(1, 17, dtype=torch.float32).reshape(1, 1, 4, 4)
patches = patchify(img, 2)
print("image =\n", img[0, 0])
print("patches (N=4, P²C=4) =\n", patches[0])
assert patches.shape == (1, 4, 4)
assert patches[0, 0].tolist() == [1, 2, 5, 6]  # 왼쪽 위 2x2 patch
toy["patchify"] = {"image": img[0, 0].tolist(), "patches": patches[0].tolist()}

# patch embedding: E (4 -> D=2)로 투영 + [class] 토큰 → (N+1, D) = (5, 2)
E_toy = torch.tensor([[0.1, 0.0], [0.0, 0.1], [0.1, 0.0], [0.0, 0.1]])  # (P²C=4, D=2)
z = patches[0] @ E_toy
print("x_p E =\n", z)
toy["patch_embed"] = {"E": E_toy.tolist(), "xpE": z.tolist()}

# (2) fine-tuning 해상도 변경: 2x2 grid의 위치 임베딩(D=1)을 3x3 grid로 2D 보간
pe_grid = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
pe_up = F.interpolate(pe_grid[None, None], size=(3, 3), mode="bilinear", align_corners=True)[0, 0]
print("2x2 위치 임베딩 =\n", pe_grid, "\n3x3로 보간 =\n", pe_up)
toy["pos_interp"] = {"before": pe_grid.tolist(), "after": pe_up.tolist()}

# (3) mean attention distance: 3x3 patch grid, patch 크기 4px, 가운데 patch가 query
coords = torch.stack(torch.meshgrid(torch.arange(3), torch.arange(3), indexing="ij"), -1).reshape(9, 2).float() * 4
dist = torch.cdist(coords, coords)  # (9, 9) 픽셀 거리
a_local = torch.tensor([0.05, 0.1, 0.05, 0.1, 0.4, 0.1, 0.05, 0.1, 0.05])  # 가까운 patch에 집중
a_global = torch.full((9,), 1 / 9)  # 고르게
for name, a in [("local", a_local), ("global", a_global)]:
    d_mean = (a * dist[4]).sum().item()
    print(f"{name}: 거리 {dist[4].reshape(3, 3).tolist()} → mean attention distance {d_mean:.2f}px")
    toy[f"attn_dist_{name}"] = {"attn": a.reshape(3, 3).tolist(), "dist": dist[4].reshape(3, 3).tolist(), "mean": d_mean}

# (4) 전체 shape 흐름
x = torch.randn(2, 3, 32, 32)
vit = ViT(cfg)
print("x:", tuple(x.shape), "-> patches:", tuple(patchify(x, 4).shape), "-> z0:", tuple(vit.embed(x).shape),
      "-> logits:", tuple(vit(x).shape))
toy["params"] = {"vit": count_params(vit), "resnet": count_params(ResNet18()),
                 "vit_2d": count_params(ViT(replace(cfg, pos="2d"))), "vit_nopos": count_params(ViT(replace(cfg, pos="none")))}
(OUT / "toy.json").write_text(json.dumps(toy, indent=2), encoding="utf-8")

# %% [markdown]
# ## 3. 데이터: CIFAR-10
#
# 전체 데이터를 GPU 메모리에 올리고 증강(random crop + 좌우 반전)도 GPU에서 한다. Windows/Colab에서 DataLoader 병목을 피하기 위해서다.

# %%
from datasets import load_dataset

cifar = load_dataset("uoft-cs/cifar10", cache_dir=str(DATA / "hf"))  # HF 미러가 원본 서버보다 빠르다


def to_tensors(split):
    arr = np.stack(cifar[split].with_format("numpy")["img"])  # (n, 32, 32, 3) uint8
    return torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous(), torch.tensor(cifar[split]["label"])


X_train, y_train = to_tensors("train")  # (50000, 3, 32, 32) uint8
X_test, y_test = to_tensors("test")
if SMOKE:
    X_train, y_train, X_test, y_test = X_train[:2000], y_train[:2000], X_test[:500], y_test[:500]
X_train, y_train, X_test, y_test = (t.to(device) for t in (X_train, y_train, X_test, y_test))
MEAN = torch.tensor([0.4914, 0.4822, 0.4465], device=device).view(1, 3, 1, 1)
STD = torch.tensor([0.2470, 0.2435, 0.2616], device=device).view(1, 3, 1, 1)
CLASSES = cifar["train"].features["label"].names
print("train", tuple(X_train.shape), "test", tuple(X_test.shape))


def normalize(x_uint8):
    return (x_uint8.float() / 255 - MEAN) / STD


def augment(x):  # x: (B, 3, 32, 32) float. random crop(padding 4) + horizontal flip
    B = x.size(0)
    x = F.pad(x, (4, 4, 4, 4), mode="reflect")
    i = torch.randint(0, 9, (B,), device=x.device)
    j = torch.randint(0, 9, (B,), device=x.device)
    rows = (i[:, None] + torch.arange(32, device=x.device))[:, None, :, None].expand(B, 3, 32, 40)
    x = x.gather(2, rows)
    cols = (j[:, None] + torch.arange(32, device=x.device))[:, None, None, :].expand(B, 3, 32, 32)
    x = x.gather(3, cols)
    flip = torch.rand(B, device=x.device) < 0.5
    return torch.where(flip[:, None, None, None], x.flip(3), x)


def subset_indices(frac, seed):
    """클래스 비율을 유지한 채 학습 데이터의 frac만 고른다. seed가 같으면 같은 부분집합."""
    g = torch.Generator().manual_seed(1000 + seed)
    y = y_train.cpu()
    keep = []
    for k in range(10):
        idx = (y == k).nonzero().flatten()
        keep.append(idx[torch.randperm(len(idx), generator=g)[: max(1, int(len(idx) * frac))]])
    return torch.cat(keep).to(device)


# %% [markdown]
# ## 4. 학습
#
# 두 모델 모두 같은 설정: AdamW(weight decay 0.05), linear warmup + cosine decay, label smoothing 0.1, 7,800 step(batch 256).
# learning rate만 모델별로 다르다. ViT는 2.5e-4(DeiT 규칙), ResNet은 1e-3. ViT에 1e-3을 쓰면 100% 데이터에서도 train 정확도가 65%에 머물렀다.
# 데이터 양이 달라도 **step 수를 같게** 해서, 차이가 학습량이 아니라 데이터 양에서 오도록 한다.

# %%
@torch.no_grad()
def evaluate(model):
    model.eval()
    correct = 0
    for i in range(0, len(X_test), 1000):
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(normalize(X_test[i: i + 1000]))
        correct += (logits.argmax(-1) == y_test[i: i + 1000]).sum().item()
    return correct / len(X_test)


def train(c: Config, name: str):
    seed_everything(c.seed)
    model = build(c).to(device)
    idx = subset_indices(c.data_frac, c.seed)
    decay = [p for n, p in model.named_parameters() if p.dim() > 1 and "E_pos" not in n and "x_class" not in n
             and "E_row" not in n and "E_col" not in n and "E_cls" not in n]
    no_decay = [p for n, p in model.named_parameters() if not any(p is q for q in decay)]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": c.weight_decay},
                             {"params": no_decay, "weight_decay": 0.0}], lr=c.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / c.warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / c.steps))))
    hist = {"step": [], "train_loss": [], "train_acc": [], "test_acc": []}
    t0, run_loss, run_correct, run_n = time.time(), 0.0, 0, 0
    for step in range(1, c.steps + 1):
        model.train()
        b = idx[torch.randint(0, len(idx), (c.batch_size,), device=device)]
        x, y = augment(normalize(X_train[b])), y_train[b]
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(x)
        loss = F.cross_entropy(logits.float(), y, label_smoothing=c.label_smoothing)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        run_loss += loss.item()
        run_correct += (logits.argmax(-1) == y).sum().item()
        run_n += 1
        if step % c.eval_every == 0 or step == c.steps:
            acc = evaluate(model)
            hist["step"].append(step)
            hist["train_loss"].append(run_loss / run_n)
            hist["train_acc"].append(run_correct / (run_n * c.batch_size))
            hist["test_acc"].append(acc)
            print(f"[{name}] step {step:5d} train loss {run_loss / run_n:.3f} acc {hist['train_acc'][-1]:.3f} "
                  f"test acc {acc:.4f} {time.time() - t0:.0f}s", flush=True)
            run_loss, run_correct, run_n = 0.0, 0, 0
    return model, {"config": asdict(c), "history": hist, "final_test_acc": hist["test_acc"][-1],
                   "best_test_acc": max(hist["test_acc"]), "params": count_params(model),
                   "n_train": len(idx), "train_time_sec": round(time.time() - t0, 1)}


# %% [markdown]
# ## 5. 실험
#
# | 이름 | 내용 | 논문 대응 |
# |------|------|-----------|
# | vit / resnet × 10%, 30%, 100% | 데이터 양에 따른 두 구조의 차이 | Figure 3, 4 (사전학습 데이터 규모) |
# | vit_pos2d / vit_nopos (100%) | 위치 임베딩 방식 | 부록 D.4, Table 8 |

#
# 심화 글 `vit-learning-rate-fairness`의 실험은 환경 변수로 재현한다 (기본값은 리뷰 설정).
# - `VIT_LR=1e-3 OUTDIR=outputs_lr1e-3 python vit.py` : ViT도 ResNet과 같은 lr로 전체 실행
# - `VIT_LR=5e-4 RUNS=vit_100 OUTDIR=outputs_lr5e-4 python vit.py` : ViT 100% 설정만 lr 진단

# %%
VIT_LR = float(os.environ.get("VIT_LR", cfg.lr))  # 기본: DeiT 규칙 2.5e-4
RESNET_LR = 1e-3
runs = {}
for frac in [0.1, 0.3, 1.0]:
    for model_name in ["vit", "resnet"]:
        runs[f"{model_name}_{int(frac * 100)}"] = {"model": model_name, "data_frac": frac,
                                                    "lr": RESNET_LR if model_name == "resnet" else VIT_LR}
runs["vit_pos2d_100"] = {"model": "vit", "pos": "2d", "lr": VIT_LR}
runs["vit_nopos_100"] = {"model": "vit", "pos": "none", "lr": VIT_LR}
if SMOKE:
    runs = {k: runs[k] for k in ("vit_100", "resnet_10")}
if os.environ.get("RUNS"):  # 일부 설정만 돌릴 때 (쉼표로 구분)
    runs = {k: v for k, v in runs.items() if k in os.environ["RUNS"].split(",")}
SEEDS = [0] if SMOKE else [int(s) for s in os.environ.get("SEEDS", "0").split(",")]

results, start, vit_full = {}, time.time(), None
for name, change in runs.items():
    for seed in SEEDS:
        model, r = train(replace(cfg, seed=seed, **change), f"{name}/s{seed}")
        results.setdefault(name, []).append(r)
        if name == "vit_100" and seed == SEEDS[0]:
            vit_full = model
        else:
            del model
        torch.cuda.empty_cache()
total_time = time.time() - start

# %% [markdown]
# ## 6. 결과 저장

# %%
summary = {}
for name, rs in results.items():
    acc = [r["final_test_acc"] for r in rs]
    summary[name] = {"final_acc_mean": float(np.mean(acc)), "final_acc_std": float(np.std(acc)), "final_acc": acc,
                     "best_acc": [r["best_test_acc"] for r in rs], "params": rs[0]["params"],
                     "n_train": rs[0]["n_train"], "train_time_sec_mean": float(np.mean([r["train_time_sec"] for r in rs]))}
metrics = {
    "config": asdict(cfg), "seeds": SEEDS, "results": summary,
    "history": {name: [r["history"] for r in rs] for name, rs in results.items()},
    "env": {"device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
            "torch": torch.__version__, "total_time_sec": round(total_time, 1), "smoke": SMOKE},
}
(OUT / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
for name, s in summary.items():
    print(f"{name:15s} n_train {s['n_train']:6d}  test acc {s['final_acc_mean']:.4f} ± {s['final_acc_std']:.4f}  params {s['params']:,}")

# %% [markdown]
# ## 7. 학습된 ViT 들여다보기 (논문 Figure 7 재현)
#
# - **위치 임베딩 유사도**: 각 patch의 위치 임베딩과 나머지 patch 위치 임베딩의 cosine similarity를 8x8 grid로 본다.
# - **Mean attention distance**: attention 가중치로 가중 평균한 query-key patch 사이 픽셀 거리. CNN의 receptive field 크기에 해당한다.

# %%
@torch.no_grad()
def analyze(model, n_images=1000):
    model.eval()
    pe = model.embed.pos_embedding()[0, 1:].float()  # (N, D), [class] 제외
    pe = F.normalize(pe, dim=-1)
    sim = (pe @ pe.T).cpu().numpy()  # (N, N)
    g, P = model.embed.grid, model.embed.P
    coords = torch.stack(torch.meshgrid(torch.arange(g), torch.arange(g), indexing="ij"), -1).reshape(-1, 2).float()
    dist = (torch.cdist(coords, coords) * P).to(device)  # (N, N) 픽셀 거리
    for blk in model.blocks:
        blk.msa.keep_attn = True
    model(normalize(X_test[:n_images]).float())
    mad, cls_mass = [], []  # 둘 다 (L, heads)
    for blk in model.blocks:
        attn = blk.msa.attn.float()  # (B, h, N+1, N+1)
        cls_mass.append(attn[:, :, 1:, 0].mean(dim=(0, 2)).cpu().tolist())  # patch query가 [class]에 준 비율
        a = attn[:, :, 1:, 1:]  # patch query → patch key (B, h, N, N)
        mass = a.sum(-1, keepdim=True)  # [class]에 거의 다 준 query는 patch 쪽 합이 0에 가깝다
        d_q = (a / mass.clamp_min(1e-6) * dist).sum(-1)  # query별 평균 거리 (B, h, N)
        valid = (mass.squeeze(-1) > 1e-3).float()  # patch 쪽 합이 너무 작은 query는 평균에서 뺀다
        mad.append(((d_q * valid).sum(dim=(0, 2)) / valid.sum(dim=(0, 2)).clamp_min(1)).cpu().tolist())
        blk.msa.keep_attn = False
    return sim, mad, cls_mass


if vit_full is None:  # RUNS로 vit_100을 빼고 돌린 경우
    print("vit_100을 돌리지 않아 내부 분석을 건너뛴다")
else:
    sim, mad, cls_mass = analyze(vit_full)
    (OUT / "analysis.json").write_text(json.dumps({"pos_sim": sim.tolist(), "mean_attn_dist": mad, "cls_attn_mass": cls_mass},
                                                  indent=2), encoding="utf-8")

    g = vit_full.embed.grid
    fig, axes = plt.subplots(g, g, figsize=(6, 6))
    for r in range(g):
        for c_ in range(g):
            axes[r, c_].imshow(sim[r * g + c_].reshape(g, g), cmap="viridis", vmin=-1, vmax=1)
            axes[r, c_].axis("off")
    fig.suptitle("position embedding cosine similarity")
    fig.savefig(OUT / "pos-similarity.png", dpi=150, bbox_inches="tight")
    plt.show()

    fig, ax = plt.subplots(figsize=(6, 4))
    for l, ds_ in enumerate(mad):
        ax.scatter([l + 1] * len(ds_), ds_, color="C0")
    ax.set_xlabel("layer")
    ax.set_ylabel("mean attention distance (px)")
    fig.savefig(OUT / "attention-distance.png", dpi=150, bbox_inches="tight")
    plt.show()


# %% [markdown]
# ## 8. 결과 그래프

# %%
fig, ax = plt.subplots(figsize=(6, 4))
for model_name in ["vit", "resnet"]:
    if not all(f"{model_name}_{p}" in summary for p in (10, 30, 100)):
        continue  # SMOKE 모드 등 일부 설정만 돌린 경우
    xs = [summary[f"{model_name}_{p}"]["n_train"] for p in (10, 30, 100)]
    ys = [summary[f"{model_name}_{p}"]["final_acc_mean"] for p in (10, 30, 100)]
    ax.plot(xs, ys, marker="o", label=model_name)
ax.set_xscale("log")
ax.set_xlabel("number of training images")
ax.set_ylabel("CIFAR-10 test accuracy")
ax.legend()
fig.savefig(OUT / "data-scaling.png", dpi=150, bbox_inches="tight")
plt.show()
