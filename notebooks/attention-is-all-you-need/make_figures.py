"""블로그 글용 그림을 만든다. outputs/의 toy.json, metrics.json, cross_attention.json을 읽는다.

    python make_figures.py
"""
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import viz_helpers as vz  # noqa: E402

vz.setup_style()
FIG = ROOT / "assets/img/posts/attention-is-all-you-need"
OUT = HERE / "outputs"
toy = json.loads((OUT / "toy.json").read_text(encoding="utf-8"))
saved = []


def save(fig, name):
    saved.append(vz.save(fig, FIG / name))


# 1) scaled dot-product attention의 shape 흐름
fig, ax = vz.new_canvas(13, 7.5)
y1, y0 = 6.4, 0
q = vz.draw_tensor(ax, (3, 2), origin=(0, y1), color="blue", labels=("$n=3$", "$d_k=2$"), name="$Q$", grid=True)
k = vz.draw_tensor(ax, (2, 3), origin=(3.6, y1 + 0.5), color="orange", labels=("$d_k=2$", "$n=3$"), name=r"$K^\top$", grid=True)
vz.label(ax, ((q["right"][0] + k["left"][0]) / 2, y1 + 1.5), "×", size=22)
s = vz.draw_tensor(ax, (3, 3), origin=(10, y1), color="green", labels=("$n$", "$n$"), name=r"$QK^\top / \sqrt{d_k}$", grid=True)
vz.arrow(ax, k["right"], s["left"], text="내적")
a = vz.draw_tensor(ax, (3, 3), origin=(0, y0), color="green", labels=("$n$", "$n$"), name="$A$ (행마다 합 1)", grid=True)
vz.arrow(ax, (s["bottom"][0], s["bottom"][1] - 0.9), (a["right"][0] + 0.3, a["top"][1] + 1.2), rad=-0.12)
vz.label(ax, (4.6, y1 - 1.45), "softmax (행 방향)", size=11)
v = vz.draw_tensor(ax, (3, 2), origin=(5, y0), color="purple", labels=("$n=3$", "$d_v=2$"), name="$V$", grid=True)
vz.label(ax, ((a["right"][0] + v["left"][0]) / 2, y0 + 1.5), "×", size=22)
o = vz.draw_tensor(ax, (3, 2), origin=(10.5, y0), color="teal", labels=("$n=3$", "$d_v=2$"), name="출력", grid=True)
vz.arrow(ax, v["right"], o["left"], text="가중합")
vz.finalize(ax)
save(fig, "attention-shape.png")

# 2) 저차원 예시: score → attention → 출력
att = toy["attention"]
fig, ax = vz.new_canvas(13, 4.2)
tl = ["t1", "t2", "t3"]
g1 = vz.draw_grid(ax, np.array(att["scaled"]), origin=(0, 0), name=r"$QK^\top / \sqrt{2}$", fmt="{:.2f}",
                  row_labels=tl, col_labels=tl)
g2 = vz.draw_grid(ax, np.array(att["attn"]), origin=(5.5, 0), name="attention", fmt="{:.2f}", cmap="Oranges",
                  row_labels=tl, col_labels=tl, highlight=[(2, 2)])
g3 = vz.draw_grid(ax, np.array(att["out"]), origin=(11, 0.5) if False else (11, 0), name=r"출력 $= AV$", fmt="{:.2f}",
                  cmap="Purples", row_labels=tl, col_labels=["d1", "d2"])
vz.arrow(ax, g1["right"], (g2["left"][0] - 0.7, g2["left"][1]), text="softmax")
vz.arrow(ax, g2["right"], (g3["left"][0] - 0.7, g3["left"][1]), text="× V")
vz.finalize(ax)
save(fig, "attention-toy.png")

# 3) sqrt(d_k) 포화 그래프는 심화 글(notebooks/deep-dives/transformer-without-scaling)로 옮겼다

# 4) multi-head: split heads → head별 attention → concat
fig, ax = vz.new_canvas(14, 4.2)
x = vz.draw_tensor(ax, (2, 4, 8), origin=(0, 0), color="blue", labels=("B", "n", "$d_{model}$"), name="$XW^Q$: (B, n, $d_{model}$)", grid=True)
h = vz.draw_tensor(ax, (4, 4, 2), origin=(14.5, 0), color="orange", labels=("h", "n", "$d_k$"), name="(B, h, n, $d_k$): 배치마다", grid=True)
vz.arrow(ax, (x["right"][0], 2), (h["left"][0], 2), text="view + transpose")
vz.label(ax, (15.8, -1.6), r"$d_k = d_{model}/h$, head마다 따로 attention", size=11)
vz.finalize(ax)
save(fig, "multihead-split.png")

# 5) positional encoding 행렬과 위치 사이 유사도
import torch  # noqa: E402

def sinusoidal_pe(max_len, d_model):
    pos = torch.arange(max_len).unsqueeze(1)
    i2 = torch.arange(0, d_model, 2)
    freq = torch.exp(-math.log(10000.0) * i2 / d_model)
    pe = torch.zeros(max_len, d_model)
    pe[:, 0::2] = torch.sin(pos * freq)
    pe[:, 1::2] = torch.cos(pos * freq)
    return pe.numpy()

pe = sinusoidal_pe(64, 128)
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw={"width_ratios": [1.6, 1]})
im = axes[0].imshow(pe, cmap="RdBu_r", aspect="auto", vmin=-1, vmax=1)
axes[0].set(xlabel="차원 i (왼쪽일수록 높은 주파수)", ylabel="위치 pos", title=r"$PE$ ($d_{model}=128$)")
axes[0].grid(False)
fig.colorbar(im, ax=axes[0], fraction=0.03)
sim = pe @ pe.T
im = axes[1].imshow(sim, cmap="viridis")
axes[1].set(xlabel="위치", ylabel="위치", title=r"$PE_{pos} \cdot PE_{pos'}$")
axes[1].grid(False)
fig.colorbar(im, ax=axes[1], fraction=0.045)
fig.tight_layout()
save(fig, "positional-encoding.png")

# 6) causal mask
cz = toy["causal"]
mask = np.triu(np.ones((4, 4)), 1).astype(bool)
masked_scores = np.where(mask, np.nan, np.array(cz["scores"]))
fig, ax = vz.new_canvas(13, 4.2)
tl4 = ["y1", "y2", "y3", "y4"]
m1 = vz.draw_grid(ax, np.array(cz["attn_nomask"]), origin=(0, 0), name="mask 없음", fmt="{:.2f}", cmap="Oranges",
                  row_labels=tl4, col_labels=tl4)
m2 = vz.draw_grid(ax, masked_scores, origin=(6.5, 0), name=r"score ($-\infty$ = 빈칸)", fmt="{:.1f}", colored=False,
                  row_labels=tl4, col_labels=tl4)
m3 = vz.draw_grid(ax, np.array(cz["attn_mask"]), origin=(13, 0), name="causal mask 후 attention", fmt="{:.2f}",
                  cmap="Oranges", row_labels=tl4, col_labels=tl4)
vz.arrow(ax, m2["right"], (m3["left"][0] - 0.7, m3["left"][1]), text="softmax")
vz.finalize(ax)
save(fig, "causal-mask.png")

# 7) 결과 그래프 (실험이 끝났을 때만)
mpath = OUT / "metrics.json"
if mpath.exists():
    m = json.loads(mpath.read_text(encoding="utf-8"))
    res = m["results"]
    order = [n for n in ["base", "h1", "h4", "h32", "learned_pe"] if n in res]  # 논문 Table 3에 대응하는 설정만 (나머지는 심화 글)
    labels = {"base": "base\n(h=8)", "h1": "h=1", "h4": "h=4", "h32": "h=32", "learned_pe": "learned\nPE",
              "no_pe": "PE 없음", "no_scale": "scaling 없음\n(h=8)", "no_scale_h1": "scaling 없음\n(h=1)"}
    colors = {"base": "blue", "h1": "teal", "h4": "green", "h32": "gray", "learned_pe": "purple", "no_pe": "red",
              "no_scale": "orange", "no_scale_h1": "yellow"}
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), gridspec_kw={"width_ratios": [1.3, 1]})
    xs = np.arange(len(order))
    means = [res[n]["bleu_mean"] for n in order]
    errs = [res[n]["bleu_std"] for n in order]
    axes[0].bar(xs, means, yerr=errs, color=[vz.color(colors[n]) for n in order], capsize=3)
    for x_, mu, e in zip(xs, means, errs):
        axes[0].text(x_, mu + e + 0.5, f"{mu:.1f}", ha="center", fontsize=9)
    axes[0].set_xticks(xs, [labels[n] for n in order], fontsize=9)
    axes[0].set(ylabel="test BLEU (greedy)", title=f"Multi30k En→De, seed {len(m['seeds'])}개 평균 ± 표준편차",
                ylim=(0, max(means) + 4))
    for n in order:
        hist = m["history"][n][0]
        axes[1].plot(hist["step"], hist["val_loss"], label=labels[n].replace("\n", " "), color=vz.color(colors[n]),
                     ls="-" if n in ("base", "no_pe", "no_scale_h1", "learned_pe") else ":")
    axes[1].set(xlabel="step", ylabel="validation loss", title="검증 loss (seed 0)", ylim=(1.8, 2.8))
    axes[1].legend(fontsize=8, ncol=2)
    fig.tight_layout()
    save(fig, "ablation.png")

cpath = OUT / "cross_attention.json"
if cpath.exists():
    c = json.loads(cpath.read_text(encoding="utf-8"))
    A = np.array(c["attn"])
    fig, ax = plt.subplots(figsize=(7.5, 6))
    ax.imshow(A, cmap="Blues")
    clean = lambda ts: [t.replace("▁", "") for t in ts]  # BPE 단어 시작 기호(▁)는 폰트에 없어서 지운다
    ax.set_xticks(range(len(c["src_tokens"])), clean(c["src_tokens"]), rotation=60, ha="right", fontsize=10)
    ax.set_yticks(range(len(c["tgt_tokens"])), clean(c["tgt_tokens"]), fontsize=10)
    ax.set(xlabel="영어 입력 (BPE)", ylabel="독일어 출력 (BPE)", title="encoder-decoder attention (마지막 layer, head 평균)")
    ax.grid(False)
    fig.tight_layout()
    save(fig, "cross-attention.png")

for p in saved:
    print(p)
