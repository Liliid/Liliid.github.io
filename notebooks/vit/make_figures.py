"""블로그 글용 그림을 만든다. outputs/의 toy.json, metrics.json, analysis.json을 읽는다.

    python make_figures.py
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import viz_helpers as vz  # noqa: E402

vz.setup_style()
FIG = ROOT / "assets/img/posts/vit"
OUT = HERE / "outputs"
toy = json.loads((OUT / "toy.json").read_text(encoding="utf-8"))
saved = []
PATCH_COLORS = ["blue", "orange", "green", "purple"]


def save(fig, name):
    saved.append(vz.save(fig, FIG / name))


# 1) voxel: 이미지 (C, H, W)에서 patch 하나 = C×P×P 블록
img = np.ones((3, 8, 8), dtype=bool)
patch = np.zeros_like(img)
patch[:, 0:4, 4:8] = True
fig, ax = vz.draw_voxels(img, dim_labels=("C", "H", "W"), color="blue", highlight=patch, alpha=0.1,
                         title="patch 하나 = C×P×P 픽셀 (P=4: 3×4×4=48개 값)", figsize=(5.5, 5))
save(fig, "patch-voxel.png")

# 2) 저차원 예시: 4×4 이미지를 P=2로 patchify → (N=4, P²C=4) → E → (4, D=2)
pt = toy["patchify"]
image = np.array(pt["image"])
patches = np.array(pt["patches"])
xpE = np.array(toy["patch_embed"]["xpE"])
fig, ax = vz.new_canvas(14, 4.6)
g_img = vz.draw_grid(ax, image, origin=(0, 0), fmt="{:.0f}", colored=False, name="이미지 (4×4, C=1)")
for k, (r0, c0) in enumerate([(0, 0), (0, 2), (2, 0), (2, 2)]):
    ax.add_patch(Rectangle((c0, 4 - r0 - 2), 2, 2, facecolor=vz.color(PATCH_COLORS[k]), alpha=0.25,
                           edgecolor=vz.color(PATCH_COLORS[k]), lw=3, zorder=3))
g_p = vz.draw_grid(ax, patches, origin=(6.5, 0), fmt="{:.0f}", colored=False, name=r"$x_p$ (N=4, $P^2C$=4)",
                   row_labels=["p1", "p2", "p3", "p4"])
for k in range(4):
    ax.add_patch(Rectangle((6.5, 3 - k), 4, 1, facecolor=vz.color(PATCH_COLORS[k]), alpha=0.25, edgecolor="none", zorder=3))
g_z = vz.draw_grid(ax, xpE, origin=(13.5, 0), fmt="{:.1f}", cmap="Greens", name=r"$x_p E$ (N=4, D=2)",
                   row_labels=["p1", "p2", "p3", "p4"])
vz.arrow(ax, g_img["right"], (g_p["left"][0] - 0.8, 2), text="patchify")
vz.arrow(ax, g_p["right"], (g_z["left"][0] - 0.8, 2), text=r"$\times E$ (4×2)")
vz.finalize(ax)
save(fig, "patchify-toy.png")

# 3) 전체 shape 흐름: (3, 32, 32) → (64, 48) → (64, 192) → [class] 추가 (65, 192)
fig, ax = vz.new_canvas(15, 4.4)
a = vz.draw_tensor(ax, (1.5, 4, 4), origin=(0, 0), color="gray", labels=("C=3", "H=32", "W=32"), name="이미지", grid=True)
b = vz.draw_tensor(ax, (4, 2.4), origin=(9, 0), color="blue", labels=("N=64", "$P^2C$=48"), name="patch 시퀀스")
c = vz.draw_tensor(ax, (4, 3.4), origin=(15, 0), color="green", labels=("N=64", "D=192"), name=r"$x_p E$")
d = vz.draw_tensor(ax, (4.5, 3.4), origin=(22, -0.5), color="teal", labels=("N+1=65", "D=192"), name=r"$z_0 = [x_{class}; x_pE] + E_{pos}$")
ax.add_patch(Rectangle((22, 3.5), 3.4, 0.5, facecolor=vz.color("red"), alpha=0.45, edgecolor="#333", lw=1.3, zorder=4))
vz.arrow(ax, (a["right"][0], 2), (b["left"][0], 2), text="patchify")
vz.arrow(ax, (b["right"][0], 2), (c["left"][0], 2), text="E")
vz.arrow(ax, (c["right"][0], 2), (d["left"][0], 2), text="+class, +pos")
vz.label(ax, (25.6, 3.75), "[class]", size=10, color_=vz.color("red"), ha="left")
vz.finalize(ax)
save(fig, "vit-shape.png")

# 4) CNN 3×3 conv vs ViT self-attention: 첫 layer에서 한 위치가 보는 범위 (8×8 patch/픽셀 grid)
cnn = np.zeros((8, 8))
cnn[2:5, 2:5] = 1
vit = np.ones((8, 8))
fig, ax = vz.new_canvas(12, 5)
q = (3, 3)
vz.draw_grid(ax, cnn, origin=(0, 0), fmt="", cmap="Blues", vmin=0, vmax=1.2, name="CNN 3×3 conv: 주변 9칸만", highlight=[q])
vz.draw_grid(ax, vit, origin=(10, 0), fmt="", cmap="Oranges", vmin=0, vmax=1.6, name="ViT self-attention: 첫 layer부터 64칸 전부", highlight=[q])
vz.finalize(ax)
save(fig, "receptive-field.png")

# 5) 위치 임베딩 2D 보간 (fine-tuning 해상도 변경)
pi = toy["pos_interp"]
fig, ax = vz.new_canvas(8, 3.6)
p1 = vz.draw_grid(ax, np.array(pi["before"]), origin=(0, 0.5), fmt="{:.1f}", cmap="Purples", vmin=0, vmax=3.5, name="2×2 grid")
p2 = vz.draw_grid(ax, np.array(pi["after"]), origin=(5, 0), fmt="{:.1f}", cmap="Purples", vmin=0, vmax=3.5, name="3×3로 보간")
vz.arrow(ax, p1["right"], (p2["left"][0] - 0.2, 1.5), text="bilinear")
vz.finalize(ax)
save(fig, "pos-interp.png")

# 6) mean attention distance 예시: 3×3 grid, 가운데 query
dl, dg = toy["attn_dist_local"], toy["attn_dist_global"]
fig, ax = vz.new_canvas(13, 4.2)
g1 = vz.draw_grid(ax, np.array(dl["dist"]), origin=(0, 0), fmt="{:.1f}", colored=False, name="query와의 거리 (px)", highlight=[(1, 1)])
g2 = vz.draw_grid(ax, np.array(dl["attn"]), origin=(5, 0), fmt="{:.2f}", cmap="Oranges", vmin=0, vmax=0.45,
                  name=f"local attention → {dl['mean']:.2f}px")
g3 = vz.draw_grid(ax, np.array(dg["attn"]), origin=(10, 0), fmt="{:.2f}", cmap="Oranges", vmin=0, vmax=0.45,
                  name=f"고른 attention → {dg['mean']:.2f}px")
vz.finalize(ax)
save(fig, "attention-distance-toy.png")

# 7) 결과 (실험이 끝났을 때만)
mpath = OUT / "metrics.json"
if mpath.exists():
    m = json.loads(mpath.read_text(encoding="utf-8"))
    res = m["results"]
    if all(f"{mn}_{p}" in res for mn in ("vit", "resnet") for p in (10, 30, 100)):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
        for mn, col, base in [("vit", "orange", "ViT"), ("resnet", "blue", "ResNet-18")]:
            lab = f"{base} ({res[f'{mn}_100']['params'] / 1e6:.1f}M)"
            xs = [res[f"{mn}_{p}"]["n_train"] for p in (10, 30, 100)]
            ys = [res[f"{mn}_{p}"]["final_acc_mean"] * 100 for p in (10, 30, 100)]
            es = [res[f"{mn}_{p}"]["final_acc_std"] * 100 for p in (10, 30, 100)]
            axes[0].errorbar(xs, ys, yerr=es, marker="o", color=vz.color(col), label=lab, capsize=3)
            for x_, y_ in zip(xs, ys):
                axes[0].annotate(f"{y_:.1f}", (x_, y_), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9)
        axes[0].set(xscale="log", xlabel="학습 이미지 수", ylabel="CIFAR-10 test 정확도 (%)", title="데이터 양에 따른 정확도")
        axes[0].legend()
        gaps = [(res[f"resnet_{p}"]["final_acc_mean"] - res[f"vit_{p}"]["final_acc_mean"]) * 100 for p in (10, 30, 100)]
        axes[1].bar(["10%\n(5k)", "30%\n(15k)", "100%\n(50k)"], gaps, color=vz.color("gray"))
        for i, gp in enumerate(gaps):
            axes[1].text(i, gp + 0.2, f"{gp:.1f}%p", ha="center")
        axes[1].set(ylabel="ResNet - ViT (%p)", title="두 모델의 정확도 차이")
        fig.tight_layout()
        save(fig, "data-scaling.png")

        fig, ax = plt.subplots(figsize=(7.5, 4))
        for name, col, ls in [("vit_10", "orange", ":"), ("vit_100", "orange", "-"), ("resnet_10", "blue", ":"), ("resnet_100", "blue", "-")]:
            h = m["history"][name][0]
            ax.plot(h["step"], np.array(h["test_acc"]) * 100, color=vz.color(col), ls=ls, label=f"{name} test")
            ax.plot(h["step"], np.array(h["train_acc"]) * 100, color=vz.color(col), ls=ls, alpha=0.35)
        ax.set(xlabel="step", ylabel="정확도 (%)", title="학습 곡선 (진한 선: test, 연한 선: train)")
        ax.legend(fontsize=9)
        fig.tight_layout()
        save(fig, "learning-curves.png")

    pos_names = [n for n in ["vit_nopos_100", "vit_100", "vit_pos2d_100"] if n in res]
    if len(pos_names) == 3:
        fig, ax = plt.subplots(figsize=(6, 3.8))
        labs = ["위치 임베딩 없음", "1D 학습 (기본)", "2D 학습 (행+열)"]
        ys = [res[n]["final_acc_mean"] * 100 for n in pos_names]
        es = [res[n]["final_acc_std"] * 100 for n in pos_names]
        ax.bar(labs, ys, yerr=es, color=[vz.color(c) for c in ("red", "orange", "purple")], capsize=3)
        for i, y_ in enumerate(ys):
            ax.text(i, y_ + 0.3, f"{y_:.1f}", ha="center")
        ax.set(ylabel="CIFAR-10 test 정확도 (%)", ylim=(min(ys) - 5, max(ys) + 2), title="위치 임베딩 ablation (100% 데이터)")
        fig.tight_layout()
        save(fig, "pos-ablation.png")

apath = OUT / "analysis.json"
if apath.exists():
    an = json.loads(apath.read_text(encoding="utf-8"))
    sim = np.array(an["pos_sim"])
    g = int(round(np.sqrt(len(sim))))
    fig, axes = plt.subplots(g, g, figsize=(6.2, 6.2))
    for r in range(g):
        for c_ in range(g):
            axes[r, c_].imshow(sim[r * g + c_].reshape(g, g), cmap="viridis", vmin=-1, vmax=1)
            axes[r, c_].set_xticks([])
            axes[r, c_].set_yticks([])
            axes[r, c_].grid(False)
            for s in axes[r, c_].spines.values():
                s.set_visible(False)
    fig.suptitle("위치 임베딩 cosine similarity\n(칸 (i, j) = patch (i, j)와 나머지 patch의 유사도)", fontsize=12)
    fig.subplots_adjust(wspace=0.08, hspace=0.08)
    save(fig, "pos-similarity.png")

    mad = np.array(an["mean_attn_dist"])  # (L, heads)
    # 기준선: 64개 patch에 고르게 attention할 때와 3x3 이웃에만 고르게 attention할 때의 평균 거리
    co = np.array([(i // g, i % g) for i in range(g * g)]) * 4.0
    dmat = np.sqrt(((co[:, None] - co[None]) ** 2).sum(-1))
    uniform = dmat.mean()
    local = np.mean([dmat[i, np.max(np.abs(co - co[i]), axis=1) <= 4].mean() for i in range(g * g)])
    fig, ax = plt.subplots(figsize=(7, 4.2))
    cols = list(vz.PALETTE.values())
    for h in range(mad.shape[1]):
        ax.scatter(np.arange(1, mad.shape[0] + 1) + (h - mad.shape[1] / 2) * 0.06, mad[:, h], color=cols[h % len(cols)],
                   label=f"head {h + 1}", s=36)
    for y_, txt in [(uniform, f"전체에 고르게 ({uniform:.1f}px)"), (local, f"3×3 이웃에만 ({local:.1f}px)")]:
        ax.axhline(y_, color=vz.color("gray"), ls="--", lw=1)
        ax.text(0.6, y_ + 0.3, txt, fontsize=9, ha="left", color="#555")
    ax.set(xlabel="layer", ylabel="mean attention distance (px)", title="layer별 attention이 보는 거리 (32×32 이미지)",
           xticks=range(1, mad.shape[0] + 1), ylim=(0, 20))
    ax.legend(fontsize=8, ncol=3, loc="lower right")
    fig.tight_layout()
    save(fig, "attention-distance.png")

for p in saved:
    print(p)
