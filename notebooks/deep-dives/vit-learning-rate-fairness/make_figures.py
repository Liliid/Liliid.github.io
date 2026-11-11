"""심화: ViT와 ResNet을 같은 learning rate로 비교해도 공정할까?

리뷰 노트북(notebooks/vit)의 두 실행 결과를 비교한다.
- outputs_lr1e-3/: 두 모델 모두 lr 1e-3 (첫 실행)
- outputs/: ViT 2.5e-4, ResNet 1e-3 (본 실험)
글에 쓸 작은 문제(2차 함수 경사하강법) 예시도 계산해 numbers.json에 저장한다.

결과를 다시 만드는 명령 (notebooks/vit/에서): VIT_LR=1e-3 OUTDIR=outputs_lr1e-3 python vit.py, VIT_LR=5e-4 RUNS=vit_100 OUTDIR=outputs_lr5e-4 python vit.py

    python make_figures.py
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import viz_helpers as vz  # noqa: E402

vz.setup_style()
SLUG = "vit-learning-rate-fairness"
FIG = ROOT / "assets/img/posts" / SLUG
SRC = ROOT / "notebooks/vit"
numbers = {}

# 1) 작은 문제: 곡률이 다른 두 2차 함수 f(x) = a x^2 / 2 에 같은 learning rate로 경사하강법
#    x <- x - lr * a * x = (1 - lr * a) x.  |1 - lr*a| < 1 이어야 수렴한다 (lr < 2/a).
curv = {"완만한 문제 (a=1)": 1.0, "가파른 문제 (a=10)": 10.0}
lrs = [0.05, 0.15, 0.25]
traj = {}
for name, a in curv.items():
    for lr in lrs:
        x, xs = 1.0, [1.0]
        for _ in range(5):
            x = x - lr * a * x
            xs.append(round(x, 4))
        traj[f"{name}, lr={lr}"] = xs
numbers["toy_gd"] = traj

fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
cols = {0.05: "teal", 0.15: "blue", 0.25: "red"}
for ax, (name, a) in zip(axes, curv.items()):
    for lr in lrs:
        ys = traj[f"{name}, lr={lr}"]
        ax.plot(range(6), ys, marker="o", color=vz.color(cols[lr]), label=f"lr={lr}")
    ax.axhline(0, color="#999", lw=1)
    ax.set(title=name, xlabel="step", ylim=(-2.2, 2.2))
axes[0].set_ylabel("x (최솟값은 0)")
axes[1].legend()
fig.tight_layout()
vz.save(fig, FIG / "toy-gd.png")

# 2) lr 진단: ViT 100%
diag = json.loads((HERE / "lr_diag.json").read_text(encoding="utf-8"))["runs"]
numbers["lr_diag"] = diag
fig, ax = plt.subplots(figsize=(6.2, 3.8))
xs = np.arange(len(diag))
ax.bar(xs - 0.18, [r["train_acc"] * 100 for r in diag], 0.36, color=vz.color("gray"), label="train")
ax.bar(xs + 0.18, [r["test_acc"] * 100 for r in diag], 0.36, color=vz.color("orange"), label="test")
for i, r in enumerate(diag):
    ax.text(i + 0.18, r["test_acc"] * 100 + 1, f"{r['test_acc'] * 100:.1f}", ha="center", fontsize=9)
ax.set_xticks(xs, [f"lr={r['lr']:g}" for r in diag])
ax.set(ylabel="정확도 (%)", ylim=(0, 105), title="ViT, CIFAR-10 100%, 7,800 step")
ax.legend(loc="upper left")
fig.tight_layout()
vz.save(fig, FIG / "vit-lr.png")

# 3) 같은 lr(1e-3) vs 모델별 lr: 데이터 규모 실험과 위치 임베딩 ablation의 결론이 어떻게 달라지나
same = json.loads((SRC / "outputs_lr1e-3/metrics.json").read_text(encoding="utf-8"))["results"]
tuned = json.loads((SRC / "outputs/metrics.json").read_text(encoding="utf-8"))["results"]
fr = (10, 30, 100)
summary = {}
for tag, res in [("same_lr", same), ("tuned_lr", tuned)]:
    summary[tag] = {
        "vit": [round(res[f"vit_{p}"]["final_acc_mean"] * 100, 2) for p in fr],
        "resnet": [round(res[f"resnet_{p}"]["final_acc_mean"] * 100, 2) for p in fr],
        "gap": [round((res[f"resnet_{p}"]["final_acc_mean"] - res[f"vit_{p}"]["final_acc_mean"]) * 100, 2) for p in fr],
        "pos": {k: round(res[k]["final_acc_mean"] * 100, 2) for k in ("vit_nopos_100", "vit_100", "vit_pos2d_100")},
    }
numbers["comparison"] = summary

fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
n_train = [5000, 15000, 50000]
for tag, ls, lab in [("same_lr", ":", "같은 lr (둘 다 1e-3)"), ("tuned_lr", "-", "모델별 lr (ViT 2.5e-4)")]:
    s = summary[tag]
    axes[0].plot(n_train, s["vit"], marker="o", ls=ls, color=vz.color("orange"), label=f"ViT, {lab}")
    axes[1].plot(n_train, s["gap"], marker="o", ls=ls, color=vz.color("gray" if tag == "same_lr" else "blue"), label=lab)
    for x_, y_ in zip(n_train, s["gap"]):
        axes[1].annotate(f"{y_:.1f}", (x_, y_), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=9)
axes[0].plot(n_train, summary["tuned_lr"]["resnet"], marker="s", color=vz.color("blue"), label="ResNet-18 (두 경우 같음, lr 1e-3)")
axes[0].set(xscale="log", xlabel="학습 이미지 수", ylabel="test 정확도 (%)", title="정확도")
axes[0].legend(fontsize=8)
axes[1].set(xscale="log", xlabel="학습 이미지 수", ylabel="ResNet - ViT (%p)", title="두 모델의 차이", ylim=(0, 40))
axes[1].legend(fontsize=9)
fig.tight_layout()
vz.save(fig, FIG / "conclusion-shift.png")

(HERE / "numbers.json").write_text(json.dumps(numbers, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps(numbers, indent=1, ensure_ascii=False))
