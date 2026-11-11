"""심화: 위치 정보가 없어도 Transformer는 번역할 수 있을까?

리뷰 노트북(notebooks/attention-is-all-you-need)의 실험 결과를 재사용하고, 글에 쓸 작은 예시를 계산해 numbers.json에 저장한다.

결과를 다시 만드는 명령 (notebooks/attention-is-all-you-need/에서): DEEP_DIVE=transformer-without-position OUTDIR=outputs_dd_position python attention-is-all-you-need.py

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
SLUG = "transformer-without-position"
FIG = ROOT / "assets/img/posts" / SLUG
SRC = ROOT / "notebooks/attention-is-all-you-need"
numbers = {}


def softmax(x):
    e = np.exp(x - x.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def attention(Q, K, V, mask=None):
    s = Q @ K.T / np.sqrt(Q.shape[1])
    if mask is not None:
        s = np.where(mask, -np.inf, s)
    a = softmax(s)
    return a @ V, a


# 1) encoder self-attention은 순서를 섞으면 출력도 똑같이 섞일 뿐이다 (permutation equivariance)
X = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])  # 토큰 3개 (리뷰 글의 예시와 같은 값)
out, _ = attention(X, X, X)
perm = [2, 1, 0]  # "개가 사람을 문다" → "문다 사람을 개가"처럼 순서만 뒤집기
out_p, _ = attention(X[perm], X[perm], X[perm])
assert np.allclose(out_p, out[perm])
numbers["perm"] = {"X": X.tolist(), "out": out.round(3).tolist(), "perm": perm, "out_perm": out_p.round(3).tolist()}

# 2) causal mask가 있으면 위치를 알아낼 수 있다: 모든 query가 고르게 attention하면 BOS의 기여가 1/i
n = 5
V = np.zeros((n, 2))
V[0] = [1.0, 0.0]  # BOS 토큰의 value
V[1:] = [0.0, 1.0]  # 나머지 토큰의 value (내용과 무관하게 같다고 가정)
Q = np.zeros((n, 2))
K = np.zeros((n, 2))  # 모든 score가 0 → 볼 수 있는 토큰에 고르게
causal = np.triu(np.ones((n, n), dtype=bool), 1)
out_c, a_c = attention(Q, K, V, causal)
out_nc, a_nc = attention(Q, K, V, None)
numbers["causal_uniform"] = {"attn": a_c.round(3).tolist(), "out": out_c.round(3).tolist(),
                             "out_no_mask": out_nc.round(3).tolist()}

fig, ax = vz.new_canvas(12.5, 4.6)
labs = ["BOS", "y1", "y2", "y3", "y4"]
g1 = vz.draw_grid(ax, np.where(causal, np.nan, a_c), origin=(0, 0), fmt="{:.2f}", cmap="Oranges", vmin=0, vmax=1,
                  name="causal mask + 고른 attention", row_labels=labs, col_labels=labs)
g2 = vz.draw_grid(ax, V, origin=(7.2, 0), fmt="{:.0f}", colored=False, name="value", row_labels=labs, col_labels=["v1", "v2"])
g3 = vz.draw_grid(ax, out_c, origin=(11.6, 0), fmt="{:.2f}", cmap="Blues", vmin=0, vmax=1, name="출력 (v1 = 1/i)",
                  row_labels=labs, col_labels=["v1", "v2"])
vz.label(ax, ((g1["right"][0] + g2["left"][0]) / 2 - 0.2, 2.5), "×", size=22)
vz.arrow(ax, (g2["right"][0] + 0.1, 2.5), (g3["left"][0] - 0.6, 2.5))
vz.finalize(ax)
vz.save(fig, FIG / "causal-position.png")

# 3) 학습 결과 (리뷰 노트북 metrics.json, seed 3개)
m = json.loads((SRC / "outputs/metrics.json").read_text(encoding="utf-8"))
res = m["results"]
names = ["base", "learned_pe", "no_pe"]
numbers["results"] = {nm: {k: res[nm][k] for k in ("bleu_mean", "bleu_std", "bleu", "val_ppl_mean")} for nm in names}
numbers["final_train_loss"] = {nm: [round(h["train_loss"][-1], 3) for h in m["history"][nm]] for nm in names}
numbers["final_val_loss"] = {nm: [round(h["val_loss"][-1], 3) for h in m["history"][nm]] for nm in names}

fig, axes = plt.subplots(1, 2, figsize=(12, 4.0), gridspec_kw={"width_ratios": [1, 1.3]})
labels = {"base": "sinusoidal PE", "learned_pe": "learned PE", "no_pe": "PE 없음"}
cols = {"base": "blue", "learned_pe": "purple", "no_pe": "red"}
for i, nm in enumerate(names):
    mu, sd = res[nm]["bleu_mean"], res[nm]["bleu_std"]
    axes[0].bar(i, mu, yerr=sd, color=vz.color(cols[nm]), capsize=3)
    axes[0].text(i, mu + sd + 0.6, f"{mu:.1f}", ha="center")
axes[0].set_xticks(range(3), [labels[n] for n in names])
axes[0].set(ylabel="test BLEU", title="seed 3개 평균 ± 표준편차", ylim=(0, 40))
for nm in ["base", "no_pe"]:
    for j, h in enumerate(m["history"][nm]):
        axes[1].plot(h["step"], h["val_loss"], color=vz.color(cols[nm]), lw=1.5, alpha=0.8,
                     label=labels[nm] if j == 0 else None)
        axes[1].plot(h["step"], h["train_loss"], color=vz.color(cols[nm]), lw=1, ls=":", alpha=0.6)
axes[1].set(xlabel="step", ylabel="loss", title="실선: validation, 점선: train (label smoothing 포함)", ylim=(1.8, 4.5))
axes[1].legend()
fig.tight_layout()
vz.save(fig, FIG / "results.png")

(HERE / "numbers.json").write_text(json.dumps(numbers, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps(numbers, indent=1, ensure_ascii=False))
