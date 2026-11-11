"""심화: √d_k로 나누지 않으면 Transformer는 어떻게 될까?

리뷰 노트북(notebooks/attention-is-all-you-need)의 실험 결과를 재사용하고, 글에 쓸 수치를 계산해 numbers.json에 저장한다.

결과를 다시 만드는 명령 (notebooks/attention-is-all-you-need/에서): DEEP_DIVE=transformer-without-scaling OUTDIR=outputs_dd_scaling python attention-is-all-you-need.py

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
import torch

sys.stdout.reconfigure(encoding="utf-8")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import viz_helpers as vz  # noqa: E402

vz.setup_style()
SLUG = "transformer-without-scaling"
FIG = ROOT / "assets/img/posts" / SLUG
SRC = ROOT / "notebooks/attention-is-all-you-need"
numbers = {}

# 1) 무작위 q, k: d_k에 따른 내적 분산, softmax 최댓값, Jacobian 크기
rng = np.random.default_rng(0)
dks = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
rows = {}
for d in dks:
    q = rng.standard_normal((4000, d))
    k = rng.standard_normal((4000, 10, d))
    dots = np.einsum("bd,bnd->bn", q, k)
    r = {"var_dot": float(dots.var())}
    for name, sc in [("unscaled", dots), ("scaled", dots / math.sqrt(d))]:
        p = np.exp(sc - sc.max(-1, keepdims=True))
        p /= p.sum(-1, keepdims=True)
        J = p[:, :, None] * (np.eye(10)[None] - p[:, None, :])
        r[f"max_p_{name}"] = float(p.max(-1).mean())
        r[f"jac_{name}"] = float(np.linalg.norm(J.reshape(len(J), -1), axis=1).mean())
    rows[d] = r
numbers["random_qk"] = {d: rows[d] for d in [4, 32, 64, 256, 512]}

fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
axes[0].plot(dks, [rows[d]["var_dot"] for d in dks], marker="o", color=vz.color("gray"), label="측정값")
axes[0].plot(dks, dks, ls="--", color=vz.color("red"), label="$d_k$")
axes[0].set(xscale="log", yscale="log", xlabel="$d_k$", ylabel=r"Var$(q \cdot k)$", title="내적의 분산은 $d_k$에 비례")
axes[0].legend()
for i, (key, title, ylabel) in enumerate([("max_p", "softmax 최댓값 (key 10개)", "평균 max 확률"),
                                          ("jac", "softmax Jacobian 크기", r"$\Vert J \Vert_F$ 평균")]):
    axes[i + 1].plot(dks, [rows[d][f"{key}_unscaled"] for d in dks], marker="o", color=vz.color("orange"), label="나누지 않음")
    axes[i + 1].plot(dks, [rows[d][f"{key}_scaled"] for d in dks], marker="o", color=vz.color("blue"), label=r"$\sqrt{d_k}$로 나눔")
    for d in (32, 256):
        axes[i + 1].axvline(d, color="#bbb", lw=1, ls=":")
    axes[i + 1].set(xscale="log", xlabel="$d_k$", ylabel=ylabel, title=title)
    axes[i + 1].legend()
fig.tight_layout()
vz.save(fig, FIG / "scaling.png")

# 2) 실제 모델의 초기 attention logit: 리뷰 노트북의 Transformer를 초기화해서 측정
src = (SRC / "attention-is-all-you-need.py").read_text(encoding="utf-8").split("# ## 2. 저차원 예시로 검증")[0]
src = src.replace('OUT = Path(os.environ.get("OUTDIR", "outputs"))', 'OUT = Path(r"%s")' % (HERE / "_tmp"))
src = src.replace('DATA = Path("data")', 'DATA = Path(r"%s")' % (HERE / "_tmp"))
g = {"__name__": "dd"}
exec(compile(src, "transformer", "exec"), g)
Transformer, Config, replace = g["Transformer"], g["Config"], g["replace"]
torch.manual_seed(0)
init = {}
for h in (8, 1):
    c = replace(Config(), h=h, vocab_size=8000)
    model = Transformer(c, pad_id=0).eval()
    ids = torch.randint(4, 8000, (64, 15))
    with torch.no_grad():
        x = model.embed(ids)
        att = model.encoder[0].self_attn
        Q, K = att.split_heads(att.W_Q(x)), att.split_heads(att.W_K(x))
        logits = Q @ K.transpose(-2, -1)  # (B, h, n, n)
        d_k = Q.size(-1)
        res = {"d_k": d_k, "q_var": float(Q.var()), "logit_std_unscaled": float(logits.std()),
               "logit_std_scaled": float((logits / math.sqrt(d_k)).std())}
        for name, lg in [("unscaled", logits), ("scaled", logits / math.sqrt(d_k))]:
            res[f"max_p_{name}"] = float(lg.softmax(-1).max(-1).values.mean())
    init[f"h{h}"] = res
numbers["model_init"] = init
for f in (HERE / "_tmp").glob("*"):
    f.unlink()
(HERE / "_tmp").rmdir()

# 3) 학습 결과 (리뷰 노트북 metrics.json, seed 3개)
m = json.loads((SRC / "outputs/metrics.json").read_text(encoding="utf-8"))
res = m["results"]
names = ["base", "no_scale", "h1", "no_scale_h1"]
numbers["results"] = {n: {k: res[n][k] for k in ("bleu_mean", "bleu_std", "bleu", "val_ppl_mean")} for n in names}

fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.2), gridspec_kw={"width_ratios": [1, 1.25]})
xs = np.array([0, 1])
w = 0.36
for j, (nm_s, nm_u) in enumerate([("base", "no_scale"), ("h1", "no_scale_h1")]):
    for k, (nm, col, lab) in enumerate([(nm_s, "blue", r"$\sqrt{d_k}$로 나눔"), (nm_u, "orange", "나누지 않음")]):
        mu, sd = res[nm]["bleu_mean"], res[nm]["bleu_std"]
        axes[0].bar(j + (k - 0.5) * w, mu, w, yerr=sd, color=vz.color(col), capsize=3, label=lab if j == 0 else None)
        axes[0].text(j + (k - 0.5) * w, mu + sd + 0.6, f"{mu:.1f}", ha="center", fontsize=10)
axes[0].set_xticks(xs, ["h=8, $d_k$=32", "h=1, $d_k$=256"])
axes[0].set(ylabel="test BLEU", title="seed 3개 평균 ± 표준편차", ylim=(0, 46))
axes[0].legend(loc="upper right", ncol=2)
for nm, col, ls, lab in [("base", "blue", "-", "나눔, $d_k$=32"), ("no_scale", "orange", "-", "안 나눔, $d_k$=32"),
                         ("h1", "blue", ":", "나눔, $d_k$=256"), ("no_scale_h1", "orange", ":", "안 나눔, $d_k$=256")]:
    hst = m["history"][nm]
    steps = hst[0]["step"]
    vl = np.mean([h["val_loss"] for h in hst], axis=0)
    axes[1].plot(steps, vl, color=vz.color(col), ls=ls, lw=2, label=lab)
axes[1].set(xlabel="step", ylabel="validation loss (seed 평균)", title="학습 곡선", ylim=(1.8, 6.5))
axes[1].legend()
fig.tight_layout()
vz.save(fig, FIG / "results.png")

(HERE / "numbers.json").write_text(json.dumps(numbers, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps(numbers, indent=1, ensure_ascii=False))
