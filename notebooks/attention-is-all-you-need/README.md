# Attention Is All You Need — 미니 구현과 재현 실험

Transformer(encoder-decoder)를 PyTorch로 처음부터 구현하고, Multi30k 영어→독일어 번역에서 논문 Table 3의 ablation 일부를 재현한다.

- 블로그 글: <https://Liliid.github.io/posts/attention-is-all-you-need/>
- 논문: <https://arxiv.org/abs/1706.03762>
- Colab: [attention-is-all-you-need.ipynb](https://colab.research.google.com/github/Liliid/Liliid.github.io/blob/main/notebooks/attention-is-all-you-need/attention-is-all-you-need.ipynb)

## 파일

| 파일 | 내용 |
|------|------|
| `attention-is-all-you-need.py` | 단일 소스 (percent 형식). 구현 + 저차원 예시 검증 + ablation 실험 |
| `attention-is-all-you-need.ipynb` | 위 파일에서 생성한 Colab 노트북 |
| `make_figures.py` | 블로그 글 그림 생성 (`../../assets/img/posts/attention-is-all-you-need/`) |
| `outputs/metrics.json` | 실험 결과 (설정별 BLEU, perplexity, 학습 곡선) |

## 로컬 실행

```bash
# 블로그 루트에서
uv venv --python 3.12 .venv
uv pip install --python .venv torch --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv numpy matplotlib datasets tokenizers sacrebleu

cd notebooks/attention-is-all-you-need
SMOKE=1 MPLBACKEND=Agg ../../.venv/Scripts/python attention-is-all-you-need.py   # 빠른 동작 확인
MPLBACKEND=Agg ../../.venv/Scripts/python attention-is-all-you-need.py           # 본 실험 (seed 3개)
../../.venv/Scripts/python make_figures.py
```

환경 변수: `SEEDS=0,1,2` (돌릴 seed), `OUTDIR=outputs` (결과 폴더), `DEEP_DIVE=<id>` (심화 글의 설정 묶음).

기본 실행은 논문 Table 3에 대응하는 5개 설정(base, h1, h4, h32, learned_pe)만 돌린다. 심화 글의 설정은 `DEEP_DIVE=transformer-without-scaling`(scaling 제거) 또는 `DEEP_DIVE=transformer-without-position`(위치 정보 제거)으로 돌린다. 지금 `outputs/metrics.json`에는 이 스위치를 나누기 전에 한 번에 돌린 8개 설정의 결과가 함께 들어 있다.

## 결과

Multi30k test2016 (1,000문장), greedy 디코딩, seed 3개 평균 ± 표준편차. RTX 4070 Ti SUPER에서 설정당 약 2분 10초.

| 설정 | head당 d_k | test BLEU |
|------|-----------|-----------|
| base (N=3, d_model=256, h=8, sinusoidal PE) | 32 | 34.18 ± 0.16 |
| h=1 | 256 | 34.29 ± 0.50 |
| h=4 | 64 | 35.07 ± 0.23 |
| h=32 | 8 | 32.44 ± 0.21 |
| learned PE | 32 | 32.36 ± 0.55 |
| PE 없음 | 32 | 26.62 ± 0.65 |
| √d_k scaling 없음 (h=8) | 32 | 20.67 ± 0.75 |
| √d_k scaling 없음 (h=1) | 256 | 1.43 ± 0.23 |

결과 해석은 블로그 글의 "미니 재현 실험" 섹션 참고. `outputs/metrics.json`은 seed 0과 seed 1,2를 따로 실행해 합친 것이다(seed 0 단독 결과는 `outputs/metrics_seed0.json`).
