# ViT (An Image is Worth 16x16 Words) — 미니 구현과 재현 실험

Vision Transformer를 PyTorch로 처음부터 구현하고, CIFAR-10에서 학습 데이터 양(10%, 30%, 100%)에 따른 ViT와 ResNet-18의 차이, 위치 임베딩 ablation, 학습된 ViT의 위치 임베딩 구조와 attention 거리를 확인한다.

- 블로그 글: <https://Liliid.github.io/posts/vit/>
- 논문: <https://arxiv.org/abs/2010.11929>
- Colab: [vit.ipynb](https://colab.research.google.com/github/Liliid/Liliid.github.io/blob/main/notebooks/vit/vit.ipynb)

## 파일

| 파일 | 내용 |
|------|------|
| `vit.py` | 단일 소스 (percent 형식). 구현 + 저차원 예시 검증 + 실험 + 분석 |
| `vit.ipynb` | 위 파일에서 생성한 Colab 노트북 |
| `make_figures.py` | 블로그 글 그림 생성 (`../../assets/img/posts/vit/`) |
| `outputs/metrics.json` | 실험 결과 (설정별 정확도, 학습 곡선) |
| `outputs/analysis.json` | 위치 임베딩 유사도, layer별 mean attention distance |

## 로컬 실행

```bash
# 블로그 루트에서
uv venv --python 3.12 .venv
uv pip install --python .venv torch torchvision --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv numpy matplotlib datasets

cd notebooks/vit
SMOKE=1 MPLBACKEND=Agg ../../.venv/Scripts/python vit.py   # 빠른 동작 확인
MPLBACKEND=Agg ../../.venv/Scripts/python vit.py           # 본 실험
../../.venv/Scripts/python make_figures.py
```

CIFAR-10 전체를 GPU 메모리에 올리고 증강도 GPU에서 하므로 DataLoader 설정이 필요 없다.

## 결과

CIFAR-10 test 정확도, seed 1개, 모든 설정 7,800 step (batch 256). RTX 4070 Ti SUPER에서 전체 약 47분.

| 설정 | 학습 이미지 | ViT (14.2M) | ResNet-18 (11.2M) |
|------|-----------|-------------|-------------------|
| 10% | 5,000 | 53.7% | 82.9% |
| 30% | 15,000 | 67.9% | 89.7% |
| 100% | 50,000 | 81.5% | 93.8% |

| 위치 임베딩 (ViT, 100%) | 정확도 |
|------------------------|--------|
| 없음 | 72.4% |
| 1D 학습 (기본) | 81.5% |
| 2D 학습 (행+열) | 81.3% |

learning rate는 ViT 2.5e-4(DeiT 규칙), ResNet 1e-3이다. ViT에 1e-3을 쓰면 100% 데이터에서 63.5%(train 정확도 65%)로 과소적합한다. 그 첫 실행 결과는 `outputs_lr1e-3/`에 남겨 두었다. 해석은 블로그 글의 "미니 재현 실험" 섹션 참고.
