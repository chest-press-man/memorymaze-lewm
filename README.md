# Memory Maze LeWM Baseline

공식 Memory Maze 9×9 NPZ에서 LeWM을 공동 학습하고, 초기 관측 이후 예측 latent의 빨간 공 상대 좌표를 단일 linear Probe로 평가한다.

## 문서

- [실제 실행 결과·한계](docs/BASELINE500_RESULTS.md)

- [실험 계획·선택 기준](docs/BASELINE_500_PLAN.md)
- [500개 재현·2,000개 동일 조건 비교](docs/BASELINE_V2_HANDOFF.md)
- [모델·Probe 로딩 및 개선 접점](docs/BASELINE_MODEL_USAGE.md)
- [팀 협업 안내](TEAM_GUIDE.md)
- [이전 소규모 점검 기록](docs/LEGACY_SMALL_PILOT.md)

## 고정 조건

| 항목 | Baseline v2 |
|---|---|
| 데이터 | LeWM train/val 500/100, Probe 후보 train/val/test 300/100/200 |
| 원본·실제 구간 | 공식 NPZ 1,200개, 초기 관측 승인 208/72/156 구간 |
| 학습 | ViT-Tiny Encoder·Transformer Predictor 공동 fine-tuning, 10,000 update·batch 16·seed 42 |
| 입력 | RGB 64×64→테두리 마스킹·224 resize·ImageNet 정규화, history 3·frameskip 1·action 6 |
| 예측 | 초기 실제 RGB 3장 이후 기록 action만 제공. 미래 RGB·좌표는 모델 입력 금지 |
| Readout | 0-step 관측 + 1~50 모든 예측 latent, 단일 ridge Probe, validation으로 λ 선택 |
| 선택·test | LeWM model-val로 best 선택, Probe-val로 λ 선택, 최종 test는 설정 선택에서 제외 |
| 실행 | 학과 클라우드 RTX 4000 Ada, CPU 3·RAM 20G·GPU 1. 계산은 jobsubmit 사용 |
| 2,000개 비교 | model-train만 확대. holdout·Probe·test·초기화·normalizer·학습 예산 고정 |

500개 학습은 모든 원본 프레임을 한 번씩 처리했다는 뜻이 아니다. 160,000개 window를 복원 추출하는 고정 예산이다. 가시성의 미래 hidden 표시는 휴리스틱이며, 낮은 latent MSE나 좋은 Probe 결과만으로 기억 보존을 인과적으로 입증하지 않는다.

## 설치·데이터

Python 3.11 환경에서 프로젝트 루트 기준으로 실행한다. 클러스터에서는 설치·파일 전송 외의 아래 계산 명령을 Job으로 제출한다.

```bash
python -m pip install -r requirements-linux.txt
python -m pytest -q
python -m memorymaze_lewm acquire --root data --train 1000 --eval 200 --eval-skip 40
python -m memorymaze_lewm validate --root data --reference protocol/data_reference.json
python -m memorymaze_lewm prepare --root data --scaled-protocol --model-train 500
python -m memorymaze_lewm pretrained --out data/pretrained/cube
```

학습·검수·rollout·Probe 명령은 재현 문서를 따른다. 같은 정상 원본은 재사용하고 224 float32 영상을 전체 저장하지 않는다. 2,000개 실험은 acquire의 train=2500, prepare의 model-train=2000만 바꾸며 공통 데이터 SHA를 reference와 대조한다.

데이터·대형 가중치·로그·가상환경·접속 정보는 소스 저장소에 포함하지 않는다. 모델·Probe는 별도 baseline 배포 묶음으로 제공한다. 공식 모델 원문·MIT 라이선스·commit/hash는 `memorymaze_lewm/vendor/`에 보존한다.

## 모델 교체 계약

- 학습: 팀 모듈의 `build_model(config)`를 `--model-factory`로 연결. encode/predict/action_encoder 규약은 모델 사용 문서 참고.
- 평가: `build_adapter(checkpoint,device)`를 `--factory`로 연결. initial_latent→[D], rollout→[T,D]. 각 구간에서 memory 초기화.
- 모델마다 같은 episode·step·선택 규칙으로 Probe 재학습. 다른 latent 축에 동일 Probe 가중치를 공유하지 않는다.
- 수정·기여는 별도 브랜치와 PR. main 병합은 저장소 소유자가 통제한다.
