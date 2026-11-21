# 확보 모델 사용·개선 접점

## 바로 평가

GitHub 소스와 baseline release의 모델·Probe·protocol 자료를 같은 버전으로 준비한다. 데이터 확보·분할·검수 결정 적용은 `BASELINE_V2_HANDOFF.md`를 따른다. 새 출력 폴더를 사용한다.

```bash
python -m memorymaze_lewm rollout --checkpoint artifacts/baseline500_v2/inference.pt \
  --segments data/segments_reviewed.json --device cuda --threads 3 \
  --max-step 50 --out runs/replay500_001/latents
python -m memorymaze_lewm probe --latents runs/replay500_001/latents \
  --out runs/replay500_001/probe
```

두 번째 명령은 같은 규칙으로 Probe를 다시 학습·선택하는 재현 경로다. 제공된 Probe를 그대로 사용하려면 아래처럼 `predict`를 호출한다.

## 코드에서 사용

```python
import numpy as np
from memorymaze_lewm.model import load_adapter
from memorymaze_lewm.probe import predict

adapter = load_adapter('artifacts/baseline500_v2/inference.pt', device='cuda')
with np.load('artifacts/baseline500_v2/probe.npz', allow_pickle=False) as f:
    readout = {k: f[k] for k in f.files}

# images: uint8 (3,64,64,3), raw RGB
# past_actions: float32 (2,6), raw recorded one-hot actions
# future_actions: float32 (50,6), raw recorded one-hot actions
z0 = adapter.initial_latent(images)
z_future = adapter.rollout(images, past_actions, future_actions)
z = np.concatenate([z0[None], z_future], axis=0)  # (51,192)
relative_xy = predict(readout, z)                  # (51,2)
```

Adapter에서 테두리 마스킹·resize·영상 정규화·action 정규화를 수행한다. 외부에서 같은 처리를 중복 적용하지 않는다. 제공하는 미래 정보는 **action뿐**이다. 미래 RGB·agent/target 좌표·지도는 모델 입력이 아니다. 좌표 정답은 평가기가 따로 연결한다.

0번 행은 마지막 초기 관측의 readout이고, 1~50번 행은 각 미래 예측의 readout이다. 출력 단위는 원본 Memory Maze ego-relative 평면 좌표다.

## 모델을 개선할 때

| 변경 | 수정 위치·방법 | 고정할 것 |
|---|---|---|
| 같은 학습 규약의 모델 구조 | 팀 모듈에 `build_model(config)` 구현, train `--model-factory` 사용 | 입력 규약·학습 데이터·초기화·예산·loss·선택 규칙 |
| 저장 형식·추론 구조가 다른 모델 | 팀 모듈의 `build_adapter(checkpoint,device)`, rollout `--factory` 사용 | 초기 관측·기록 action·평가 구간·정답 |
| Memory state 추가 | 구간/학습 window 시작 시 reset, 정의된 latent 반환 | 다른 episode의 memory 유입 금지 |
| 데이터 500→2,000 | 공식 train 목록 1000~2499만 model-train에 추가 | LeWM val·Probe/test·normalizer·config·초기 가중치 |

Adapter 계약은 `initial_latent(initial_images)->[D]`, `rollout(initial_images,past_actions,future_actions)->[T,D]`다. 실제 미래 RGB나 좌표를 받는 인자는 추가하지 않는다.

새 모델마다 latent를 다시 추출하고 같은 ridge 규칙으로 Probe를 새로 학습한다. 입력 차원·Probe parameter 수·λ와 모델 parameter 수를 기록한다. 서로 다른 모델의 latent 축이 같다고 가정하지 않는다.

기존 trainer는 4-frame window로 1-step 세 위치를 공동 학습한다. 긴 시퀀스 학습이 필요한 recurrent 방법을 모듈 하나만 바꿔 같은 방식으로 학습 가능하다고 가정하면 안 된다. 그 경우 별도 학습 경로와 비교 조건을 정의한 뒤 공통 평가 adapter에 연결한다.

검증: `tests/test_model.py`는 모델 교체·causal rollout을, `tests/test_dense_protocol.py`는 0-step/연속 step 정렬·분할 보존·선택 gate를 검사한다. GitHub 기여는 별도 브랜치 → PR이며 main 병합은 소유자만 한다.

## 학습 모델 인터페이스

`build_model(model_config)`의 인자는 학습 hyperparameter JSON이 아니라 encoder·predictor·action encoder·projector 설정을 담은 **모델 구조 config**다. 아래 규약을 제공하는 `torch.nn.Module`을 반환한다.

```text
encode({pixels: [B,4,3,224,224], action: [B,3,6]})
  -> {emb: [B,4,D], act_emb: [B,3,A]}
predict(emb: [B,3,D], act_emb: [B,3,A]) -> [B,3,D]
action_encoder(action: [B,T,6]) -> [B,T,A]
reset_episode() -> 선택 구현; window/평가 구간 시작의 memory 초기화
```

별도 평가 API의 `build_adapter(checkpoint,device)`에서 `checkpoint`는 **이미 torch.load한 dict**다. 파일 경로를 다시 여는 인자가 아니다. 모듈의 import 경로는 checkpoint에 저장되므로 팀원의 실행 환경에도 같은 모듈이 있어야 한다.
