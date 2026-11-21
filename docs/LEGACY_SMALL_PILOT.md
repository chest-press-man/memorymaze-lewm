> 과거 문서 원문 보존본. 현재 실행은 [v2 재현 문서](BASELINE_V2_HANDOFF.md)를 따른다. 아래 상대경로는 과거 저장소 루트 기준이다.

# Memory Maze LeWM 파이프라인

> **현재 실험: 500개 baseline v2.** LeWM train/val 500/100, Probe 후보 train/val/test 300/100/200. 0-step 관측 latent와 1~50 모든 예측 latent를 구분해 단일 Probe로 읽는다. 실행·선택 기준은 [실험 계획](docs/BASELINE_500_PLAN.md), 정확한 재실행·2,000개 비교는 [재현·팀 전달](docs/BASELINE_V2_HANDOFF.md)을 따른다. 아래 소규모 실험 설명은 이전 실행 기록이다.

> **GitHub 소스 저장소:** 데이터·가중치·실험 결과·가상환경은 포함하지 않는다. 처음 clone한 팀원은 [팀 실행·협업 안내](TEAM_GUIDE.md)를 먼저 따른다. 아래 `artifacts/` 관련 설명은 기존 파일 전달 방식이며 clone만으로 해당 파일이 생기지는 않는다.
>
> 기존 연구실 GPU 학습·Probe 평가 기록은 로컬에 별도 보관돼 있다. 아래의 GPU 미실행 설명은 과거 CPU 점검 시점의 기록이며, 이 저장소가 클러스터 실행까지 검증됐다는 뜻은 아니다. 장기 기억 성능 개선은 아직 입증되지 않았다.

공식 Memory Maze 9×9 데이터를 읽어 기본 LeWM을 학습하고, 초기 RGB 3장 이후 50-step latent rollout을 생성한 뒤 linear Probe로 빨간 공의 상대 좌표를 평가한다.

연구실 컴퓨터의 Codex에 이어 맡길 때는 **[HANDOFF_CODEX.md](HANDOFF_CODEX.md)**를 사용한다. 공유 ZIP을 그 컴퓨터에 복사하면 SSH 없이 직접 이어서 실행할 수 있다.

## 실행 범위

- 데이터: 공식 NPZ, 64×64 RGB, 6차원 one-hot action.
- 전처리: 목표색 2픽셀 테두리 제거, ImageNet 정규화, bilinear 224×224. 원본은 보존.
- 학습: 화면 4장·전이 action 3개. 입력 latent 3개와 한 칸 뒤 정답 latent 3개를 비교하는 공식 MSE+0.09 SIGReg.
- 초기화: 공식 Cube checkpoint의 호환 구조를 엄격하게 로딩한 뒤 6-action encoder를 새로 초기화하고 전체 모델을 fine-tuning. `--pretrained` 생략 시 같은 기본 구조를 처음부터 학습.
- 평가: 미래 RGB·좌표는 모델에 전달하지 않음. 1·3·5·10·20·30·40·50 step의 예측 latent만 Probe 입력으로 사용.
- 모델 교체: 동일한 데이터·구간·행동·Probe·평가를 유지하면서 model factory 또는 rollout adapter만 교체.

이번 구현은 기존 보고서의 실행 구체화다. 보고서의 scratch-only 계획과 달리 **기존 가중치 재사용 경로를 기본 실행으로 선택**했다. 또한 실행 관리는 최소 PyTorch runner를 사용한다. LeWM 구조와 loss는 upstream 원문을 유지하며, Lightning·stable-worldmodel 전체 패키지·simulator 설치는 필요하지 않다. 공식 전체 학습 recipe를 그대로 재현했다는 뜻은 아니다.

## 설치

프로젝트 루트에서 실행한다. 기존 Python 설치에 패키지를 직접 추가하지 않는다.

```bash
# Ubuntu/클러스터: Python 3.11 권장
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-linux.txt
python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())'
python -m pytest -q
```

맥북 Intel에서 연결을 점검하는 경우 `requirements-linux.txt` 대신 `requirements-macos-intel.txt`를 사용한다. 이 환경은 CUDA가 없는 CPU 점검용이다. Linux requirements는 준비했지만 실제 연구실 GPU/클러스터에서 설치·실행을 검증한 것은 아니다. 실행 전 `nvidia-smi`와 가용 저장 공간·메모리를 확인한다.

## 데이터 확보·준비

```bash
python -m memorymaze_lewm acquire --root data --train 160 --eval 30 --eval-skip 10
python -m memorymaze_lewm prepare --root data
python -m memorymaze_lewm pretrained --out data/pretrained/cube
```

전체 100GB ZIP을 받지 않고 공식 ZIP의 필요한 episode만 HTTP Range로 추출한다. 외부 ZIP CRC·NPZ 필드 검사·SHA-256을 기록하고 이미 받은 정상 파일은 재사용한다. Range 응답이 다르거나 소스 크기가 바뀌면 중단한다. 실패했다면 같은 acquire 명령으로 재실행한다.

train 160개를 모델 학습 100·모델 검증 20·Probe 학습 30·Probe 검증 10으로 파일명 순서에 따라 분리한다. 최종 test는 기존 점검용 eval 10개를 건너뛴 신규 30개다. 같은 episode는 용도 사이에 겹치지 않는다. 미로 배치 자체의 중복은 `data/splits.json`에 측정해서 기록하며 배치까지 분리됐다고 주장하지 않는다.

장비로 폴더를 옮기면 `prepare`를 다시 실행해 현재 데이터 루트에 맞는 split 경로를 만든다. split의 파일명이 기존 실행과 같은지 확인하고 새 run을 사용한다. 기존 run 재시작에는 당시 split·config를 그대로 사용해야 한다.

## 구간 후보·검수

```bash
python -m memorymaze_lewm segments --splits data/splits.json --out data/segments_candidates.json
```

공식 기본 색 매핑의 빨간 공 ID 0을 사용하고, 기록된 현재 목표색·좌표에서 확인 가능한 경우 매핑을 교차 검사한다. 초기 3장에 붉은 픽셀이 있고, 공의 상대 위치가 정면 중심에 있으며, 벽 ray 검사를 통과하고, 미래 50 step에 붉은 픽셀 0인 연속 5개 화면이 있는 구간을 찾는다. 이 필터는 **정확한 자동 공 가시성 판정기가 아니다.** 붉은 벽·작은 공·부분 가림 때문에 사람이 초기 3장을 확인해야 한다.

`data/review/candidates_*.png`에서 초기 3장과 미래 확인 화면을 읽는다. `review_decisions.json`은 다음처럼 작성한다.

```json
{
  "EPISODE_ID:ORIGIN_STEP": {
    "status": "approved",
    "note": "초기 3장에서 같은 빨간 공 식별 가능, 테두리 잘림 없음"
  }
}
```

```bash
python -m memorymaze_lewm review --segments data/segments_candidates.json --decisions data/review_decisions.json --out data/segments_reviewed.json
```

미기재 구간은 pending으로 남고 기본 rollout은 approved만 사용한다. `--allow-unreviewed`는 연결 디버깅에만 사용하며 해당 결과는 exploratory로 표시한다. 초기 관측 검수와 미래 50장 전체의 가시성 라벨 검수는 별개다. 현재 숨김 상태별 지표는 휴리스틱 결과이므로 최종 기억 성능 근거로 사용하지 않는다.

## LeWM 학습·저장·재시작

```bash
# 짧은 CPU 연결 점검: 20 update
python -m memorymaze_lewm train --config configs/pilot.json --splits data/splits.json --pretrained data/pretrained/cube --run runs/pilot_001

# 초기 적응 실험: CPU 300 update, 충분한 본 학습을 뜻하지 않음
python -m memorymaze_lewm train --config configs/initial.json --splits data/splits.json --pretrained data/pretrained/cube --run runs/initial_001

# GPU 본 학습 시작안: batch 4·10,000 update. 자원 측정 후 새 config에서 규모 확정
python -m memorymaze_lewm train --config configs/team.json --splits data/splits.json --pretrained data/pretrained/cube --run runs/team_001

# 중단한 실행 재시작: 동일 config와 split, 동일 model-factory 옵션 사용
python -m memorymaze_lewm train --config configs/team.json --splits data/splits.json --run runs/team_001 --resume
```

새 run은 빈 경로만 허용한다. 완료된 run을 같은 이름으로 새로 학습해 덮어쓰지 않는다. config·split hash가 달라지면 resume을 거부한다. 모델·optimizer·update·best 지표·sampler/RNG를 저장하며 learning rate는 복원한 update로 재계산한다. 현재 runner는 FP32이다. FP16/BF16을 구현·검증 없이 지원한다고 가정하지 않는다.

학습 화면과 action만 모델에 넣는다. 좌표 supervision을 LeWM loss에 추가하지 않는다. action 통계는 모델 train의 실제 전이 100,000개로 계산하며 초기 0벡터는 제외한다. 같은 action을 겹치는 window 때문에 반복 집계하지 않는다.

last checkpoint는 재시작용이다. best checkpoint는 validation prediction MSE와 유효 latent 분산으로 고르며 test 오차로 선택하지 않는다. 낮은 latent MSE만으로 좋은 공간 표현·기억을 확보했다고 판단하지 않는다.

## 저장한 Maze 가중치로 팀 실험 시작

이번 초기 실험의 `inference.pt`는 이미 6-action Memory Maze 구조다. 이를 새 장비에 옮겨 더 학습하려면 다음처럼 **새 run을 warm-start**한다. optimizer·학습 스케줄을 이어가는 resume과 구분한다.

```bash
python -m memorymaze_lewm train --config configs/team.json --splits data/splits.json --init-checkpoint artifacts/inference.pt --run runs/team_from_maze_001
```

`--pretrained`와 `--init-checkpoint`를 함께 지정하지 않는다. 개선 모델도 동일한 Maze 초기 가중치를 사용하려면 `--model-factory your_team_model:build_model`을 함께 지정한다. 호환되는 공유 parameter와 새 parameter 목록을 기록한다.

## Rollout·Probe·평가

```bash
python -m memorymaze_lewm rollout --checkpoint runs/initial_001/checkpoints/best.pt --segments data/segments_reviewed.json --device cpu --out runs/initial_001/latents
python -m memorymaze_lewm probe --latents runs/initial_001/latents --out runs/initial_001/probe
```

GPU에서는 rollout의 `--device cuda`를 사용한다. 출력 디렉터리는 새 경로여야 한다. rollout에는 초기 화면 3장·그 사이 action 2개·미래 action 50개만 전달한다. 각 평가 시점에서 원본 `targets_vec[origin+h,0,:]`를 별도로 정답에 연결한다.

Probe는 모델마다 별도 ridge linear regression으로 학습한다. 입력은 최종 예측 상태 latent, 출력은 상대 좌표 2개다. Probe train latent에서 구한 mean·std만 적용하며 λ는 `{0,0.0001,0.001,0.01,0.1,1}` 중 별도 Probe validation으로 선택한다. epoch/step별 별도 Probe를 쓰지 않는다.

결과는 구간→episode 순서로 평균해 episode마다 같은 비중을 준다. 1-step 대비 오차 증가량과 train 좌표 평균만 출력하는 기준도 저장한다. `metrics.json`·`metrics.csv`·`predictions.npz`·`position_error.png`가 생성된다. 실패 구간은 목록과 성공률을 남긴다. 모델끼리 비교할 때는 같은 test 목록과 공통 성공 구간을 함께 확인한다.

보이는 공의 현재 위치가 실제 관측 latent에서 읽히는지는 별도 진단으로 확인한다. 이 결과는 미래 예측 결과와 합치지 않는다.

```bash
python -m memorymaze_lewm diagnose --checkpoint runs/initial_001/checkpoints/best.pt --segments data/segments_reviewed.json --out runs/initial_001/observed_diagnostic
```

현재 이 진단 명령은 episode당 검수 구간 1개인 조건을 지원한다.

## 모델 교체

### 같은 학습 입출력·loss를 사용하는 개선 모델

별도 Python 모듈에 `build_model(config) -> torch.nn.Module`을 구현한다. 모델은 다음 인터페이스를 유지한다.

```text
encode({pixels: [B,4,3,224,224], action: [B,3,6]})
  → {emb: [B,4,D], act_emb: [B,3,A]}
predict(emb: [B,3,D], act_emb: [B,3,A]) → [B,3,D]
action_encoder(action: [B,T,6]) → [B,T,A]
선택: reset_episode() → memory 초기화 (학습 batch/평가 구간 시작마다 호출)
```

```bash
# 아래 your_team_model은 팀이 작성할 실제 모듈명으로 바꿔야 함
python -m memorymaze_lewm train --config configs/team.json --splits data/splits.json --pretrained data/pretrained/cube --model-factory your_team_model:build_model --run runs/improved_001
```

factory 경로는 checkpoint에 저장돼 이후 재로딩에도 사용된다. 위 모듈 파일은 실행 장비에도 있어야 한다. pretrained와 key·shape가 맞는 부분만 복사하고 새 parameter·사용하지 못한 source parameter 목록을 기록한다. 기본·개선 모델의 공유 부분 초기값·Maze 학습 데이터·예산·checkpoint 선택 규칙을 맞춘다. loss 자체를 바꾸는 방법은 그 방법의 학습 코드를 제공하고 아래 rollout adapter 경계로 연결한다.

### 다른 저장·추론 API를 사용하는 모델

`build_adapter(checkpoint,device)`를 구현해 `rollout(initial_images,past_actions,future_actions) -> numpy.ndarray[50,D]`를 제공한다. 구간별 memory reset을 이 함수에서 보장하고 미래 RGB/정답을 요구하면 안 된다.

```bash
python -m memorymaze_lewm rollout --checkpoint YOUR_CHECKPOINT --segments data/segments_reviewed.json --device cuda --factory your_team_adapter:build_adapter --out runs/improved_001/latents
python -m memorymaze_lewm probe --latents runs/improved_001/latents --out runs/improved_001/probe
```

공통 수집·구간·정답·Probe·평가 코드를 수정하지 않는다. `tests/test_model.py`의 alternate-model checkpoint 테스트가 이 교체 경계를 검증한다. 전체 memory bank를 Probe로 읽는 조건은 현재 기본 비교 범위에 포함하지 않는다.

## 장비 운영·현재 한계

- 개인 사전 점검은 맥북 CPU에서 실제 실행했다. RTX 2080·클러스터 GPU 성능으로 해석하지 않는다.
- 기존 SSH 대상은 timeout이어서 현재 실제 GPU 학습은 미실행이다. 접속 정보·네트워크가 확보되면 같은 코드를 전송하고 Linux 환경을 확인해야 한다.
- simulator 렌더링은 필요 없다. 저장 NPZ로 학습·평가한다.
- SSH에서 긴 실행은 해당 장비 정책에 맞춰 tmux 또는 클러스터 작업 제출 방식을 사용한다. 다른 작업을 종료하거나 GPU를 독점하지 않는다.
- 지연 시간은 rollout 경과 시간을 기록하지만 warm-up·반복·CUDA event 기반의 최종 벤치마크는 별도 실행이 필요하다. 실시간성을 검증한 결과가 아니다.
- 모델이 학습되고 Probe 파일이 만들어졌다는 것과 장기 기억 성능이 확보됐다는 것은 다르다. 실제 실행 범위·수치는 `RESULTS.md`에 구분해 기록한다.

## 공유 파일 사용 순서

압축을 푼 프로젝트 루트에서 설치를 마친 뒤 실행한다. `artifacts/inference.pt`와 `artifacts/probe.npz`는 이번에 실제 생성한 모델·Probe다. 데이터 NPZ·가상환경·대형 재시작 checkpoint는 전달 ZIP에 포함하지 않는다.

```bash
python -m memorymaze_lewm acquire --root data --train 160 --eval 30 --eval-skip 10
python -m memorymaze_lewm prepare --root data
python -m memorymaze_lewm segments --out data/segments_candidates.json
python -m memorymaze_lewm review --segments data/segments_candidates.json --decisions artifacts/review_decisions.json --out data/segments_reviewed.json
python -m memorymaze_lewm rollout --checkpoint artifacts/inference.pt --segments data/segments_reviewed.json --device cpu --out runs/shared_001/latents
python -m memorymaze_lewm probe --latents runs/shared_001/latents --out runs/shared_001/probe
```

장비를 옮겼을 때는 **split뿐 아니라 segments도 재생성**해 새 데이터 경로를 적용한다. 같은 원본과 코드이면 구간 ID가 같아 검수 결정을 재사용할 수 있다. 원본 hash·승인 구간 수가 `artifacts/`의 전달 기록과 일치하는지 확인한다. 코드·구간 규칙·데이터가 달라지면 검수도 다시 해야 한다. 이어서 팀 GPU 학습에는 위의 `--init-checkpoint artifacts/inference.pt` 새 run 경로를 사용한다.

Probe의 실제 목적함수는 `sum_i w_i × ||예측좌표_i−정답좌표_i||² + λ × ||W||²`이며 bias는 정규화하지 않는다. 좌표 두 성분의 제곱오차 합을 사용하므로, 좌표 성분 평균 MSE를 쓰는 외부 solver로 바꾸면 λ의 배율도 맞춰야 한다.

## 파일

```text
memorymaze_lewm/  수집·전처리·모델·학습·구간·rollout·Probe·평가
configs/         pilot·initial·team 설정
memorymaze_lewm/vendor/  commit 고정 공식 모델 원문·라이선스·SOURCES.json
 tests/          정렬·경계·누수·causal·회귀·집계·모델 교체 테스트
 data/           원본·출처/hash·split·검수 자료 (설치·공유 코드와 분리)
 runs/RUN_ID/    설정·로그·last/best·latent·Probe·결과
```

공식 원문은 `vendor/SOURCES.json`의 URL·commit·hash로 추적한다. 공개 가중치는 지정된 Cube revision과 SHA-256으로 확인한다. 데이터·가중치·가상환경은 소스 압축파일과 분리하며 실제 전송·공개는 수행하지 않았다.
