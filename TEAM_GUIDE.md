# 팀 실행·협업 안내

현재 500개 baseline과 2,000개 비교는 [v2 재현·전달 문서](docs/BASELINE_V2_HANDOFF.md)를 우선한다. 아래 160/30 데이터 예시는 이전 소규모 연결 점검이다.

## 저장소 범위

Memory Maze 9×9의 저장 RGB·action으로 LeWM을 학습하고, 초기 관측 이후 예측 latent에서 빨간 공의 상대 좌표를 읽는 평가 파이프라인이다. Simulator를 실행하지 않고 공식 NPZ를 사용한다.

이 저장소는 소스·설정·테스트를 관리한다. 데이터·학습 가중치·Probe·실험 로그는 Git에 올리지 않는다. 계정·비밀번호·SSH 키도 포함하지 않는다. 기존 README의 `artifacts/` 파일은 별도 전달 자료이며 clone에 포함되지 않는다.

## 첫 실행

아래 명령은 clone한 저장소 루트에서 실행한다. 클러스터에서는 설치·파일 전송만 로그인 노드에서 수행하고 데이터 처리·테스트·학습·Probe 등 부하 작업은 `jobsubmit`으로 제출한다.

```bash
git clone https://github.com/chest-press-man/memorymaze-lewm.git
cd memorymaze-lewm
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-linux.txt
python -m pytest -q
```

클러스터에서 Conda를 사용해도 된다. Python 3.11 환경을 만든 뒤 같은 requirements를 설치한다. 맥북 Intel CPU 점검에는 `requirements-macos-intel.txt`를 사용한다. 데이터·원본 가중치에 의존하는 통합 테스트는 자료가 없으면 skip될 수 있다.

## 데이터 준비

```bash
python -m memorymaze_lewm acquire --root data --train 160 --eval 30 --eval-skip 10
python -m memorymaze_lewm prepare --root data
```

공식 공개 데이터 중 train 160개·eval 30개를 확보하는 설정이다. 공식 데이터 전체를 받는 명령은 아니다. 취득 시 크기·CRC·필드 형식을 검사하고 SHA-256을 기록한다. 기존 점검 eval 10개는 건너뛴다.

| 용도 | Episode 수 |
|---|---:|
| 모델 학습 / 검증 | 100 / 20 |
| Probe 학습 / 검증 후보 | 30 / 10 |
| 최종 test 후보 | 30 |

`data/acquisition.json`과 `data/splits.json`은 실행 장비에서 생성한다. 모델 비교 때 episode 목록을 고정한다. Episode 독립성과 미로 배치 독립성은 다르며, 배치 중복은 splits 기록으로 확인한다.

## 학습과 평가

학습은 Encoder와 Transformer Predictor를 함께 갱신하는 단계다. 이후 선택한 모델을 고정하고 미래 예측 latent로 linear Probe를 학습한다. Probe 출력은 빨간 공의 상대 좌표 2개다.

1. 기본 구조를 처음부터 학습하거나 기존 Maze checkpoint를 별도로 전달받아 초기화한다. Cube 공개 가중치 활용은 기존 README의 `pretrained` 경로를 참고한다. 초기화 방식을 실험마다 기록한다.
2. 클러스터에서는 새 설정 파일에 CPU 스레드·device·batch·학습 예산을 정한다. 학부생 GPU CPU 3개 요청이면 `threads=3`으로 맞춘다. 기존 설정 파일을 덮어쓰지 않는다.
3. 짧은 5–10 update 실행으로 forward/backward·checkpoint 저장·재로드를 확인한 뒤 새 run으로 본 학습을 시작한다. Best는 모델 validation으로만 선택한다.
4. `segments`로 평가 후보를 만들고 초기 화면을 검수한다. 별도 전달받은 `review_decisions.json`이 있으면 원본·구간 ID 일치를 확인해 재사용한다. 없으면 README 형식으로 직접 검수한다.
5. 선택 checkpoint로 `rollout`을 실행한 뒤 `probe`를 실행한다. 미래 RGB·정답 좌표를 rollout 모델에 입력하지 않는다.
6. 모델과 평균 좌표 기준의 horizon별 오차를 함께 확인한다. 낮은 학습 loss만으로 공 위치 기억을 주장하지 않는다.

```bash
# 자료 확보 후 실행 가능한 CPU 연결 점검; 본 학습이나 성능 검증을 뜻하지 않음
python -m memorymaze_lewm train --config configs/pilot.json \
  --splits data/splits.json --run runs/pilot_local_001

# 실제 검수 후 아래 파일을 준비해야 함
python -m memorymaze_lewm segments --splits data/splits.json --out data/segments_candidates.json
python -m memorymaze_lewm review --segments data/segments_candidates.json \
  --decisions data/review_decisions.json --out data/segments_reviewed.json

python -m memorymaze_lewm rollout --checkpoint runs/pilot_local_001/checkpoints/best.pt \
  --segments data/segments_reviewed.json --device cpu --out runs/pilot_local_001/latents
python -m memorymaze_lewm probe --latents runs/pilot_local_001/latents --out runs/pilot_local_001/probe
```

출력 폴더는 새 경로를 사용한다. 기존 모델에서 새 run을 시작할 때는 `--init-checkpoint`, 같은 run을 이어갈 때는 동일 config·split의 `--resume`을 사용한다. 최신 모델에는 그 모델에서 새로 학습한 Probe를 짝지어야 한다.

## 모델 교체

기존 README의 `모델 교체` 절에 입출력 규약이 있다.

- 같은 학습 규약: `build_model(config)`와 `encode`, `predict`, `action_encoder`를 제공하고 `--model-factory`로 연결한다.
- 다른 학습·저장 구조: 평가용 `build_adapter(checkpoint, device)`를 만들고 `--factory`로 연결한다.
- 모델이 바뀌면 같은 구간·Probe 학습 규칙으로 새 Probe를 학습한다. 모든 모델에 동일 Probe 가중치를 적용하지 않는다.
- Recurrent 모델은 구간 시작마다 memory를 초기화하고, 미래 관측·정답을 입력받지 않는다.

`tests/test_model.py`는 교체 경계를 검증하는 예제를 포함한다. 임의의 새 모델이 추가 작업 없이 자동 호환된다는 뜻은 아니다.

## 팀원 합류

저장소 소유자가 GitHub의 **Settings → Collaborators**에서 팀원의 GitHub ID를 초대한다. 팀원은 초대를 수락한 뒤 clone한다. 아직 누구에게도 초대를 보내지 않았다.

```bash
git switch main
git pull --ff-only
git switch -c feat/작업이름
# 수정과 관련 테스트 실행 후
# 변경한 파일만 명시적으로 add
git add 경로/파일
git commit -m "변경 내용"
git push -u origin HEAD
```

GitHub에서 Pull Request를 만들고 팀원이 검토한 뒤 main에 합친다. 동일 실험 비교에서는 데이터·전처리·초기화·학습 예산·checkpoint 선택·평가 구간을 함께 기록한다. 실제 학습 기록은 각 장비 `runs/RUN_ID/`에 남기고 필요한 결과 요약만 별도로 공유한다.

모델·데이터 전달은 Git 코드 버전과 대응 파일 SHA-256을 함께 기록한다. 대형 파일을 commit하거나 비밀번호를 README·설정에 적지 않는다.

## 원본 출처

`memorymaze_lewm/vendor/SOURCES.json`에 upstream commit·출처·hash를 기록했다. 포함된 LeWM 및 stable-worldmodel 라이선스 파일을 유지한다. 데이터·공개 가중치의 사용 조건은 해당 원본 배포 조건을 따른다.
