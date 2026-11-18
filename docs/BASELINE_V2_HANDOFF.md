# Baseline v2 재현·팀 전달

## 공통 조건

LeWM train 500개와 2,000개 실험은 **모델 학습 episode 수만 다르다.** 실제 GPU·Python·PyTorch 버전도 동일하게 맞추고 실행 환경 기록을 대조한다. 모델 검증 100·Probe 학습 후보 300·Probe 검증 후보 100·test 후보 200, 초기 가중치 revision, action normalizer의 공통 500개, seed 42, batch 16, 10,000 update, 학습률·loss·모델 선택·구간·0~50 readout 규칙을 고정한다. 두 실험은 같은 160,000 window 추출 예산이며 전체 epoch 수를 맞추는 비교가 아니다.

공식 train 파일명을 정렬하고 기존 첫 160개의 100/20/30/10 역할을 유지한다. 추가 인덱스 160~559는 model train, 560~639는 model val, 640~909는 Probe train, 910~999는 Probe val이다. 2,000개 실험은 1000~2499를 model train에 추가한다. 큰 학습 목록의 첫 500개가 작은 학습 목록과 같아야 한다. Eval은 기존 QA·과거 평가 40개를 건너뛴 200개다. 물리 폴더에 이전 캐시가 더 있어도 `acquisition.json`과 `splits.json`에 기록된 목록만 사용한다.

초기 공 가시성을 검수한 구간만 최종 Probe에 사용한다. 후보 수와 실제 승인 수는 다르므로 결과에 모두 기록한다. 미래 가시성 휴리스틱은 정밀 객체 mask가 아니다. 유효 구간 목표는 Probe train 200·val 50·test 100개 이상이다. 0-step은 관측 readout이며 1~50은 실제 미래 화면 없이 생성한 예측 readout이다.

## 환경과 데이터

프로젝트 루트에서 Python 3.11과 `requirements-linux.txt`를 설치한다. 클러스터의 로그인 서버에서는 부하 작업을 실행하지 않고 아래 처리 명령을 Job으로 제출한다. CPU 3개 요청이면 config `threads=3`을 유지하고, 모든 출력·캐시·임시 파일을 workspace에 둔다.

```bash
# 500개 실험
python -m memorymaze_lewm acquire --root data --train 1000 --eval 200 --eval-skip 40
python -m memorymaze_lewm prepare --root data --scaled-protocol --model-train 500

# 팀원의 2,000개 실험: 이 두 명령의 train 규모만 다름
python -m memorymaze_lewm acquire --root data --train 2500 --eval 200 --eval-skip 40
python -m memorymaze_lewm prepare --root data --scaled-protocol --model-train 2000

# 두 실험이 같은 공개 초기 가중치를 사용
python -m memorymaze_lewm pretrained --out data/pretrained/cube
```

원본 데이터와 다운로드 SHA는 `data/acquisition.json`에 기록된다. 추가 자료를 model-train에 넣으려고 model-val·Probe·test를 옮기지 않는다. `splits.json`의 `normalizer_train`은 두 규모 모두 같은 500 episode다. Fixed holdout·공통 normalizer·nested train 목록은 `tests/test_dense_protocol.py`로 검사한다.

## 클라우드 실행

전용 환경은 `~/workspace/.envs/lewm-baseline-v2`에 준비하고 `scripts/cluster_run.sh`를 Job 명령의 공통 wrapper로 사용한다. 이 wrapper는 CPU 스레드 3, 캐시·임시 파일을 workspace에 고정한다. `scripts/prepare_cloud.py 500`은 validate → 고정 분할 → 구간 생성을 수행한다. 2,000개에는 마지막 인자만 2000을 쓴다.

```bash
jobsubmit -c 3 -m 20G -G 0 -t 0-00:30:00 bash scripts/cluster_run.sh scripts/prepare_cloud.py 500
jobsubmit -p 4000ada -c 3 -m 20G -G 1 -t 0-04:00:00 bash scripts/cluster_run.sh -m memorymaze_lewm train --config configs/baseline500.json --splits data/splits.json --pretrained data/pretrained/cube --run runs/baseline500_001
```

첫 Job이 정상 종료된 뒤 다음 단계를 제출한다. 같은 본 학습 Job의 중단 재시작은 동일 config·split과 `--resume`으로 수행한다.

## 공동 학습과 모델 선택

```bash
python -m memorymaze_lewm train --config configs/baseline500.json \
  --splits data/splits.json --pretrained data/pretrained/cube \
  --run runs/baseline500_001
```

2,000개 실험은 같은 config를 사용하고 run 이름만 `baseline2000_001`로 바꾼다. 새 run을 사용하며 이전 500개 모델을 2,000개 실험의 초기 가중치로 쓰지 않는다. 그러면 초기 조건과 누적 연산량이 달라진다.

학습 입력은 4 RGB frame·3 outgoing action이다. Encoder·projector·action encoder·Transformer Predictor 등을 공동 갱신한다. 목표는 한 칸 이동한 실제 latent 3개이며 위치 정답은 LeWM loss에 넣지 않는다. Prediction MSE와 SIGReg weight .09를 사용한다.

Model-val의 episode당 고정 window 1개, 총 100개를 250 update마다 같은 순서로 평가한다. Validation 수치가 유한하고 latent std가 `max(.001, 초기 validation std×.1)` 이상인 상태에서 prediction MSE 최저 checkpoint를 선택한다. 동률은 먼저 나온 것을 유지한다. `initial_validation.json`, `validation_index.json`, `training.jsonl`과 best hash를 보관한다. 이 gate는 표현 붕괴 점검이며 충분한 표현·기억 성능을 보장하지 않는다.

검증 결과나 초기화가 동일한지 확인하려면 `config.json`·`initial_validation.json`·공통 normalizer 값을 대조한다. Test·Probe 점수로 학습 길이·best를 다시 고르지 않는다.

## 구간·미래 latent·Probe

```bash
python -m memorymaze_lewm segments --splits data/splits.json \
  --max-step 50 --out data/segments_candidates.json
# 동일 후보·초기 관측을 검수한 review_decisions.json을 적용
python -m memorymaze_lewm review --segments data/segments_candidates.json \
  --decisions data/review_decisions.json --out data/segments_reviewed.json
python -m memorymaze_lewm rollout --checkpoint runs/baseline500_001/checkpoints/best.pt \
  --segments data/segments_reviewed.json --device cuda --threads 3 \
  --max-step 50 --out runs/baseline500_001/latents
python -m memorymaze_lewm probe --latents runs/baseline500_001/latents \
  --out runs/baseline500_001/probe
```

후보·검수 결정은 두 데이터 규모에서 동일해야 한다. 저장한 승인 목록과 segment ID를 대조한다. 장비를 옮기면 segments를 재생성해 현재 원본 경로에 연결하고 같은 결정 파일을 적용한다. 초기 공 가시성 검수의 수행자·방식은 결과에 기록한다.

0-step은 초기 RGB 3장의 마지막 latent다. 1~50은 기록된 action을 사용한 재귀 예측이며 미래 RGB·좌표는 adapter에 전달하지 않는다. 모든 행에 episode·segment·step·원본 시점·물체 ID·관측/예측 구분을 저장한다.

Probe는 모든 0~50 latent를 함께 쓰는 단일 ridge linear regression이다. Train만으로 표준화하고 episode·step에 같은 비중을 준다. λ∈{0,.0001,.001,.01,.1,1} 중 Probe-val의 step별 episode 평균 L2를 다시 평균한 값이 최소인 것을 선택한다. 동률은 큰 λ. LeWM은 고정하며 Probe로 역전파하지 않는다. 선택 Probe를 저장·재로드한 뒤 test를 평가한다.

## 평가와 해석

각 step에서 공 상대 좌표 L2, train 전체 평균 좌표 기준, step별 train 평균 좌표 기준, 0-step 대비 증가량을 보고한다. 모델끼리는 같은 test 구간과 공통 성공 구간으로 비교하고 실패 수·사유를 남긴다. 같은 episode의 51개 행을 독립 51개 경험으로 세지 않는다. 현재 실험은 episode 분리이며 미로 배치 완전 독립은 보장하지 않는다.

최대 50 이후 확장은 이번 test를 보고 조절하지 않는다. Probe-val에서 평균 좌표 기준보다 유용한 성능이 유지되는지와 50−0-step 오차의 episode paired bootstrap 95% 구간을 함께 참고하되, 하락 미검출을 성능 유지 증명으로 해석하지 않는다. 나중에 최대 step 100 등으로 protocol을 먼저 고정하고 `segments`와 `rollout` 모두 같은 `--max-step`으로 다시 실행한다. 우선 기존 Probe를 고정해 51~100에 적용하고, Probe까지 재학습한 실험은 기존 0~50도 재평가해 별도로 보고한다. 현재 `probe` 명령은 재학습 경로이며 고정 Probe의 적용은 `memorymaze_lewm.probe.predict`로 구분한다.

## 확보 모델 사용과 모델 개선

완료 후 제공하는 inference checkpoint·대응 Probe·검수 결정·portable 데이터 명세를 별도 받는다. Git clone에는 대형 모델 파일이 포함되지 않는다. 저장 모델을 평가할 때는 새 `latents` 출력 폴더로 rollout하고, 그 모델의 latent로 Probe를 새로 학습하거나 이미 제공된 대응 Probe를 사용한다. 다른 모델의 Probe를 무조건 재사용하지 않는다.

- 같은 학습 규약: 별도 모듈에 `build_model(config)`를 구현하고 train의 `--model-factory`로 연결한다. `encode`·`predict`·`action_encoder` 입출력은 README의 규약을 유지한다.
- 다른 학습 규약: 학습 코드를 별도로 제공하고 rollout의 `--factory module:build_adapter`로 연결한다. Adapter는 `initial_latent(initial_images)->[D]`와 `rollout(initial_images,past_actions,future_actions)->[max_step,D]`를 제공한다.
- Memory를 추가하면 매 구간 시작마다 초기화한다. Memory 구조만 변경하고 데이터·전처리·예산·초기화·평가를 함께 바꾸지 않는다.
- 수정 위치: 모델 연결 `model.py`, 팀의 새 모델 모듈, 필요할 때만 `train.py`. 데이터 정렬·공통 split·정답·Probe·평가 코드는 비교 실험에서 고정한다.
- 입력 horizon 숫자를 Probe feature로 추가하지 않는다. 모델별 latent 차원이 달라도 같은 Probe 구조·선택 규칙을 적용한다.

개선 모델은 공유 parameter를 같은 초기 가중치에서 가져오고 새 parameter의 초기화·수를 기록한다. 새 구조가 공통 4-frame window 학습으로 장기 state를 학습할 수 있는지는 별도 확인해야 한다. 인터페이스 일치가 임의의 recurrent 학습법까지 보장하지는 않는다.

## 전달 목록

소스 commit·환경·config·초기 가중치 출처/hash, train/val/Probe/test episode 명세/hash, 공통 normalizer, 고정 validation index, 검수 결정/승인 목록, best/export/Probe hash, 학습 로그·곡선·매 step 지표·실패 목록·재로드 검증을 함께 전달한다. 로컬·서버 절대경로와 접속 정보는 공개 결과에서 제거한다.

초기 가중치는 [공식 LeWM Cube 모델](https://huggingface.co/quentinll/lewm-cube)의 고정 revision이며 모델 카드의 MIT 조건과 vendor 라이선스·출처 기록을 함께 유지한다. 로봇팔 실험을 수행한다는 뜻은 아니다.

GitHub 반영은 작업 브랜치 → PR을 사용한다. main 직접 push는 차단되며 소유자 계정만 PR을 반영할 수 있다. 팀원에게 병합 권한을 주거나 보호 규칙을 바꾸지 않는다.
