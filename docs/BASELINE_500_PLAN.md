# Memory Maze 500개 Baseline 실험 계획

## 목표와 고정 조건

LeWM 학습 500·검증 100, Probe 학습 후보 300·검증 후보 100·test 후보 200 episode를 확보해 Encoder·Transformer 공동 학습, 0~50-step readout 학습·평가를 실행한다. GPU 실행 성공과 장기 기억 성능 입증은 구분한다.

| 단계 | 실행·이유 | 완료 기준 |
|---|---|---|
| 데이터 | 공식 9×9 NPZ train 1,000개·eval 200개를 필요한 파일만 Range 다운로드. 원본 RGB·action·상대 좌표 보존 | CRC·크기·SHA-256·배열 규약 통과, 목적별 episode 중복 0 |
| 분할 | train 파일명 정렬의 첫 100개=model val, 다음 300개=Probe train, 다음 100개=Probe val. 그 뒤 500개=model train. eval은 기존 QA·이전 평가에서 확인한 40개를 건너뛴 신규 200개 | 2,000개 실험은 동일 holdout 500개 뒤 model train을 2,000개로 확장; 첫 500개가 부분집합 |
| 공동 학습 | 공개 Cube 초기 checkpoint의 호환 부분을 로드하고 6-action encoder를 seed 42로 초기화. 기존 Maze 학습본을 쓰지 않아 새로운 holdout 유입 방지. RGB 224·테두리 마스킹·history 3·frameskip 1, 4화면/3action window, MSE+0.09 SIGReg | batch 16·2,500 update·AdamW lr 5e-5·wd .001·seed 42. 짧은 저장/재로드·resume 점검 후 별도 본 run |
| 모델 선택 | model val 100 episode에서 고정 window 1개씩, 250 update마다 동일 자료 평가. 모든 값이 유한하고 latent std가 초기 검증 std의 10% 이상 및 1e-3 이상인 checkpoint 중 최소 prediction MSE. 동률은 먼저 나온 것 | best 하나 고정·hash 기록. test/Probe 성능으로 재선택하지 않음. 분산 조건은 붕괴 방지 점검이며 기억 성능 보장 아님 |
| 미래 latent | 초기 실제 RGB 3장만 입력. 기록 action으로 예측 latent를 재귀 입력. 0=초기 마지막 관측 latent, 1~50=예측 latent | 모든 step·episode·공 ID·원본 시점 보존. 미래 RGB·좌표는 모델 입력 금지. 초기 공 가시성 검수·미래 가시성 추정의 한계 별도 표시 |
| Probe | 고정 모델의 0~50 전체 latent→빨간 공 상대 좌표 2개, 단일 ridge Probe. Train만 feature 정규화. Episode·step 균형 가중치. λ∈{0,1e-4,1e-3,.01,.1,1} | Probe validation의 episode·step 평균 L2 최소, 동률은 큰 λ. 선택 후 고정·재로드. 각 모델별 같은 규칙으로 새 Probe |
| 평가 | test에서 매 step L2·평균 좌표 기준·0-step 대비 증가량. 실제 관측 위치 진단은 별도 보고 | episode별 동일 비중, 실패 수 포함. 0-step은 예측 정확도로 표현하지 않음 |
| 전달 | 소스 commit·환경·설정·분할/hash·검수 결정·가중치·Probe·곡선·실행 명령 | 재로드 일치·검사 통과, GitHub 코드·요약 공개. 데이터/가중치/계정 정보는 소스 commit 제외 |

## 500개와 2,000개 비교

모델 학습 episode 수만 바꾼다. 초기 가중치·seed·batch 16·2,500 update(총 40,000 window 추출)·학습률·loss·검증 window·Probe 후보·test 목록·전처리·평가 최대 step·Probe 선택 규칙을 고정한다. 같은 epoch 수는 데이터가 늘면 연산량도 늘어나므로 사용하지 않는다. Train 전이 action normalizer도 공통 500개에서 계산한 동일 통계를 사용해 단순 데이터 수 외 입력 규약 차이를 막는다. Train 목록만 확장하며 validation/test를 train으로 이동하지 않는다.

## 50 이후 확장

이번 실행은 최대 50으로 고정한다. Probe validation에서 episode별 50-step과 0-step L2 차이의 paired bootstrap 95% 구간으로 하락 검출 여부를 보고한다. 구간이 0을 포함하면 하락을 검출하지 못한 것이며 성능 유지의 증명은 아니다. 그 경우 추후 최대 100 등 별도 설정·구간 규칙을 먼저 고정하고 동일 pipeline을 재실행한다. Test를 보며 최대 step을 반복 조정하지 않는다.

## 구현·검증 순서

- [x] `data.py`·`__main__.py`: 500/2,000 안정 분할·공통 normalizer·설정 가능한 범위. 부분집합·중복 검사.
- [x] `model.py`·`rollout.py`·`segments.py`: 0-step과 전체 미래 step, causal·시점·길이 검사.
- [x] `train.py`: 고정 validation window·분산 gate·공통 통계·checkpoint provenance.
- [x] `evaluate.py`: 입력의 실제 step 범위 사용, episode 균형·검증 선택·0 기준·bootstrap.
- [ ] 공식 자료 확보·검사, 초기 공 가시성 검수, GPU smoke·본 학습·새 Probe·test·재로드 실행.
- [ ] 결과 요약·2,000개 실행 절차·모델 개선 접점 작성, 테스트 후 GitHub 업로드.

실행은 별도 worktree·별도 data/run에서 진행한다. 클러스터 미접속 시 기존 연구실 GPU 사용 여부와 실제 장비를 결과에 명시한다. 기존 데이터·보고서·실험 결과를 덮어쓰지 않는다.
