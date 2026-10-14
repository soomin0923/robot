# LeRobot 연습 프로젝트 가이드 (포트폴리오용, 로봇 없이)

## 0. 결론부터

**프로젝트 한 줄 정의:** "ACT 정책을 ALOHA 시뮬레이션 과제에 학습시키고, 변수 하나를 바꿔 성공률을 비교하고, 실패 에피소드를 유형별로 분석한다."

- 기간: 3~5일 (Colab GPU 시간이 대부분)
- 결과물: GitHub 폴더 1개 (README, 결과표, 실패 GIF, 재현 명령어)
- 핵심 평가 포인트: **튜토리얼을 돌렸다**가 아니라 **질문을 세우고 실험으로 답했다**

냉정하게 말하면, 공식 예제를 그대로 돌린 결과만 올리면 포트폴리오 가치는 거의 없습니다. 같은 노트북을 돌린 지원자가 많습니다. 차별점은 아래 세 가지뿐입니다.
1. 직접 정한 비교 실험 1개
2. 실패 원인 분석 (로보티즈 공고의 "실패 원인 분석" 업무와 직결)
3. 남이 그대로 재현할 수 있는 문서 (공고의 "예제 코드와 매뉴얼" 업무와 직결)

---

## 1. 확인된 사실 (2026-10 기준, LeRobot main 브랜치)

| 항목 | 내용 | 출처 |
|---|---|---|
| Python | `>=3.12` | pyproject.toml |
| 시뮬레이션 extras | `aloha`, `pusht`, `libero`(Linux 전용), `metaworld` | pyproject.toml |
| 등록된 env 타입 | `aloha`, `pusht`, `libero`, `metaworld` 외 | src/lerobot/envs/configs.py |
| aloha 기본값 | task `AlohaInsertion-v0`, 50fps, 에피소드 400스텝 | 같은 파일 |
| pusht 기본값 | task `PushT-v0`, 10fps, 에피소드 300스텝 | 같은 파일 |
| CLI | `lerobot-train`, `lerobot-eval`, `lerobot-dataset-viz`, `lerobot-info` | pyproject.toml |

**직접 확인해야 하는 것** (이번에 huggingface.co 접속이 막혀 검증하지 못함):
- 데이터셋 ID `lerobot/aloha_sim_transfer_cube_human`, `lerobot/aloha_sim_insertion_human`가 Hub에 있는지
- task 이름 `AlohaTransferCube-v0`가 유효한지
- Colab 기본 Python이 3.12 이상인지 (`!python --version`)

→ Day 1 첫 30분에 이 세 가지를 확인하고 README의 "환경" 절에 기록하세요.

---

## 2. 과제 선택

| 후보 | 장점 | 단점 | 추천 |
|---|---|---|---|
| **ALOHA Transfer Cube + ACT** | ACT 원 논문 과제라 비교 기준이 있음, 양팔 조작이라 로봇팔 직무와 연결 | 50fps, 이미지 입력이라 학습이 무거움 | **1순위** |
| PushT + ACT/Diffusion | 가볍고 빠름 | 2D 장난감 과제, 로봇팔 직무와 연결 약함 | 시간 부족 시 대안 |
| LIBERO / MetaWorld | 현재 README의 표준 벤치마크 | 설치·연산 부담 큼, ACT 예제가 적음 | 비추천 (이번 범위 초과) |

**권장:** ALOHA Transfer Cube로 시작. Colab 무료 GPU에서 1시간 내 학습이 안 끝나면 PushT로 바꾸고, 바꾼 이유를 README에 적으세요. 바꾼 이유도 의사결정 기록입니다.

---

## 3. 실험 설계: 변수 하나만 바꾼다

질문 하나를 고르세요. 둘 이상 하지 마세요.

| 질문 | 바꿀 값 | 왜 의미가 있나 |
|---|---|---|
| **A. 시연 데이터가 얼마나 필요한가?** (추천) | 학습 에피소드 수: 전체 vs 절반 vs 1/4 | 실물 로봇에서 데이터 수집 비용을 판단하는 근거. 면접에서 가장 설명하기 쉬움 |
| B. action chunk 크기의 영향 | `chunk_size` 예: 10 / 50 / 100 | ACT의 핵심 아이디어를 이해했는지 보여줌 |
| C. 학습 스텝 수와 성공률 | 체크포인트별 평가 | 가장 쉬움, 대신 차별점도 가장 약함 |

**고정할 것:** 시드, 평가 에피소드 수(최소 50), 평가 환경, 나머지 하이퍼파라미터.
**적을 것:** 각 설정의 성공률, 평균 보상, 학습 시간, GPU 종류.

평가 에피소드가 10개면 성공률 차이 10%p는 우연일 수 있습니다. 최소 50개로 하고, 시드 2개 이상이면 더 좋습니다.

---

## 4. 일정

### Day 1: 환경 + 베이스라인 시작
```bash
# Colab
!python --version              # 3.12 이상인지 확인
!pip install "lerobot[aloha]"  # 시뮬레이션 extra 포함
!lerobot-info                  # 설치 확인
```
- 데이터셋 확인: `lerobot-dataset-viz`로 에피소드 1~2개 보고, 관측(카메라, 관절)과 행동 차원을 README에 적기
- 베이스라인 학습을 **짧게** 먼저 실행 (예: 수천 스텝)해서 끝까지 도는지 확인

```bash
!lerobot-train \
  --policy.type=act \
  --dataset.repo_id=lerobot/aloha_sim_transfer_cube_human \
  --env.type=aloha \
  --env.task=AlohaTransferCube-v0 \
  --output_dir=outputs/act_baseline \
  --steps=5000 \
  --batch_size=8
```
플래그 이름(`--steps`, `--batch_size`, `--output_dir`, `--policy.push_to_hub` 등)은 버전에 따라 바뀝니다. `lerobot-train --help`로 확인하고, 실제로 쓴 명령어만 README에 남기세요.

### Day 2: 베이스라인 완주 + 평가
```bash
!lerobot-eval \
  --policy.path=outputs/act_baseline/checkpoints/last/pretrained_model \
  --env.type=aloha \
  --env.task=AlohaTransferCube-v0 \
  --eval.n_episodes=50 \
  --eval.batch_size=10
```
- 체크포인트는 Google Drive나 Hugging Face Hub에 저장 (Colab 세션 끊기면 사라짐)
- 학습 곡선(loss)과 평가 영상 저장

### Day 3: 비교 실험
- 3절에서 고른 변수로 1~2개 설정 추가 학습·평가
- 결과표 작성

### Day 4: 실패 분석 (가장 중요)
평가 영상에서 실패 에피소드를 보고 유형별로 분류합니다.

| 실패 유형 (예시) | 횟수 | 대표 영상 | 추정 원인 |
|---|---|---|---|
| 큐브를 못 잡음 (그리퍼 타이밍) | | | |
| 잡았으나 전달 중 떨어뜨림 | | | |
| 접근 경로가 엇나감 (위치 오차 누적) | | | |
| 멈춤/떨림 | | | |

"추정 원인"은 근거와 함께 쓰세요. 근거 없는 추측은 "가설"이라고 표시하세요.

### Day 5: 문서화
README 구조는 아래 5절 그대로.

---

## 5. 결과물 README 템플릿

```markdown
# ACT on ALOHA Transfer Cube: 시연 데이터 양에 따른 성공률

## 요약 (3줄)
- 질문: ...
- 결과: 전체 데이터 xx%, 절반 xx%, 1/4 xx% (평가 50 에피소드)
- 배운 점: ...

## 환경
LeRobot 버전/커밋, Python, GPU, 데이터셋 ID, 학습 시간

## 재현 방법
실제로 실행한 명령어 그대로

## 결과
표 + 학습 곡선 이미지 + 성공/실패 GIF 각 1개

## 실패 분석
유형별 표 + 대표 사례

## 막혔던 점과 해결
설치 오류, 메모리 부족, 세션 끊김 등 (실제로 겪은 것만)

## 한계
- 시뮬레이션만 함. 실물 로봇 수집·배포는 하지 않음
- 평가 에피소드 수, 시드 수의 한계
- 하지 않았지만 다음에 할 것
```

---

## 6. 하지 말 것

- **숫자 만들기 금지.** 돌리지 않은 결과, 반올림 조작, 잘 나온 시드만 고르기 모두 면접에서 드러납니다.
- **"실물 로봇 적용" 표현 금지.** 시뮬레이션이라고 명시하세요.
- **범위 확장 금지.** LIBERO, VLA(pi0, SmolVLA) 비교, 커스텀 환경 제작은 이번 범위 밖입니다. 하나를 끝내는 게 셋을 시작하는 것보다 낫습니다.
- **로컬 노트북(Quadro P520, VRAM 2GB)에서 학습 금지.** 평가 영상 확인, 문서 작성만 로컬에서.

---

## 7. 이 저장소의 기존 작업과 연결 (선택, 이번 범위 밖)

이 저장소에는 이미 Doosan 로봇 ROS2 패키지와 RealSense 기반 빨간 물체 검출 코드(`red.py`, `realsensered.py`)가 있습니다. 자소서에서는 다음처럼 **분리해서** 쓰는 것이 정직합니다.

- 기존 작업: 규칙 기반 비전(HSV 임계값) + ROS2로 실물 로봇 제어 경험
- 이번 작업: 학습 기반 정책(ACT)을 시뮬레이션에서 학습·평가·실패 분석

연결 문장 예시: "규칙 기반 검출은 조명이 바뀌면 임계값을 다시 잡아야 했고, 그래서 시연 데이터로 동작을 학습하는 방식을 시뮬레이션에서 직접 검증해 봤습니다." (규칙 기반에서 실제로 그런 문제를 겪었을 때만 쓰세요.)

---

## 8. 완료 체크리스트

- [ ] 환경(버전, GPU, 데이터셋 ID) 기록
- [ ] 베이스라인 학습·평가 완료 (평가 50 에피소드 이상)
- [ ] 비교 실험 1개 완료, 결과표 작성
- [ ] 실패 유형 분류표 + 대표 GIF
- [ ] 재현 명령어가 README와 실제 실행이 일치
- [ ] 한계 절에 "시뮬레이션만" 명시
- [ ] 자소서 부족한 점 단락을 실제 결과로 수정
