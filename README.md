# Steam Lens AI

> **Sentence-BERT 임베딩과 태그 리랭킹을 결합한 하이브리드 게임 추천 웹 서비스**

게임 설명문을 문장 임베딩으로 벡터화해 두고, 검색어 또는 유저의 플레이 기록으로 만든
프로필 벡터와의 코사인 유사도를 계산한 뒤, 스팀 태그 유사도로 리랭킹하여 추천합니다.
단순 키워드 매칭으로는 잡히지 않는 "분위기가 비슷한 게임"을 찾는 것이 목표입니다.

## 기술 스택

| 영역 | 사용 기술 |
|---|---|
| Backend | Python 3.11+, Django 5.2 |
| 추천 엔진 | sentence-transformers (`all-MiniLM-L6-v2`), NumPy, pandas |
| 인증 | django-allauth (Steam OpenID 2.0) |
| Frontend | Bootstrap 5 (CDN) |
| 외부 API | Steam Web API, Steam Storefront API, IsThereAnyDeal API v3 |

---

## 핵심 기능

**1. 문장 임베딩 기반 유사도**
`Genres → Tags → Description` 순서로 결합한 텍스트를 384차원 벡터로 임베딩합니다.
순서는 의도적입니다 — `all-MiniLM-L6-v2`의 `max_seq_length`가 256토큰이라 긴 설명문은
뒤쪽이 잘리므로, 변별력이 큰 장르·태그를 앞에 배치해 잘림에 대비합니다.

**2. 2단계 랭킹 (Retrieval → Ranking)**
전체 카탈로그에 태그 유사도를 계산하면 느립니다. 벡터 내적으로 상위 1,000개까지
후보를 좁힌 뒤, 그 안에서만 태그 기반으로 정밀 정렬합니다.

```
final_score = bert_score × (1 + tag_score × w_tag + meta_score × w_meta)
```

**3. 플레이타임 가중 유저 프로필**
보유 게임 중 플레이 시간 상위 N개의 임베딩을 `log(1 + 플레이시간)`으로 가중 평균합니다.
단순 평균을 쓰면 2000시간 플레이한 게임과 1시간짜리가 동등하게 기여합니다.

**4. 방어적 외부 API 연동**
모든 호출에 타임아웃, 데이터 성질별 차등 TTL 캐싱(가격 1시간 / 최저가 24시간 / 번역 7일),
ITAD 장애 시 로컬 Mock 폴백. 실패 결과도 캐싱해 반복 실패 호출을 막습니다(negative caching).

**5. 지연 번역 (Lazy Translation)**
전체를 미리 번역하지 않고, 화면에 실제로 보이는 상위 N개만 요청 시점에 번역하고 캐싱합니다.

---

## 빠른 시작

### 1. 가상환경 및 패키지 설치

```bash
python -m venv venv
```

```bash
venv\Scripts\activate
```

<sub>macOS / Linux: `source venv/bin/activate`</sub>

```bash
pip install -r requirements.txt
```

### 2. 환경변수 설정

```bash
copy .env.example .env
```

<sub>macOS / Linux: `cp .env.example .env`</sub>

`.env`를 열어 값을 채웁니다. **전부 비워 둬도 검색 추천은 동작합니다.**

| 변수 | 없으면 | 발급처 |
|---|---|---|
| `DJANGO_SECRET_KEY` | 실행마다 임시 키 생성 (재시작 시 세션 초기화) | `python -c "import secrets; print(secrets.token_urlsafe(50))"` |
| `STEAM_API_KEY` | 스팀 로그인/대시보드 기능만 비활성 | [steamcommunity.com/dev/apikey](https://steamcommunity.com/dev/apikey) |
| `ITAD_API_KEY` | `mock_lowest_price.json`으로 자동 대체 | [isthereanydeal.com/apps/my](https://isthereanydeal.com/apps/my/) |

### 3. 데이터 준비

추천에는 게임 데이터셋이 필요합니다. 용량 문제로 저장소에 포함하지 않았습니다.

원본은 Kaggle의 **Steam Games Dataset** (약 97,000개 게임, `name` / `about_the_game` /
`genres` / `tags` / `metacritic_score` / `header_image` 컬럼 포함)을 사용했습니다.
받은 파일을 `games.csv`로 프로젝트 루트에 두고:

```bash
python data_check.py --sample 300
```

컬럼을 **이름으로** 찾으므로 데이터셋 버전이 조금 달라도 동작합니다.
컬럼명이 다르면 `data_check.py`의 `COLUMN_MAP`에 후보를 추가하세요.
원본 컬럼 구성을 먼저 보려면:

```bash
python data_check.py --inspect
```

그다음 임베딩을 생성합니다 (CPU 기준 전체 데이터 약 1시간):

```bash
python embed.py
```

<sub>샘플만으로 빠르게 확인하려면 `python embed.py --input sample_games.csv`</sub>

### 4. DB 마이그레이션 및 실행

```bash
python manage.py migrate
```

```bash
python manage.py runserver
```

→ http://127.0.0.1:8000

---

## 프로젝트 구조

```text
Steam-Lens-AI/
├── recommend/
│   ├── similarity.py       순수 계산 로직 (Django 무관 · 서비스/평가/테스트가 공유)
│   ├── services.py         추천 비즈니스 로직 (카탈로그 로딩, 랭킹, 프로필)
│   ├── utils.py            외부 API 연동 및 캐싱
│   ├── views.py            HTTP 계층 (얇게 유지)
│   ├── models.py           SearchLog / RecommendationClick (행동 로그)
│   └── tests.py            유닛 테스트 68개
├── steamlens/              Django 설정
├── data_check.py           원본 CSV 정제 (+ --inspect, --sample)
├── embed.py                Sentence-BERT 임베딩 생성
├── bulk_evaluate.py        오프라인 성능 평가 (3가지 프로토콜)
├── .env.example            환경변수 템플릿
└── requirements.txt
```

`similarity.py`를 따로 둔 이유: 예전에는 `services.py`와 `bulk_evaluate.py`가 태그 파싱
정규식을 각자 복사해 갖고 있었습니다. 한쪽만 고치면 **평가 결과와 실제 서비스 동작이
어긋나도 아무 에러가 나지 않습니다.**

---

## 테스트

```bash
python manage.py test recommend
```

68개 테스트가 태그 파싱, 유사도 계산, 랭킹 로직, 프로필 가중치, 외부 API 폴백,
뷰 동작을 검증합니다. 외부 네트워크 호출은 전부 mock으로 차단되어 오프라인에서도 실행됩니다.

---

## 성능 평가 — 방법론에 대한 정직한 설명

### 이전 버전 평가의 치명적 결함

초기 `bulk_evaluate.py`는 이런 구조였습니다.

```
정답(Hit) 기준  :  태그 자카드 유사도 >= 0.15
하이브리드 정렬  :  bert_score × (1 + 태그 자카드 × 3.0)
```

**태그가 겹치도록 정렬해 놓고, 태그가 겹치는지로 채점**하고 있었습니다.
하이브리드가 베이스라인을 이기는 것은 모델이 좋아서가 아니라 수학적으로 당연한 결과입니다.
채점 기준을 최적화 목표로 그대로 쓴 **순환논법(circular reasoning)**이며,
이 실험은 추천 품질이 향상됐다는 증거가 되지 못합니다.

### 현재: 정렬 신호와 채점 신호를 분리한 3가지 프로토콜

| 프로토콜 | 방식 | 신뢰도 | 자동 실행 |
|---|---|:---:|:---:|
| `holdout` | 태그 어휘를 겹치지 않는 두 묶음으로 나눠, 정렬엔 A만 · 채점엔 B만 사용 | ★★☆ | ✅ |
| `genre` | 정렬은 Tags로, 채점은 별도 컬럼인 Genres로 | ★☆☆ | ✅ |
| `golden` | 사람이 직접 라벨링한 정답셋 | ★★★ | 라벨링 필요 |

```bash
python bulk_evaluate.py --protocol all
```

측정 지표는 Precision@10, **nDCG@10**(순위 반영), **다양성 ILD**(필터버블 확인),
**커버리지**, **추천된 게임의 평균 메타크리틱**(인기 편향 확인)입니다.
정확도만 올리면 비슷한 게임만 나열되므로 다양성과 함께 봐야 합니다.

### 남아 있는 한계

임베딩 입력 문자열이 `"Genres: ... Tags: ... Description: ..."`이라 장르·태그 정보가
이미 벡터에 녹아 있습니다. 따라서 `holdout`과 `genre`도 **완전한 독립 검증은 아닙니다.**
다만 이 누수는 베이스라인과 하이브리드에 똑같이 적용되므로 **두 방법의 비교는 공정합니다.**
완전한 독립 검증은 `golden` 프로토콜뿐입니다.

### 골든셋 만들기

```bash
python bulk_evaluate.py --make-template 40
```

`eval_goldenset.template.csv`가 생성됩니다. `relevant` 열에 0/1을 채워
`eval_goldenset.csv`로 저장한 뒤:

```bash
python bulk_evaluate.py --protocol golden
```

베이스라인과 하이브리드의 추천 결과를 합집합으로 섞어 두므로 라벨링 편향이 없습니다.
전부 채울 필요 없이 30~50쌍만으로도 독립적인 검증이 됩니다.

### 가중치 튜닝

`w_tag = 3.0` 같은 값은 원래 근거 없이 정해진 상수였습니다. 이제 그리드 서치로 찾습니다.

```bash
python bulk_evaluate.py --grid
```

결과를 `.env`의 `SEARCH_TAG_WEIGHT` 등에 반영하면 코드 수정 없이 적용됩니다.

---

## 알려진 한계와 다음 계획

| 한계 | 현재 상태 | 계획 |
|---|---|---|
| 한국어 검색 | 검색어를 영어로 번역해 매칭하는 임시 처방 | `multilingual-e5` 등으로 교체 → 한국어 자연어 문장 검색 |
| 학습 없음 | 사전학습 모델 추론만 사용 | 스팀 "함께 구매" 쌍으로 대조학습(MultipleNegativesRankingLoss) 파인튜닝 |
| 전수 유사도 계산 | 매 요청 `embeddings @ query` (정규화 후 내적, argpartition) | 규모 확대 시 FAISS / hnswlib ANN 인덱스 |
| 온라인 지표 부재 | `SearchLog` / `RecommendationClick` 적재 시작 | CTR@6 측정 → 협업 필터링과 결합한 하이브리드 |
| 다양성 미최적화 | 지표는 측정하되 최적화는 안 함 | MMR 재정렬로 관련성-다양성 트레이드오프 |
| 번역 라이브러리 | `deep-translator` (비공식, PoC용) | 상용화 시 공식 Cloud Translation API |
| 임베딩 정적 | 신작 추가 시 `embed.py` 수동 재실행 | 신규 게임만 인코딩해 append하는 증분 배치 |

---

## 배포 시 체크리스트

```bash
python manage.py check --deploy
```

`.env`에서 `DJANGO_DEBUG=False`, `DJANGO_SECRET_KEY` 지정, `DJANGO_ALLOWED_HOSTS`를
실제 도메인으로, HTTPS 환경이면 `DJANGO_SECURE_SSL=True`(HSTS·보안 쿠키·SSL 리다이렉트 활성화).
워커를 여러 개 띄운다면 `REDIS_URL`을 반드시 지정하세요 —
기본 `LocMemCache`는 프로세스 로컬이라 워커 간 캐시가 공유되지 않습니다.

---

## 작성자

**김진웅** (컴퓨터공학과)
