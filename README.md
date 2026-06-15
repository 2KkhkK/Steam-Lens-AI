# Steam Lens AI

> **딥러닝 기반 하이브리드 게임 추천 및 실시간 혜택 큐레이션 웹 서비스**

기존의 게임 추천 시스템이 지닌 단순 키워드 매칭의 한계를 극복하고자, 자연어 처리(NLP) 모델과 유저 플레이 성향 분석을 결합하여 개인화된 큐레이션을 제공합니다.

## 🛠 기술 스택 (Tech Stack)

### Backend & AI
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Django](https://img.shields.io/badge/Django-092E20?style=for-the-badge&logo=django&logoColor=white)
![Sentence-BERT](https://img.shields.io/badge/Sentence--BERT-FF9900?style=for-the-badge&logo=huggingface&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)

### Frontend
![Bootstrap 5](https://img.shields.io/badge/Bootstrap_5-7952B3?style=for-the-badge&logo=bootstrap&logoColor=white)

### API
![Steam Web API](https://img.shields.io/badge/Steam_Web_API-000000?style=for-the-badge&logo=steam&logoColor=white)
![IsThereAnyDeal API v3](https://img.shields.io/badge/IsThereAnyDeal_API-4285F4?style=for-the-badge&logo=api&logoColor=white)

---

## ✨ 핵심 기능 요약 (Key Features)

1. **Sentence-BERT 기반 게임 설명문 문맥 분석**
   - `all-MiniLM-L6-v2` 모델을 활용하여 게임의 시놉시스를 고차원 벡터로 임베딩하고, 단순 키워드 매칭을 넘어선 문맥 기반의 코사인 유사도를 산출합니다.

2. **유저 플레이 기록 기반의 맞춤형 추천 (User Profiling)**
   - 스팀 계정 연동을 통해 유저가 실제 보유하고 주로 플레이한 상위 게임들의 임베딩 평균 벡터를 구하여 유저 플레이 성향이 반영된 동적 '유저 프로필 벡터' 기반 대시보드를 제공합니다.

3. **장르 순도 향상을 위한 하이브리드 리랭킹**
   - 1차 임베딩 유사도 점수에 장르/태그 기반의 `Jaccard Similarity` 점수를 가중합으로 결합하여 리랭킹함으로써 추천 결과의 장르 정확도(Precision@10)를 획기적으로 향상시켰습니다.

4. **Steam & ITAD API 연동 및 방어적 프로그래밍**
   - Steam API로 유저 라이브러리 및 최신 가격을, ITAD API로 역대 최저가 및 할인 정보를 큐레이션합니다.
   - 외부 API 지연 장애에 대비한 방어적 프로그래밍(Timeout 설정 및 Mock Data Fallback 전략)을 적용하여 서비스의 안정성을 극대화했습니다.

5. **실시간 한국어 번역 및 캐싱 (Lazy Translation)**
   - 원천 영문 데이터를 국내 유저가 쉽게 읽을 수 있도록 Google Translator API를 연동했습니다. 서버 부하 방지를 위해 수만 개의 데이터를 미리 번역하지 않고 상위 6개 결과에만 실시간 지연 번역을 요청한 뒤 캐싱(Caching)하여 응답 속도를 최적화했습니다.

---

## 📁 폴더 구조 (Directory Structure)

본 프로젝트는 다음과 같은 플랫(Flat)한 구조로 구성되어 있습니다.

```text
Steam-Lens-AI/
├── recommend/                  # 추천 및 비즈니스 로직(검색, 번역, 유사도 계산)이 포함된 메인 앱
├── steamlens/                  # Django 프로젝트 환경 설정 폴더
├── bulk_evaluate.py            # 오프라인 추천 성능 검증 스크립트 (Baseline vs Hybrid 평가)
├── data_check.py               # 데이터 결측치 정제 및 자료구조 변환 스크립트
├── embed.py                    # Sentence-BERT를 이용한 텍스트 임베딩 벡터 생성 스크립트
├── cleaned_games.csv           # 정규표현식으로 정제 완료된 게임 데이터셋
├── steam_embeddings.pkl        # 추출된 게임 텍스트 임베딩 벡터 파일
├── mock_lowest_price.json      # 외부 API 장애 시 우회하기 위한 Mock 데이터
├── manage.py                   # Django 웹 서버 실행 스크립트
└── requirements.txt            # 프로젝트 의존성 패키지 목록
```

---

## 🚀 빠른 시작 가이드 (Quick Start / How to Run)

본 프로젝트를 로컬 환경에서 실행하기 위한 단계별 가이드입니다. 아래 명령어를 터미널에 순서대로 복사하여 실행해 주세요.

### 1단계: 가상환경 생성 및 활성화
```bash
# 가상환경 생성 (venv)
python -m venv venv

# Windows 환경에서 활성화
venv\Scripts\activate

# Mac/Linux 환경에서 활성화
source venv/bin/activate
```

### 2단계: 패키지 설치
```bash
pip install -r requirements.txt
```

### 3단계: 로컬 서버 실행
```bash
# 최상단 프로젝트 폴더에서 곧바로 실행
python manage.py runserver
```

### 4단계: 접속 URL 안내
웹 브라우저를 열고 아래 주소로 접속합니다.
```text
http://127.0.0.1:8000
```

---

## 👨‍💻 작성자 (Author)

- **김진웅** (컴퓨터공학과)
