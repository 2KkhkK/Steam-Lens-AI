# Steam Lens AI

> **BERT 문맥 분석과 Jaccard 태그 검증을 융합한 하이브리드 게임 추천 및 실시간 혜택 큐레이션 웹 서비스**

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

1. **NLP 기반 게임 설명문 문맥 분석**
   - `Sentence-BERT (all-MiniLM-L6-v2)` 모델을 활용하여 게임의 시놉시스, 리뷰 등 텍스트 데이터를 분석하고, 단순 키워드 매칭을 넘어선 문맥 기반의 게임 간 유사도를 산출합니다.
2. **장르 순도 검증 및 어뷰징 방지를 위한 하이브리드 리랭킹**
   - 게임에 부여된 태그 데이터를 바탕으로 `Jaccard Similarity`를 계산합니다. 이를 NLP 분석 결과와 결합하여 추천의 정확도를 높이고, 일부 유저의 악의적인 태그 달기(어뷰징)로 인한 추천 왜곡을 방지합니다.
3. **Steam & ITAD API 연동 및 방어적 프로그래밍**
   - Steam Web API를 통해 유저의 실제 라이브러리 데이터를 연동하여 맞춤형 추천을 제공합니다.
   - IsThereAnyDeal (ITAD) API v3를 활용해 추천된 게임의 실시간 최저가 및 할인 정보를 큐레이션 합니다.
   - 외부 API 지연 및 장애에 대비한 방어적 프로그래밍(Timeout/Fallback 전략)을 적용하여 서비스의 안정성을 보장합니다.

---

## 📁 폴더 구조 (Directory Structure)

```text
Steam-Lens-AI/
├── 1_SourceCode/
│   ├── (Django 프로젝트 폴더)          # 웹 서비스 메인 애플리케이션 코드
│   ├── *.ipynb                     # 임베딩 생성 및 데이터 전처리용 Jupyter Notebook
│   └── evaluate_model.py           # 오프라인 추천 성능 검증 스크립트
├── 2_Data/
│   ├── cleaned_games.csv           # 전처리 및 정제 완료된 게임 데이터셋
│   └── steam_embeddings.pkl        # Sentence-BERT를 통해 추출된 게임 텍스트 임베딩 벡터 파일
└── requirements.txt                # 프로젝트 의존성 패키지 목록
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
# 1_SourceCode 내의 Django 프로젝트 폴더로 이동 후 실행
cd 1_SourceCode
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
