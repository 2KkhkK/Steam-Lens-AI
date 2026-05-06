import pandas as pd
from sentence_transformers import SentenceTransformer
import pickle
import time
import os

# 1. 데이터 불러오기
# 진웅님이 Copilot으로 깔끔하게 정리하신 그 파일을 사용합니다.
file_path = 'cleaned_games.csv'

if not os.path.exists(file_path):
    print(f"❌ '{file_path}' 파일이 없습니다. 파일명을 다시 확인해 주세요!")
else:
    print(f"📂 {file_path} 데이터를 불러오는 중...")
    df = pd.read_csv(file_path, encoding='utf-8-sig')

    # 2. AI 모델 로드 (MiniLM)
    # 문장의 의미를 파악하는 데 특화된 가볍고 강력한 BERT 모델입니다.
    print("🧠 AI 모델(BERT)을 로드하고 있습니다...")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # 3. 임베딩(특징 추출) 시작
    # 게임 설명(About_the_game)을 읽고 384개의 숫자로 이루어진 좌표로 변환합니다.
    print(f"🚀 총 {len(df)}개 게임의 특징 분석을 시작합니다.")
    print("💡 이 작업은 약 1시간 정도 소요됩니다. 잠시 휴식을 취하고 오세요!")

    start_time = time.time()

    # 데이터에 결측치가 있을 수 있으니 문자열(str)로 강제 변환 후 리스트로 만듭니다.
    descriptions = df['About_the_game'].astype(str).tolist()

    # batch_size는 한 번에 처리할 양입니다. 사양에 따라 32~64가 적당합니다.
    embeddings = model.encode(descriptions, batch_size=32, show_progress_bar=True)

    end_time = time.time()

    # 4. 결과 저장 (Pickle 방식)
    # 매번 AI를 돌릴 수 없으니, 계산 결과를 파일로 쪄서 보관하는 겁니다.
    print("💾 분석 결과를 'steam_embeddings.pkl'로 저장하는 중...")
    with open('steam_embeddings.pkl', 'wb') as f:
        pickle.dump({
            'names': df['Name'].tolist(), 
            'embeddings': embeddings
        }, f)

    print("-" * 30)
    print(f"✅ 임베딩 완료! 소요 시간: {(end_time - start_time) / 60:.2f}분")
    print(f"📦 생성된 파일: steam_embeddings.pkl")
    print("-" * 30)