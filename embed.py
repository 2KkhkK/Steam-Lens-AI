import pandas as pd
from sentence_transformers import SentenceTransformer
import pickle
import time

print("📂 'Tags'가 포함된 완벽한 데이터를 불러오는 중...")
df = pd.read_csv('cleaned_games.csv', encoding='utf-8-sig')

# 1. 완벽한 피처 결합 (장르 + 태그 + 설명문)
def combine_features(row):
    genres = str(row['Genres']) if pd.notna(row['Genres']) else ""
    tags = str(row['Tags']) if pd.notna(row['Tags']) else ""
    about = str(row['About_the_game']) if pd.notna(row['About_the_game']) else ""
    
    # 이 한 줄이 엘든 링 추천의 퀄리티를 바꿉니다!
    return f"Genres: {genres}. Tags: {tags}. Description: {about}"

print("🧩 장르, 태그, 설명을 하나의 맥락으로 융합하는 중...")
combined_text = df.apply(combine_features, axis=1).tolist()

# 2. AI 모델 로드 및 학습
print("🧠 AI 모델(BERT) 로드 및 학습 시작...")
model = SentenceTransformer('all-MiniLM-L6-v2')

start_time = time.time()
# 결합된 텍스트로 벡터를 생성합니다. (사양에 따라 1시간 정도 소요)
embeddings = model.encode(combined_text, batch_size=32, show_progress_bar=True)
end_time = time.time()

# 3. 새로운 결과 덮어쓰기
print("💾 똑똑해진 임베딩 결과를 저장하는 중...")
with open('steam_embeddings.pkl', 'wb') as f:
    pickle.dump({
        'names': df['Name'].tolist(), 
        'embeddings': embeddings
    }, f)

print("-" * 30)
print(f"✅ 정공법 임베딩 완료! 소요 시간: {(end_time - start_time) / 60:.2f}분")
print("-" * 30)