from django.shortcuts import render
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity
# views.py 상단 수정
import os

# 현재 views.py가 있는 폴더의 부모의 부모(Steam-Lens-AI) 폴더를 가리킵니다.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CSV_PATH = os.path.join(BASE_DIR, 'cleaned_games.csv')
EMBED_PATH = os.path.join(BASE_DIR, 'steam_embeddings.pkl')

if os.path.exists(CSV_PATH) and os.path.exists(EMBED_PATH):
    df = pd.read_csv(CSV_PATH, encoding='utf-8-sig')
    with open(EMBED_PATH, 'rb') as f:
        data = pickle.load(f)
        game_names = data['names']
        embeddings = data['embeddings']
    print("✅ 추천 시스템 데이터 로드 완료!")
else:
    df = None
    print("❌ 데이터 파일을 찾을 수 없습니다. 경로를 확인해 주세요.")

def index(request):
    """메인 검색 페이지"""
    return render(request, 'recommend/index.html')

def search(request):
    query = request.GET.get('q', '').strip()
    results = []
    
    try:
        if query and df is not None:
            # 1. 대소문자 구분 없이 게임 찾기
            target_rows = df[df['Name'].str.lower() == query.lower()]
            
            if not target_rows.empty:
                idx = target_rows.index[0]
                target_vec = embeddings[idx].reshape(1, -1)
                
                # 2. 유사도 계산
                sim_scores = cosine_similarity(target_vec, embeddings).flatten()
                
                # 3. 상위 6개 추출 (본인 제외)
                related_indices = sim_scores.argsort()[-7:-1][::-1]
                
                for i in related_indices:
                    results.append({
                        'name': df.iloc[i]['Name'],
                        'about': df.iloc[i].get('About_the_game', '설명 없음')[:120],
                        'score': int(df.iloc[i].get('Metacritic_score', 0)),
                        'price': df.iloc[i].get('Price', 0),
                        'image': df.iloc[i].get('Image_URL', ''),
                        'similarity': round(sim_scores[i] * 100, 1)
                    })
            else:
                return render(request, 'recommend/index.html', {'error': f"'{query}' 게임을 찾을 수 없습니다."})
                
    except Exception as e:
        # 에러가 나면 터미널에 에러 내용을 출력합니다.
        print(f"❌ 검색 중 에러 발생: {e}")
        return render(request, 'recommend/index.html', {'error': f"서버 로직 에러: {e}"})

    return render(request, 'recommend/results.html', {'results': results, 'query': query})