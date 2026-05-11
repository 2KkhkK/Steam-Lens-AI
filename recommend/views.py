from django.shortcuts import render
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import os
import re

# (데이터 로드 부분은 기존과 동일하게 유지)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(BASE_DIR, 'cleaned_games.csv')
EMBED_PATH = os.path.join(BASE_DIR, 'steam_embeddings.pkl')

if os.path.exists(CSV_PATH) and os.path.exists(EMBED_PATH):
    df = pd.read_csv(CSV_PATH, encoding='utf-8-sig')
    with open(EMBED_PATH, 'rb') as f:
        data = pickle.load(f)
        game_names = data['names']
        embeddings = data['embeddings']

# 💡 새롭게 추가된 태그 유사도 계산 함수 (자카드 유사도)
def get_tag_similarity(tags1, tags2):
    if pd.isna(tags1) or pd.isna(tags2):
        return 0.0
    
    # 💡 정규표현식을 사용하여 {'태그명': 숫자} 형태에서 '태그명'만 리스트로 추출합니다.
    # 예: " 'Souls-like': 265 " -> "Souls-like" 추출
    list1 = re.findall(r"'([^']+)'\s*:", str(tags1))
    list2 = re.findall(r"'([^']+)'\s*:", str(tags2))
    
    # 모두 소문자로 바꾸고 집합(Set)으로 만들어 교집합을 계산할 준비를 합니다.
    set1 = set([t.lower().strip() for t in list1])
    set2 = set([t.lower().strip() for t in list2])
    
    if not set1 or not set2:
        return 0.0
        
    # 자카드 유사도: (교집합 개수) / (합집합 개수)
    return len(set1.intersection(set2)) / len(set1.union(set2))

# recommend/views.py 파일 안의 적당한 곳(search 함수 위쪽 추천)에 아래 코드를 추가하세요.

def index(request):
    """메인 검색 페이지를 렌더링하는 함수"""
    return render(request, 'recommend/index.html')

def search(request):
    query = request.GET.get('q', '').strip()
    results = []
    
    try:
        if query and df is not None:
            target_rows = df[df['Name'].str.lower() == query.lower()]
            
            if not target_rows.empty:
                idx = target_rows.index[0]
                target_vec = embeddings[idx].reshape(1, -1)
                target_tags = df.iloc[idx]['Tags']
                
                # 1단계: BERT 1차 추출
                sim_scores = cosine_similarity(target_vec, embeddings).flatten()
                top_30_indices = sim_scores.argsort()[-31:-1][::-1]
                
                candidates = []
                for i in top_30_indices:
                    bert_score = sim_scores[i]
                    candidate_tags = df.iloc[i]['Tags']
                    tag_score = get_tag_similarity(target_tags, candidate_tags)
                    
                    # 2단계: 하이브리드 점수 계산
                    final_score = bert_score * (1.0 + tag_score * 3.0) 
                    
                    candidates.append({
                        'idx': i,
                        'final_score': final_score
                    })
                
                # 3단계: 점수 순으로 줄 세우기 (Re-ranking)
                candidates.sort(key=lambda x: x['final_score'], reverse=True)
                
                # 💡 4단계: 여기서부터가 제가 빼먹었던 'HTML로 넘길 데이터 포장' 작업입니다!
                for item in candidates[:6]:
                    i = item['idx']
                    results.append({
                        'name': df.iloc[i]['Name'],
                        'about': df.iloc[i].get('About_the_game', '설명 없음')[:120],
                        'score': int(df.iloc[i].get('Metacritic_score', 0)),
                        'price': df.iloc[i].get('Price', 0),
                        'image': df.iloc[i].get('Image_URL', ''),
                        # 기존의 100% 단위가 아니라, 소수점 2자리 점수로 넘깁니다.
                        'similarity': round(item['final_score'], 2)
                    })
            else:
                return render(request, 'recommend/index.html', {'error': f"'{query}' 게임을 찾을 수 없습니다."})
                
    except Exception as e:
        print(f"❌ 검색 중 에러 발생: {e}")
        return render(request, 'recommend/index.html', {'error': f"서버 로직 에러: {e}"})

    # 포장된 results를 HTML에 성공적으로 배달합니다.
    return render(request, 'recommend/results.html', {'results': results, 'query': query})