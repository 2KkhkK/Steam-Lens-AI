from django.shortcuts import render
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import os

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
    set1 = set([t.strip().lower() for t in str(tags1).split(',')])
    set2 = set([t.strip().lower() for t in str(tags2).split(',')])
    if not set1 or not set2:
        return 0.0
    # 교집합의 개수를 합집합의 개수로 나눔 (태그가 얼마나 겹치는가?)
    return len(set1.intersection(set2)) / len(set1.union(set2))

def search(request):
    query = request.GET.get('q', '').strip()
    results = []
    
    try:
        if query and df is not None:
            target_rows = df[df['Name'].str.lower() == query.lower()]
            
            if not target_rows.empty:
                idx = target_rows.index[0]
                target_vec = embeddings[idx].reshape(1, -1)
                target_tags = df.iloc[idx]['Tags'] # 타겟 게임의 태그
                
                # 1단계: AI(BERT)로 문맥이 비슷한 상위 30개(넉넉히) 1차 추출
                sim_scores = cosine_similarity(target_vec, embeddings).flatten()
                top_30_indices = sim_scores.argsort()[-31:-1][::-1]
                
                candidates = []
                for i in top_30_indices:
                    bert_score = sim_scores[i]
                    candidate_tags = df.iloc[i]['Tags']
                    
                    # 2단계: 태그 겹침 비율(Jaccard) 계산
                    tag_score = get_tag_similarity(target_tags, candidate_tags)
                    
                    # 3단계: 하이브리드 점수 계산 (AI 문맥 60% + 태그 일치율 40%)
                    # 가중치는 진웅님이 테스트하며 조절해보세요!
                    final_score = (bert_score * 0.6) + (tag_score * 0.4)
                    
                    candidates.append({
                        'idx': i,
                        'final_score': final_score,
                        'bert_score': bert_score,
                        'tag_score': tag_score
                    })
                
                # 최종 점수를 기준으로 다시 줄 세우기 (Re-ranking)
                candidates.sort(key=lambda x: x['final_score'], reverse=True)
                
                # 상위 6개만 화면에 전달
                for item in candidates[:6]:
                    i = item['idx']
                    results.append({
                        'name': df.iloc[i]['Name'],
                        'about': df.iloc[i].get('About_the_game', '설명 없음')[:120],
                        'score': int(df.iloc[i].get('Metacritic_score', 0)),
                        'price': df.iloc[i].get('Price', 0),
                        'image': df.iloc[i].get('Image_URL', ''),
                        # 화면에는 보기 좋게 BERT 점수 기준으로 퍼센트 표시
                        'similarity': round(item['bert_score'] * 100, 1) 
                    })
            else:
                return render(request, 'recommend/index.html', {'error': f"'{query}' 게임을 찾을 수 없습니다."})
                
    except Exception as e:
        print(f"❌ 검색 중 에러 발생: {e}")
        return render(request, 'recommend/index.html', {'error': f"서버 로직 에러: {e}"})

    return render(request, 'recommend/results.html', {'results': results, 'query': query})