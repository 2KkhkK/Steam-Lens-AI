import pandas as pd
import pickle
import re
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import time

print("데이터 로딩 중...")
df = pd.read_csv('cleaned_games.csv', encoding='utf-8-sig')
with open('steam_embeddings.pkl', 'rb') as f:
    data = pickle.load(f)
    embeddings = data['embeddings']

def get_tag_similarity(tags1, tags2):
    if pd.isna(tags1) or pd.isna(tags2): return 0.0
    list1 = re.findall(r"'([^']+)'\s*:", str(tags1))
    list2 = re.findall(r"'([^']+)'\s*:", str(tags2))
    set1 = set([t.lower().strip() for t in list1])
    set2 = set([t.lower().strip() for t in list2])
    if not set1 or not set2: return 0.0
    return len(set1.intersection(set2)) / len(set1.union(set2))

def run_bulk_eval(threshold=0.15, num_samples=100):
    np.random.seed(42)
    # 어느 정도 인기 있는 게임 중에서 샘플링하기 위해 설명이 길거나 리뷰가 많은 데이터를 뽑는 것이 좋지만, 
    # 여기서는 데이터 전반을 무작위 추출합니다.
    sample_indices = np.random.choice(df.index, size=num_samples, replace=False)
    
    total_baseline_precision = 0
    total_hybrid_precision = 0
    valid_samples = 0
    
    start_time = time.time()
    for count, idx in enumerate(sample_indices):
        target_vec = embeddings[idx].reshape(1, -1)
        target_tags = df.iloc[idx]['Tags']
        
        sim_scores = cosine_similarity(target_vec, embeddings).flatten()
        
        # Baseline
        baseline_indices = sim_scores.argsort()[-11:-1][::-1]
        baseline_hits = sum([1 for i in baseline_indices if get_tag_similarity(target_tags, df.iloc[i]['Tags']) >= threshold])
        
        # Hybrid
        top_1000_indices = sim_scores.argsort()[-1001:-1][::-1]
        candidates = []
        for i in top_1000_indices:
            if i == idx: continue
            bert_score = sim_scores[i]
            tag_sim = get_tag_similarity(target_tags, df.iloc[i]['Tags'])
            final_score = bert_score * (1.0 + tag_sim * 3.0)
            candidates.append((i, final_score, tag_sim))
            
        candidates.sort(key=lambda x: x[1], reverse=True)
        hybrid_indices = candidates[:10]
        hybrid_hits = sum([1 for item in hybrid_indices if item[2] >= threshold])
        
        total_baseline_precision += (baseline_hits / 10.0) * 100
        total_hybrid_precision += (hybrid_hits / 10.0) * 100
        valid_samples += 1
        
        if (count + 1) % 20 == 0:
            print(f"진행 상황: {count+1}/{num_samples} 완료...")

    print(f"\n--- 📊 대규모 검증 결과 (무작위 {valid_samples}개 게임) ---")
    print(f"기준: 태그 유사도 {threshold*100}% 이상 시 정답(Hit) 인정")
    print(f"1. BERT 단일 모델 평균 Precision@10: {total_baseline_precision / valid_samples:.1f}%")
    print(f"2. 하이브리드 엔진 평균 Precision@10: {total_hybrid_precision / valid_samples:.1f}%")
    print(f"소요 시간: {time.time() - start_time:.1f}초")

run_bulk_eval()
