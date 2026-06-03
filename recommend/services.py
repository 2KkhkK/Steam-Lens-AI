import os
import pandas as pd
import pickle
import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
from django.core.cache import cache
# pyrefly: ignore [missing-import]
from deep_translator import GoogleTranslator

from .utils import get_steam_price_info, get_historical_low

# -----------------------------------------------------------------------------
# 1. 전역 데이터 및 AI 모델 로딩
# -----------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(BASE_DIR, 'cleaned_games.csv')
EMBED_PATH = os.path.join(BASE_DIR, 'steam_embeddings.pkl')

df = None
embeddings = None

if os.path.exists(CSV_PATH) and os.path.exists(EMBED_PATH):
    df = pd.read_csv(CSV_PATH, encoding='utf-8-sig')
    with open(EMBED_PATH, 'rb') as f:
        data = pickle.load(f)
        embeddings = data['embeddings']
        
    if 'App_ID' not in df.columns:
        df['App_ID'] = df['Image_URL'].str.extract(r'/apps/(\d+)/')[0]

# -----------------------------------------------------------------------------
# 2. 유틸리티 함수 모음
# -----------------------------------------------------------------------------
def get_tag_similarity(tags1, tags2):
    if pd.isna(tags1) or pd.isna(tags2):
        return 0.0
    list1 = re.findall(r"'([^']+)'\s*:", str(tags1))
    list2 = re.findall(r"'([^']+)'\s*:", str(tags2))
    set1 = set([t.lower().strip() for t in list1])
    set2 = set([t.lower().strip() for t in list2])
    if not set1 or not set2:
        return 0.0
    return len(set1.intersection(set2)) / len(set1.union(set2))

def safe_float(val, default=0.0):
    try:
        if isinstance(val, str):
            val = val.replace('$', '').replace(',', '').strip()
        return float(val)
    except (ValueError, TypeError):
        return default

def clean_game_text(text):
    if not isinstance(text, str):
        return "설명 없음"
    text = re.sub(r'\?{2,}', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def translate_description(text, max_length=400):
    if not text or text == "설명 없음":
        return text
    if len(text) > max_length:
        text = text[:max_length] + "..."
        
    cache_key = f"trans_{hash(text)}"
    cached_trans = cache.get(cache_key)
    if cached_trans:
        return cached_trans

    print(f"🌍 [번역 진행 중] '{text[:30]}...' -> 한국어 변환 중...")
    try:
        translated = GoogleTranslator(source='auto', target='ko').translate(text)
        cache.set(cache_key, translated, 86400 * 7) # 7일 캐시
        return translated
    except Exception as e:
        print(f"⚠️ 번역 실패: {e}")
        return text

# -----------------------------------------------------------------------------
# 3. 비즈니스 로직 래퍼 (추천)
# -----------------------------------------------------------------------------
def get_search_recommendations(query, owned_appids):
    if not query or df is None:
        return None, "데이터가 로드되지 않았거나 검색어가 없습니다."

    target_rows = df[df['Name'].str.lower() == query.lower()]

    if target_rows.empty:
        matched = df[df['Name'].str.contains(query, case=False, na=False, regex=False)]
        if not matched.empty:
            idx_of_shortest = matched['Name'].str.len().argmin()
            target_rows = matched.iloc[[idx_of_shortest]]

    if target_rows.empty:
        return None, f"'{query}'와 유사한 게임을 찾을 수 없습니다."

    idx = target_rows.index[0]
    target_vec = embeddings[idx].reshape(1, -1)
    target_tags = df.iloc[idx]['Tags']
    
    sim_scores = cosine_similarity(target_vec, embeddings).flatten()
    top_indices = sim_scores.argsort()[-1001:-1][::-1] 
    
    candidates = []
    for i in top_indices:
        app_id = df.iloc[i].get('App_ID', '')
        if app_id in owned_appids:
            continue
            
        bert_score = sim_scores[i]
        tag_score = get_tag_similarity(target_tags, df.iloc[i]['Tags'])
        final_score = bert_score * (1.0 + tag_score * 3.0) 
        candidates.append({'idx': i, 'final_score': final_score, 'bert_score': bert_score})
    
    candidates.sort(key=lambda x: x['final_score'], reverse=True)
    return format_recommendation_results(candidates[:6]), None

def get_dashboard_recommendations(top_appids, owned_appids):
    if df is None:
        return None, "데이터가 로드되지 않았습니다."
        
    user_vectors = []
    tag_counter = Counter()
    
    for app_id in top_appids:
        matched = df[df['App_ID'] == app_id]
        if not matched.empty:
            idx = matched.index[0]
            user_vectors.append(embeddings[idx])
            tags_str = str(matched.iloc[0]['Tags'])
            tags_list = re.findall(r"'([^']+)'\s*:", tags_str)
            for t in tags_list:
                tag_counter[t.lower().strip()] += 1

    if not user_vectors:
        return None, "보유한 게임 중 추천 시스템에 등록된 게임이 없습니다."
        
    core_tags = set([tag for tag, count in tag_counter.most_common(20)])
    user_profile_vector = np.mean(user_vectors, axis=0).reshape(1, -1)
    sim_scores = cosine_similarity(user_profile_vector, embeddings).flatten()
    owned_indices = set(df[df['App_ID'].isin(owned_appids)].index)
    top_indices = sim_scores.argsort()[-1000:][::-1]

    candidates = []
    for i in top_indices:
        if i in owned_indices:
            continue
            
        bert_score = sim_scores[i]
        target_tags_str = str(df.iloc[i]['Tags'])
        target_tags_list = re.findall(r"'([^']+)'\s*:", target_tags_str)
        target_tags_set = set([t.lower().strip() for t in target_tags_list])
        
        if not core_tags or not target_tags_set:
            tag_score = 0.0
        else:
            tag_score = len(core_tags.intersection(target_tags_set)) / len(core_tags.union(target_tags_set))
            
        meta_score = safe_float(df.iloc[i].get('Metacritic_score', 0)) / 100.0
        final_score = bert_score * (1.0 + tag_score * 5.0 + meta_score * 2.0)
        candidates.append({'idx': i, 'final_score': final_score, 'bert_score': bert_score})
        
    candidates.sort(key=lambda x: x['final_score'], reverse=True)
    return format_recommendation_results(candidates[:6]), None

def format_recommendation_results(candidates):
    results = []
    for item in candidates:
        i = item['idx']
        image_url = df.iloc[i].get('Image_URL', '')
        game_name = df.iloc[i]['Name']
        raw_price = df.iloc[i].get('Price', 0)
        
        display_price, original_price, discount_percent, is_discounted = get_steam_price_info(image_url)
        
        if display_price == "Free to Play" and safe_float(raw_price) > 0:
            display_price = f"${raw_price}"

        historical_low = get_historical_low(game_name)

        raw_about = df.iloc[i].get('About_the_game', '설명 없음')
        cleaned_about = clean_game_text(raw_about)
        translated_about = translate_description(cleaned_about, max_length=400)

        results.append({
            'name': game_name,
            'about': translated_about,
            'score': int(df.iloc[i].get('Metacritic_score', 0)),
            'image': image_url,
            'similarity': round(item['final_score'], 2),
            'price': display_price,
            'original_price': original_price,
            'discount_percent': discount_percent,
            'is_discounted': is_discounted,
            'historical_low': historical_low
        })
    return results
