from django.shortcuts import render, redirect
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import os
import re
import numpy as np
from allauth.socialaccount.models import SocialAccount
from .utils import get_user_owned_games, get_steam_price_info, get_historical_low

# 1. 데이터 및 모델 로드 (서버 실행 시 1회 로드)
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

# 💡 유틸리티: 태그 유사도 계산 (자카드 유사도)
def get_tag_similarity(tags1, tags2):
    if pd.isna(tags1) or pd.isna(tags2):
        return 0.0
    
    # 정규표현식으로 '태그명'만 추출
    list1 = re.findall(r"'([^']+)'\s*:", str(tags1))
    list2 = re.findall(r"'([^']+)'\s*:", str(tags2))
    
    set1 = set([t.lower().strip() for t in list1])
    set2 = set([t.lower().strip() for t in list2])
    
    if not set1 or not set2:
        return 0.0
        
    return len(set1.intersection(set2)) / len(set1.union(set2))

# 💡 View 1: 메인 페이지
def index(request):
    return render(request, 'recommend/index.html')

# 💡 View 2: 검색 및 추천 로직
def search(request):
    query = request.GET.get('q', '').strip()
    results = []
    
    if not query or df is None:
        return render(request, 'recommend/index.html')

    try:
        # 로그인 유저의 보유 게임 리스트 확보 (필터링 용도)
        owned_appids = set()
        if request.user.is_authenticated:
            try:
                steam_account = SocialAccount.objects.get(user=request.user, provider='steam')
                games = get_user_owned_games(steam_account.uid)
                owned_appids = set([str(g['appid']) for g in games])
            except SocialAccount.DoesNotExist:
                pass

        # [A] 타겟 게임 찾기 (정확한 일치 우선)
        target_rows = df[df['Name'].str.lower() == query.lower()]

        # [B] 정확한 결과가 없으면 부분 일치 검색 (Fuzzy Search)
        if target_rows.empty:
            matched = df[df['Name'].str.contains(query, case=False, na=False, regex=False)]
            if not matched.empty:
                # 이름이 가장 짧은 것을 우선 선택
                idx_of_shortest = matched['Name'].str.len().argmin()
                target_rows = matched.iloc[[idx_of_shortest]]

        # [C] 타겟 게임이 확정되었다면 추천 시작
        if not target_rows.empty:
            idx = target_rows.index[0]
            target_vec = embeddings[idx].reshape(1, -1)
            target_tags = df.iloc[idx]['Tags']
            
            # 1단계: AI(BERT) 1차 후보군 추출 (상위 1000개)
            sim_scores = cosine_similarity(target_vec, embeddings).flatten()
            top_indices = sim_scores.argsort()[-1001:-1][::-1]
            
            # 2단계: 하이브리드 리랭킹 및 보유 게임 필터링 적용
            candidates = []
            for i in top_indices:
                app_id = df.iloc[i].get('App_ID', '')
                # 이미 보유한 게임은 추천에서 제외
                if app_id in owned_appids:
                    continue
                    
                bert_score = sim_scores[i]
                tag_score = get_tag_similarity(target_tags, df.iloc[i]['Tags'])
                final_score = bert_score * (1.0 + tag_score * 3.0) 
                candidates.append({'idx': i, 'final_score': final_score, 'bert_score': bert_score})
            
            candidates.sort(key=lambda x: x['final_score'], reverse=True)
            
            # 3단계: 최종 결과 포장 + 실시간 스팀 API 연동
            for item in candidates[:6]:
                i = item['idx']
                image_url = df.iloc[i].get('Image_URL', '')
                game_name = df.iloc[i]['Name']
                
                # utils.py 의 유틸리티 함수를 이용해 API 통신 캡슐화
                display_price, original_price, discount_percent, is_discounted = get_steam_price_info(image_url)
                
                # 가격 정보를 가져오지 못했을 경우의 기본값 설정
                if display_price == "Free to Play" and df.iloc[i].get('Price', 0) > 0:
                    display_price = f"${df.iloc[i].get('Price', 0)}"

                historical_low = get_historical_low(game_name)

                # 최종 결과 리스트에 담기
                results.append({
                    'name': game_name,
                    'about': df.iloc[i].get('About_the_game', '설명 없음')[:120],
                    'score': int(df.iloc[i].get('Metacritic_score', 0)),
                    'image': image_url,
                    'similarity': round(item['final_score'], 2),
                    'price': display_price,
                    'original_price': original_price,
                    'discount_percent': discount_percent,
                    'is_discounted': is_discounted,
                    'historical_low': historical_low
                })
        else:
            return render(request, 'recommend/index.html', {'error': f"'{query}'와 유사한 게임을 찾을 수 없습니다."})
                
    except Exception as e:
        print(f"❌ Critical Error: {e}")
        return render(request, 'recommend/index.html', {'error': "서버 처리 중 오류가 발생했습니다."})

    return render(request, 'recommend/results.html', {'results': results, 'query': query})

# 💡 View 3: 로그인 유저 맞춤형 대시보드
def dashboard(request):
    if not request.user.is_authenticated:
        return redirect('/')

    # allauth를 통해 스팀 계정 정보 가져오기
    try:
        steam_account = SocialAccount.objects.get(user=request.user, provider='steam')
        steam_id = steam_account.uid
    except SocialAccount.DoesNotExist:
        return render(request, 'recommend/dashboard.html', {'error': '스팀 계정이 연동되지 않았습니다.'})

    games = get_user_owned_games(steam_id)

    if not games:
        return render(request, 'recommend/dashboard.html', {'error': '보유한 게임이 없거나 프로필이 비공개입니다.'})

    # 플레이 시간 기준으로 정렬
    games.sort(key=lambda x: x.get('playtime_forever', 0), reverse=True)
    
    # 상위 10개 추출
    top_games = games[:10]
    top_appids = [str(g['appid']) for g in top_games]
    owned_appids = set([str(g['appid']) for g in games])

    if 'App_ID' not in df.columns:
        df['App_ID'] = df['Image_URL'].str.extract(r'/apps/(\d+)/')[0]

    user_vectors = []
    
    for app_id in top_appids:
        matched = df[df['App_ID'] == app_id]
        if not matched.empty:
            idx = matched.index[0]
            user_vectors.append(embeddings[idx])

    if not user_vectors:
        return render(request, 'recommend/dashboard.html', {'error': '보유한 게임 중 추천 시스템에 등록된 게임이 없습니다.'})

    # 유저 프로필 벡터 생성 (보유한 상위 10개 게임 벡터의 평균)
    user_profile_vector = np.mean(user_vectors, axis=0).reshape(1, -1)

    # 코사인 유사도 계산
    sim_scores = cosine_similarity(user_profile_vector, embeddings).flatten()

    # 이미 보유한 게임은 추천에서 제외하기 위해 인덱스 수집
    owned_indices = set(df[df['App_ID'].isin(owned_appids)].index)

    top_indices = sim_scores.argsort()[-1000:][::-1]
    
    candidates = []
    for i in top_indices:
        if i in owned_indices:
            continue
        bert_score = sim_scores[i]
        candidates.append({'idx': i, 'final_score': bert_score})
        
    candidates.sort(key=lambda x: x['final_score'], reverse=True)

    results = []
    # 상위 6개 결과 추출
    for item in candidates[:6]:
        i = item['idx']
        image_url = df.iloc[i].get('Image_URL', '')
        game_name = df.iloc[i]['Name']
        
        display_price, original_price, discount_percent, is_discounted = get_steam_price_info(image_url)
        
        if display_price == "Free to Play" and df.iloc[i].get('Price', 0) > 0:
            display_price = f"${df.iloc[i].get('Price', 0)}"

        historical_low = get_historical_low(game_name)

        results.append({
            'name': game_name,
            'about': df.iloc[i].get('About_the_game', '설명 없음')[:120],
            'score': int(df.iloc[i].get('Metacritic_score', 0)),
            'image': image_url,
            'similarity': round(item['final_score'], 2),
            'price': display_price,
            'original_price': original_price,
            'discount_percent': discount_percent,
            'is_discounted': is_discounted,
            'historical_low': historical_low
        })

    return render(request, 'recommend/dashboard.html', {'results': results})