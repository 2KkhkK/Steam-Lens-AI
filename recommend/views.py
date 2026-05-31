from django.shortcuts import render, redirect
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import os
import re
import numpy as np
from collections import Counter
from allauth.socialaccount.models import SocialAccount
# pyrefly: ignore [missing-import]
from deep_translator import GoogleTranslator
from .utils import get_user_owned_games, get_steam_price_info, get_historical_low

# -----------------------------------------------------------------------------
# 1. 전역 데이터 및 AI 모델 로딩 (서버 구동 시 1회만 로드되어 메모리에 상주)
# -----------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(BASE_DIR, 'cleaned_games.csv')
EMBED_PATH = os.path.join(BASE_DIR, 'steam_embeddings.pkl')

df = None
embeddings = None

# CSV 파일과 임베딩(Vector) 파일이 존재하는 경우 데이터프레임으로 불러옵니다.
if os.path.exists(CSV_PATH) and os.path.exists(EMBED_PATH):
    df = pd.read_csv(CSV_PATH, encoding='utf-8-sig')
    with open(EMBED_PATH, 'rb') as f:
        data = pickle.load(f)
        embeddings = data['embeddings']
        
    # App_ID 컬럼이 없다면 Image_URL에서 정규식을 이용해 추출하여 추가합니다.
    if 'App_ID' not in df.columns:
        df['App_ID'] = df['Image_URL'].str.extract(r'/apps/(\d+)/')[0]

# -----------------------------------------------------------------------------
# 2. 유틸리티 함수 모음
# -----------------------------------------------------------------------------

def get_tag_similarity(tags1, tags2):
    """
    두 게임의 태그 리스트(문자열 형태)를 비교하여 자카드 유사도(Jaccard Similarity)를 계산합니다.
    - 교집합(겹치는 태그 수) / 합집합(전체 태그 수)
    """
    if pd.isna(tags1) or pd.isna(tags2):
        return 0.0
    
    # 정규표현식으로 '태그명' 부분만 추출
    list1 = re.findall(r"'([^']+)'\s*:", str(tags1))
    list2 = re.findall(r"'([^']+)'\s*:", str(tags2))
    
    set1 = set([t.lower().strip() for t in list1])
    set2 = set([t.lower().strip() for t in list2])
    
    if not set1 or not set2:
        return 0.0
        
    return len(set1.intersection(set2)) / len(set1.union(set2))

def safe_float(val, default=0.0):
    """
    문자열 형태의 가격 데이터("$19.99" 등)를 안전하게 float 타입으로 변환합니다.
    변환 실패 시 default 값을 반환합니다.
    """
    try:
        if isinstance(val, str):
            val = val.replace('$', '').replace(',', '').strip()
        return float(val)
    except (ValueError, TypeError):
        return default

def clean_game_text(text):
    """
    정규표현식을 사용하여 인코딩 오류 문자(??, ?) 및 불필요한 특수문자 등을 정리합니다.
    """
    if not isinstance(text, str):
        return "설명 없음"
        
    # 물음표가 2개 이상 연속되거나 맥락 없이 쓰인 인코딩 오류 패턴 제거
    text = re.sub(r'\?{2,}', '', text)
    # 불필요한 연속 공백 제거
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def translate_description(text, max_length=400):
    """
    영문 텍스트를 한국어로 번역합니다.
    너무 긴 텍스트는 max_length에서 자르고 번역을 수행합니다.
    타임아웃이나 에러 시 원문을 반환합니다.
    """
    if not text or text == "설명 없음":
        return text
        
    # 텍스트 길이 제한
    if len(text) > max_length:
        text = text[:max_length] + "..."
        
    print(f"🌍 [번역 진행 중] '{text[:30]}...' -> 한국어 변환 중...")
    try:
        translated = GoogleTranslator(source='auto', target='ko').translate(text)
        return translated
    except Exception as e:
        print(f"⚠️ 번역 실패: {e}")
        return text # 에러 시 원문 반환

# -----------------------------------------------------------------------------
# 3. View 함수 모음
# -----------------------------------------------------------------------------

def index(request):
    """
    메인 홈페이지 뷰. (단순 검색창 및 안내 문구 렌더링)
    """
    return render(request, 'recommend/index.html')

def search(request):
    """
    사용자가 입력한 검색어(게임명)를 기반으로 유사한 게임을 추천해 주는 뷰입니다.
    """
    query = request.GET.get('q', '').strip()
    results = []
    
    # 검색어가 없거나 데이터가 로드되지 않은 경우 메인으로 복귀
    if not query or df is None:
        return render(request, 'recommend/index.html')

    try:
        # 1. 로그인 유저의 보유 게임 리스트 확보 (추천 목록에서 제외하기 위함)
        owned_appids = set()
        if request.user.is_authenticated:
            try:
                steam_account = SocialAccount.objects.get(user=request.user, provider='steam')
                games = get_user_owned_games(steam_account.uid)
                owned_appids = set([str(g['appid']) for g in games])
            except SocialAccount.DoesNotExist:
                pass

        # 2. 검색어와 매칭되는 기준(타겟) 게임 찾기
        # [A] 정확한 일치 (대소문자 무시) 우선 검색
        target_rows = df[df['Name'].str.lower() == query.lower()]

        # [B] 정확한 결과가 없으면 부분 일치 검색 (Fuzzy Search)
        if target_rows.empty:
            matched = df[df['Name'].str.contains(query, case=False, na=False, regex=False)]
            if not matched.empty:
                # 여러 개 매칭될 경우 이름 길이가 가장 짧은 것을 원본 게임으로 간주
                idx_of_shortest = matched['Name'].str.len().argmin()
                target_rows = matched.iloc[[idx_of_shortest]]

        # 3. 타겟 게임 기반 추천 로직 실행
        if not target_rows.empty:
            idx = target_rows.index[0]
            target_vec = embeddings[idx].reshape(1, -1)
            target_tags = df.iloc[idx]['Tags']
            
            # [단계 1] BERT 임베딩 기반 코사인 유사도로 1차 후보군 추출 (상위 1000개)
            sim_scores = cosine_similarity(target_vec, embeddings).flatten()
            top_indices = sim_scores.argsort()[-1001:-1][::-1] # 1000개 추출 (자기 자신 제외)
            
            # [단계 2] 하이브리드 리랭킹 (태그 유사도 가중치 부여) 및 보유 게임 필터링
            candidates = []
            for i in top_indices:
                app_id = df.iloc[i].get('App_ID', '')
                
                # 사용자가 이미 보유한 게임은 추천 목록에서 배제
                if app_id in owned_appids:
                    continue
                    
                bert_score = sim_scores[i]
                tag_score = get_tag_similarity(target_tags, df.iloc[i]['Tags'])
                # 최종 점수 = BERT 점수 * (1 + 태그 유사도 * 가중치)
                final_score = bert_score * (1.0 + tag_score * 3.0) 
                candidates.append({'idx': i, 'final_score': final_score, 'bert_score': bert_score})
            
            candidates.sort(key=lambda x: x['final_score'], reverse=True)
            
            # [단계 3] 상위 6개 게임에 대해 실시간 가격/API 정보를 연동하여 최종 결과 생성
            for item in candidates[:6]:
                i = item['idx']
                image_url = df.iloc[i].get('Image_URL', '')
                game_name = df.iloc[i]['Name']
                raw_price = df.iloc[i].get('Price', 0)
                
                # utils.py 의 유틸리티 함수를 이용해 Steam & ITAD API 통신
                display_price, original_price, discount_percent, is_discounted = get_steam_price_info(image_url)
                
                # 가격 정보를 가져오지 못한 경우, CSV 상의 기존 가격을 표시 (오류 방지를 위해 safe_float 적용)
                if display_price == "Free to Play" and safe_float(raw_price) > 0:
                    display_price = f"${raw_price}"

                historical_low = get_historical_low(game_name)

                # 영문 설명 클리닝 및 번역 처리
                raw_about = df.iloc[i].get('About_the_game', '설명 없음')
                cleaned_about = clean_game_text(raw_about)
                translated_about = translate_description(cleaned_about, max_length=400)

                # 최종 결과 딕셔너리 구성
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
        else:
            return render(request, 'recommend/index.html', {'error': f"'{query}'와 유사한 게임을 찾을 수 없습니다."})
                
    except Exception as e:
        print(f"❌ Critical Error: {e}")
        return render(request, 'recommend/index.html', {'error': "서버 처리 중 오류가 발생했습니다."})

    return render(request, 'recommend/results.html', {'results': results, 'query': query})

def dashboard(request):
    """
    로그인한 스팀 유저의 플레이 기록을 바탕으로 개인화된 게임을 추천해 주는 대시보드 뷰입니다.
    """
    # 비로그인 유저는 메인으로 리다이렉트
    if not request.user.is_authenticated:
        return redirect('/')

    # 1. Django-Allauth를 통해 스팀 연동 계정(UID) 가져오기
    try:
        steam_account = SocialAccount.objects.get(user=request.user, provider='steam')
        steam_id = steam_account.uid
    except SocialAccount.DoesNotExist:
        return render(request, 'recommend/dashboard.html', {'error': '스팀 계정이 연동되지 않았습니다.'})

    # 2. 스팀 API를 호출하여 유저의 보유 게임 목록 및 플레이 타임 획득
    games = get_user_owned_games(steam_id)

    if not games:
        return render(request, 'recommend/dashboard.html', {'error': '보유한 게임이 없거나 프로필이 비공개입니다.'})

    # 플레이 시간이 긴 순서대로 내림차순 정렬
    games.sort(key=lambda x: x.get('playtime_forever', 0), reverse=True)
    
    # 가장 많이 플레이한 상위 10개 게임 추출
    top_games = games[:10]
    top_appids = [str(g['appid']) for g in top_games]
    owned_appids = set([str(g['appid']) for g in games])

    # App_ID 컬럼 안전성 체크 (없을 경우 생성)
    if 'App_ID' not in df.columns:
        df['App_ID'] = df['Image_URL'].str.extract(r'/apps/(\d+)/')[0]

    user_vectors = []
    tag_counter = Counter()
    
    # 3. 가장 많이 플레이한 상위 10개 게임의 임베딩 벡터 수집 및 유저 핵심 태그 분석
    for app_id in top_appids:
        matched = df[df['App_ID'] == app_id]
        if not matched.empty:
            idx = matched.index[0]
            user_vectors.append(embeddings[idx])
            
            # 태그 빈도수 계산
            tags_str = str(matched.iloc[0]['Tags'])
            tags_list = re.findall(r"'([^']+)'\s*:", tags_str)
            for t in tags_list:
                tag_counter[t.lower().strip()] += 1

    if not user_vectors:
        return render(request, 'recommend/dashboard.html', {'error': '보유한 게임 중 추천 시스템에 등록된 게임이 없습니다.'})
        
    # 유저의 핵심 태그 (상위 20개 추출)
    core_tags = set([tag for tag, count in tag_counter.most_common(20)])

    # 4. 유저 프로필 벡터 생성 (보유한 상위 10개 게임 벡터의 평균 좌표)
    user_profile_vector = np.mean(user_vectors, axis=0).reshape(1, -1)

    # 5. 생성된 유저 프로필 벡터와 전체 게임 임베딩 간의 코사인 유사도 계산
    sim_scores = cosine_similarity(user_profile_vector, embeddings).flatten()

    # 이미 보유한 게임들의 DataFrame 인덱스를 집합(Set)으로 수집 (O(1) 조회를 위함)
    owned_indices = set(df[df['App_ID'].isin(owned_appids)].index)

    # 상위 1000개 후보 추출
    top_indices = sim_scores.argsort()[-1000:][::-1]

    # 6. 후보군 하이브리드 필터링 (보유 게임 제외, 태그 및 메타스코어 가중치 반영)
    candidates = []
    for i in top_indices:
        if i in owned_indices:
            continue
            
        bert_score = sim_scores[i]
        
        # [A] 태그 유사도 점수 (유저의 핵심 태그와 후보 게임의 태그 비교)
        target_tags_str = str(df.iloc[i]['Tags'])
        target_tags_list = re.findall(r"'([^']+)'\s*:", target_tags_str)
        target_tags_set = set([t.lower().strip() for t in target_tags_list])
        
        if not core_tags or not target_tags_set:
            tag_score = 0.0
        else:
            tag_score = len(core_tags.intersection(target_tags_set)) / len(core_tags.union(target_tags_set))
            
        # [B] 품질(메타크리틱) 가중치 (0~100 -> 0.0~1.0)
        # 인디 게임 등 평가가 없는 경우(0) 페널티를 주어 퀄리티 높은 게임을 우선시함
        meta_score = safe_float(df.iloc[i].get('Metacritic_score', 0)) / 100.0
        
        # [C] 최종 점수: BERT 기본 점수에 태그 일치도와 메타스코어 보너스를 곱함
        final_score = bert_score * (1.0 + tag_score * 5.0 + meta_score * 2.0)
        
        candidates.append({'idx': i, 'final_score': final_score, 'bert_score': bert_score})
        
    candidates.sort(key=lambda x: x['final_score'], reverse=True)

    results = []
    # 7. 상위 6개 결과에 대해 실시간 가격 및 외부 API(ITAD 등) 정보 연동
    for item in candidates[:6]:
        i = item['idx']
        image_url = df.iloc[i].get('Image_URL', '')
        game_name = df.iloc[i]['Name']
        raw_price = df.iloc[i].get('Price', 0)
        
        display_price, original_price, discount_percent, is_discounted = get_steam_price_info(image_url)
        
        # 스팀 상점에 가격이 "Free to Play"로 표시되지만 실제 CSV에 유료 가격이 있는 경우의 예외 처리
        if display_price == "Free to Play" and safe_float(raw_price) > 0:
            display_price = f"${raw_price}"

        historical_low = get_historical_low(game_name)

        # 영문 설명 클리닝 및 번역 처리
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

    return render(request, 'recommend/dashboard.html', {'results': results})