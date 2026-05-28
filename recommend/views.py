from django.shortcuts import render
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import os
import re
import requests

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
            
            # 2단계: 하이브리드 리랭킹 (태그 가중치 적용)
            candidates = []
            for i in top_indices:
                bert_score = sim_scores[i]
                tag_score = get_tag_similarity(target_tags, df.iloc[i]['Tags'])
                final_score = bert_score * (1.0 + tag_score * 3.0) 
                candidates.append({'idx': i, 'final_score': final_score, 'bert_score': bert_score})
            
            candidates.sort(key=lambda x: x['final_score'], reverse=True)
            
            # 3단계: 최종 결과 포장 + 실시간 스팀 API 연동
            for item in candidates[:6]:
                i = item['idx']
                image_url = df.iloc[i].get('Image_URL', '')
                
                # 기본값 설정
                display_price = f"${df.iloc[i].get('Price', 0)}"
                original_price = ""
                discount_percent = 0
                is_discounted = False
                historical_low = ""

                # 💡 [1] Steam API 호출 로직 (무료 게임 빈 리스트 테러 완벽 방어)
                app_id_match = re.search(r'/apps/(\d+)/', str(image_url))
                if app_id_match:
                    app_id = app_id_match.group(1)
                    try:
                        api_url = f"https://store.steampowered.com/api/appdetails?appids={app_id}&cc=kr&filters=price_overview"
                        resp = requests.get(api_url, timeout=2).json()
                        
                        # 방어 1단계: 응답이 딕셔너리이고, success가 True인지 확인
                        if resp and isinstance(resp, dict) and resp.get(app_id, {}).get('success'):
                            
                            app_data = resp[app_id].get('data')
                            
                            # 🛡️ 방어 2단계: 스팀이 'data'를 리스트([])로 던지는 짓을 막아냄
                            if isinstance(app_data, dict):
                                p_info = app_data.get('price_overview')
                                
                                # 🛡️ 방어 3단계: price_overview 안쪽도 안전한지 최종 확인
                                if isinstance(p_info, dict):
                                    display_price = p_info.get('final_formatted', 'Free')
                                    discount_percent = p_info.get('discount_percent', 0)
                                    if discount_percent > 0:
                                        is_discounted = True
                                        original_price = p_info.get('initial_formatted', '')
                                else:
                                    display_price = "Free to Play" # 무료 게임 처리
                            else:
                                display_price = "Free to Play" # 스팀이 빈 리스트를 던졌을 때의 처리
                                
                    except Exception as e:
                        print(f"⚠️ Steam API Error ({app_id}): {e}")

              # 💡 [2] ITAD API 최신 버전(v1/v2) 디버깅 및 완벽 연동
                historical_low = ""
                # 👇 진웅님의 API Key를 유지해 주세요
                ITAD_API_KEY = "3ed9b6dbd9f23acbab9868b306f36184f8bf2c71"
                
                if ITAD_API_KEY != "3ed9b6dbd9f23acbab9868b306f36184f8bf2c71":
                    try:
                        game_name = df.iloc[i]['Name']
                        
                        # 1. URL 인코딩 문제 해결 (params 객체를 사용해 안전하게 변환)
                        search_url = "https://api.isthereanydeal.com/games/search/v1"
                        search_params = {"key": ITAD_API_KEY, "title": game_name}
                        
                        search_resp = requests.get(search_url, params=search_params, timeout=3).json()
                        
                        # 검색 결과가 배열(리스트) 형태로 잘 왔는지 확인
                        if isinstance(search_resp, list) and len(search_resp) > 0:
                            game_id = search_resp[0].get('id')
                            
                            # 2. 상세 정보 조회 (여기도 params 사용)
                            info_url = "https://api.isthereanydeal.com/games/info/v2"
                            info_params = {"key": ITAD_API_KEY, "id": game_id}
                            info_resp = requests.get(info_url, params=info_params, timeout=3).json()
                            
                            # 🔍 디버깅: ITAD가 정확히 뭐라고 보냈는지 터미널에 강제 출력!
                            print(f"\n[ITAD Info 응답] {game_name}: {info_resp}")
                            
                            # 3. 데이터 구조 껍질 벗기기 (딕셔너리, 리스트, 중첩 객체 모두 방어)
                            target_data = {}
                            if isinstance(info_resp, dict):
                                if game_id in info_resp:
                                    target_data = info_resp[game_id] # 게임 ID로 껍질이 감싸져 있는 경우 벗겨냄
                                else:
                                    target_data = info_resp # 바로 들어있는 경우
                            elif isinstance(info_resp, list) and len(info_resp) > 0:
                                target_data = info_resp[0] # 리스트로 들어오는 경우
                                
                            # 4. 역대 최저가(historyLow) 파싱
                            if target_data and target_data.get('historyLow'):
                                lowest_price = target_data['historyLow'].get('amount', 0)
                                currency = target_data['historyLow'].get('currency', 'USD')
                                
                                # ITAD가 원화를 주면 ₩로, 달러를 주면 USD로 예쁘게 출력
                                if currency == "KRW":
                                    historical_low = f"₩ {int(lowest_price):,}"
                                else:
                                    historical_low = f"{currency} {lowest_price}"
                            else:
                                print(f"❌ '{game_name}'의 historyLow 데이터가 없습니다.")
                        else:
                            print(f"🔍 ITAD 검색 실패 (결과 없음): {game_name}")
                            
                    except Exception as e:
                        print(f"⚠️ ITAD API Error ({game_name}): {e}")

                # 최종 결과 리스트에 담기 (results.append 부분은 기존과 동일하게 유지)
                results.append({
                    'name': df.iloc[i]['Name'],
                    'about': df.iloc[i].get('About_the_game', '설명 없음')[:120],
                    'score': int(df.iloc[i].get('Metacritic_score', 0)),
                    'image': image_url,
                    'similarity': round(item['final_score'], 2),
                    'price': display_price,
                    'original_price': original_price,
                    'discount_percent': discount_percent,
                    'is_discounted': is_discounted,
                    'historical_low': historical_low # 👈 여기서 빈칸 대신 진짜 가격이 넘어갑니다!
                })
        else:
            return render(request, 'recommend/index.html', {'error': f"'{query}'와 유사한 게임을 찾을 수 없습니다."})
                
    except Exception as e:
        print(f"❌ Critical Error: {e}")
        return render(request, 'recommend/index.html', {'error': "서버 처리 중 오류가 발생했습니다."})

    return render(request, 'recommend/results.html', {'results': results, 'query': query})