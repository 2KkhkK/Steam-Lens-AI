import os
import requests
import json
import re
from django.conf import settings
from django.core.cache import cache

# mock_lowest_price.json 로드 로직
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MOCK_FILE_PATH = os.path.join(BASE_DIR, 'mock_lowest_price.json')
MOCK_DATA = {}

if os.path.exists(MOCK_FILE_PATH):
    with open(MOCK_FILE_PATH, 'r', encoding='utf-8') as f:
        try:
            MOCK_DATA = json.load(f)
        except json.JSONDecodeError:
            print("⚠️ mock_lowest_price.json 파싱 에러")

def get_user_owned_games(steam_id):
    """
    주어진 steam_id(SteamID64)를 기반으로 해당 유저가 보유한 게임 리스트(플레이 시간 포함)를 반환합니다.
    """
    STEAM_API_KEY = getattr(settings, 'STEAM_API_KEY', os.environ.get('STEAM_API_KEY', '85D83C9F15D86AE77598F640BECE4827'))
    owned_games_url = f"https://api.steampowered.com/IPlayerService/GetOwnedGames/v1/?key={STEAM_API_KEY}&steamid={steam_id}&include_appinfo=1"
    
    try:
        resp = requests.get(owned_games_url, timeout=3).json()
        games = resp.get('response', {}).get('games', [])
        return games
    except Exception as e:
        print(f"⚠️ Steam 보유 게임 목록 로드 실패 ({steam_id}): {e}")
        return []

def get_steam_price_info(image_url):
    """
    이미지 URL에서 App ID를 추출하여 Steam 실시간 가격 및 할인 정보를 반환합니다.
    방어 로직(무료 게임, 빈 리스트 등)이 포함되어 있습니다.
    """
    display_price = "Free to Play"
    original_price = ""
    discount_percent = 0
    is_discounted = False
    
    app_id_match = re.search(r'/apps/(\d+)/', str(image_url))
    if not app_id_match:
        return display_price, original_price, discount_percent, is_discounted
        
    app_id = app_id_match.group(1)
    
    cache_key = f"steam_price_{app_id}"
    cached_data = cache.get(cache_key)
    if cached_data:
        return cached_data

    api_url = f"https://store.steampowered.com/api/appdetails?appids={app_id}&cc=kr&filters=price_overview"
    
    try:
        resp = requests.get(api_url, timeout=2).json()
        
        if resp and isinstance(resp, dict) and resp.get(app_id, {}).get('success'):
            app_data = resp[app_id].get('data')
            if isinstance(app_data, dict):
                p_info = app_data.get('price_overview')
                if isinstance(p_info, dict):
                    display_price = p_info.get('final_formatted', 'Free to Play')
                    discount_percent = p_info.get('discount_percent', 0)
                    if discount_percent > 0:
                        is_discounted = True
                        original_price = p_info.get('initial_formatted', '')
                else:
                    display_price = "Free to Play"
            else:
                display_price = "Free to Play"
    except Exception as e:
        print(f"⚠️ Steam API Error ({app_id}): {e}")
        # 오류 시 기본값 유지

    cache.set(cache_key, (display_price, original_price, discount_percent, is_discounted), 3600) # 1 hour cache
    return display_price, original_price, discount_percent, is_discounted

def get_historical_low(game_name):
    """
    ITAD API를 호출하여 역대 최저가 정보를 반환합니다.
    실패 시 로컬의 mock_lowest_price.json 데이터를 사용하여 반환합니다.
    """
    ITAD_API_KEY = "3ed9b6dbd9f23acbab9868b306f36184f8bf2c71"
    
    cache_key = f"itad_price_{game_name}"
    cached_data = cache.get(cache_key)
    if cached_data:
        return cached_data
    
    try:
        search_url = "https://api.isthereanydeal.com/games/search/v1"
        search_params = {"key": ITAD_API_KEY, "title": game_name}
        search_resp = requests.get(search_url, params=search_params, timeout=2).json()
        
        if isinstance(search_resp, list) and len(search_resp) > 0:
            game_id = search_resp[0].get('id')
            info_url = "https://api.isthereanydeal.com/games/info/v2"
            info_params = {"key": ITAD_API_KEY, "id": game_id}
            info_resp = requests.get(info_url, params=info_params, timeout=2).json()
            
            target_data = {}
            if isinstance(info_resp, dict):
                if game_id in info_resp:
                    target_data = info_resp[game_id]
                else:
                    target_data = info_resp
            elif isinstance(info_resp, list) and len(info_resp) > 0:
                target_data = info_resp[0]
                
            if target_data and target_data.get('historyLow'):
                lowest_price = target_data['historyLow'].get('amount', 0)
                currency = target_data['historyLow'].get('currency', 'USD')
                if currency == "KRW":
                    res = f"₩ {int(lowest_price):,}"
                else:
                    res = f"{currency} {lowest_price}"
                cache.set(cache_key, res, 86400) # 24 hours cache
                return res
    except Exception as e:
        print(f"⚠️ ITAD API Error ({game_name}): {e}. Mock 데이터로 대체 시도합니다.")

    # Fallback to Mock Data
    for mock_name, data in MOCK_DATA.items():
        if mock_name.lower() in str(game_name).lower():
            currency = data.get('currency', 'USD')
            amount = data.get('amount', 0)
            if currency == "KRW":
                res = f"₩ {int(amount):,}"
            else:
                res = f"{currency} {amount}"
            cache.set(cache_key, res, 86400) # 24 hours cache
            return res
                
    cache.set(cache_key, "", 86400)
    return ""
