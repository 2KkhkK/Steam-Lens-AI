"""외부 API 연동 계층.

Steam / IsThereAnyDeal 호출과 캐싱만 담당한다. 추천 로직은 services.py,
순수 계산은 similarity.py에 있다.

이전 버전 대비 바뀐 점
  - print() -> logging (배포 시 로그가 남고 레벨별 필터링이 가능해진다)
  - ITAD 키를 소스에서 제거하고 settings(=환경변수)에서 읽는다
  - 캐시 키를 결정적 해시로 생성한다
  - App ID 추출을 한 곳으로 모아 services.py와 중복을 없앤다
"""

import hashlib
import json
import logging
import os
import re

import requests
from django.conf import settings
from django.core.cache import cache

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MOCK_FILE_PATH = os.path.join(BASE_DIR, 'mock_lowest_price.json')

STEAM_TIMEOUT = 3
ITAD_TIMEOUT = 3

_APP_ID_RE = re.compile(r'/apps/(\d+)/')

MOCK_DATA = {}
if os.path.exists(MOCK_FILE_PATH):
    try:
        with open(MOCK_FILE_PATH, 'r', encoding='utf-8') as fp:
            MOCK_DATA = json.load(fp)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning('mock_lowest_price.json 로드 실패: %s', exc)


def cache_key(prefix, value):
    """결정적 캐시 키.

    예전에는 f"trans_{hash(text)}"를 썼는데, 파이썬은 보안상 문자열 해시를
    프로세스마다 무작위화한다(hash randomization). 그래서
      - 서버를 재시작하면 기존 캐시를 전부 찾지 못하고,
      - 워커가 여러 개면 워커마다 같은 텍스트를 따로 번역했다.
    7일 TTL을 설계한 의도가 통째로 무력화되던 버그다.
    """
    digest = hashlib.sha256(str(value).encode('utf-8')).hexdigest()[:32]
    return f'{prefix}_{digest}'


def extract_app_id(image_url):
    """스팀 헤더 이미지 URL에서 App ID를 뽑는다. 실패하면 빈 문자열."""
    match = _APP_ID_RE.search(str(image_url or ''))
    return match.group(1) if match else ''


def _format_price(amount, currency):
    try:
        amount = float(amount)
    except (TypeError, ValueError):
        return ''
    if currency == 'KRW':
        return f'₩ {int(amount):,}'
    return f'{currency} {amount}'


def get_user_owned_games(steam_id):
    """SteamID64로 보유 게임 목록(플레이 시간 포함)을 가져온다.

    키가 없거나 프로필이 비공개면 빈 리스트를 반환한다. 호출 측은 이를
    '개인화 불가' 신호로 쓰고, 검색 추천은 그대로 동작한다.
    """
    api_key = getattr(settings, 'STEAM_API_KEY', '')
    if not api_key:
        logger.warning('STEAM_API_KEY가 없어 보유 게임 조회를 건너뜁니다. .env를 확인하세요.')
        return []

    url = 'https://api.steampowered.com/IPlayerService/GetOwnedGames/v1/'
    params = {'key': api_key, 'steamid': steam_id, 'include_appinfo': 1}

    try:
        resp = requests.get(url, params=params, timeout=STEAM_TIMEOUT)
        resp.raise_for_status()
        return resp.json().get('response', {}).get('games', [])
    except Exception as exc:
        # 키가 URL에 담기므로 예외 메시지를 그대로 찍지 않고 예외 종류만 남긴다.
        # 넓게 잡는 이유: 이 함수가 실패해도 검색 추천은 그대로 동작해야 한다.
        logger.warning('Steam 보유 게임 조회 실패 (steamid=%s): %s', steam_id, type(exc).__name__)
        return []


def get_steam_price_info(image_url):
    """스팀 실시간 가격/할인 정보.

    반환: (표시가격, 정가, 할인율, 할인중여부)
    실패하거나 가격 정보가 없으면 'Free to Play'로 보수적으로 처리한다.
    """
    default = ('Free to Play', '', 0, False)

    app_id = extract_app_id(image_url)
    if not app_id:
        return default

    key = f'steam_price_{app_id}'
    cached = cache.get(key)
    if cached is not None:
        return tuple(cached)

    url = 'https://store.steampowered.com/api/appdetails'
    params = {'appids': app_id, 'cc': 'kr', 'filters': 'price_overview'}

    display_price, original_price, discount_percent, is_discounted = default

    try:
        resp = requests.get(url, params=params, timeout=STEAM_TIMEOUT)
        resp.raise_for_status()
        payload = resp.json()

        entry = payload.get(app_id) if isinstance(payload, dict) else None
        if isinstance(entry, dict) and entry.get('success'):
            data = entry.get('data')
            price = data.get('price_overview') if isinstance(data, dict) else None
            if isinstance(price, dict):
                display_price = price.get('final_formatted') or 'Free to Play'
                discount_percent = price.get('discount_percent', 0) or 0
                if discount_percent > 0:
                    is_discounted = True
                    original_price = price.get('initial_formatted', '')
    except Exception as exc:
        # 가격 조회 실패가 추천 카드 자체를 지우면 안 된다.
        # _enrich는 스레드에서 돌고, 거기서 예외가 나면 해당 추천이 통째로 누락된다.
        logger.warning('Steam 가격 조회 실패 (appid=%s): %s', app_id, exc)
        # 실패한 결과는 캐싱하지 않는다. 일시적 장애 뒤 곧바로 복구되도록.
        return default

    result = (display_price, original_price, discount_percent, is_discounted)
    cache.set(key, result, 3600)  # 할인은 수시로 바뀌므로 1시간
    return result


def _lookup_mock_low(game_name):
    """ITAD 실패 시 로컬 샘플 데이터로 대체."""
    name = str(game_name).lower()
    for mock_name, data in MOCK_DATA.items():
        if mock_name.lower() in name:
            return _format_price(data.get('amount', 0), data.get('currency', 'USD'))
    return ''


def get_historical_low(game_name):
    """ITAD API로 역대 최저가를 조회한다. 실패하면 Mock 데이터로 폴백."""
    key = cache_key('itad_price', game_name)
    cached = cache.get(key)
    if cached is not None:
        return cached

    api_key = getattr(settings, 'ITAD_API_KEY', '')

    if api_key:
        try:
            search_resp = requests.get(
                'https://api.isthereanydeal.com/games/search/v1',
                params={'key': api_key, 'title': game_name},
                timeout=ITAD_TIMEOUT,
            )
            search_resp.raise_for_status()
            found = search_resp.json()

            if isinstance(found, list) and found:
                game_id = found[0].get('id')
                info_resp = requests.get(
                    'https://api.isthereanydeal.com/games/info/v2',
                    params={'key': api_key, 'id': game_id},
                    timeout=ITAD_TIMEOUT,
                )
                info_resp.raise_for_status()
                info = info_resp.json()

                target = {}
                if isinstance(info, dict):
                    target = info.get(game_id, info)
                elif isinstance(info, list) and info:
                    target = info[0]

                low = (target or {}).get('historyLow') or {}
                if low:
                    result = _format_price(low.get('amount', 0), low.get('currency', 'USD'))
                    if result:
                        cache.set(key, result, 86400)  # 역대 최저가는 느리게 변함
                        return result
        except Exception as exc:
            # 어떤 예외가 나든 Mock 폴백으로 내려간다. 이것이 이 함수의 계약이다.
            logger.warning('ITAD 조회 실패 (%s): %s. Mock 데이터로 대체합니다.', game_name, exc)
    else:
        logger.debug('ITAD_API_KEY가 없어 Mock 데이터를 사용합니다.')

    result = _lookup_mock_low(game_name)
    # 찾지 못한 경우(빈 문자열)도 캐싱한다. 같은 게임에 대한 반복 실패 호출을
    # 막기 위한 negative caching.
    cache.set(key, result, 86400)
    return result
