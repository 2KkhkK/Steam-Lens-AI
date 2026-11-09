"""HTTP 계층.

이 파일은 요청 파싱 / 인증 확인 / 렌더링만 담당한다. 추천 계산은
services.py에 있고, 여기서는 그 결과를 템플릿에 넘겨줄 뿐이다.
"""

import logging
import time

from allauth.socialaccount.models import SocialAccount
from django.conf import settings
from django.shortcuts import redirect, render
from django.views.decorators.http import require_GET

from .models import RecommendationClick, SearchLog
from .ratelimit import is_rate_limited
from . import services
from .services import get_dashboard_recommendations, get_search_recommendations
from .utils import get_user_owned_games

logger = logging.getLogger(__name__)

STEAM_STORE_URL = 'https://store.steampowered.com/app/{app_id}/'

RATE_LIMITED_MESSAGE = '요청이 너무 많습니다. 잠시 후 다시 시도해 주세요.'

_VALID_SOURCES = {choice for choice, _ in SearchLog.SOURCE_CHOICES}


def _steam_id_for(user):
    """로그인 유저의 SteamID64. 연동돼 있지 않으면 None."""
    if not user.is_authenticated:
        return None
    account = SocialAccount.objects.filter(user=user, provider='steam').first()
    return account.uid if account else None


def _owned_games_for(user):
    """보유 게임 목록. 미연동/비공개/키 없음이면 빈 리스트."""
    steam_id = _steam_id_for(user)
    if not steam_id:
        return []
    return get_user_owned_games(steam_id)


def _log_search(request, source, query, matched_name, results, started_at):
    """검색 로그를 남긴다.

    로그 적재 실패가 사용자 응답을 망가뜨려서는 안 되므로 통째로 감싼다.
    (마이그레이션 전이거나 DB가 잠긴 상황 등)
    """
    try:
        SearchLog.objects.create(
            user=request.user if request.user.is_authenticated else None,
            source=source,
            query=(query or '')[:255],
            matched_game=(matched_name or '')[:255],
            result_count=len(results or []),
            duration_ms=int((time.monotonic() - started_at) * 1000),
        )
    except Exception as exc:
        logger.warning('검색 로그 기록 실패: %s', exc)


def index(request):
    """메인 홈페이지. 검색창과 로그인 버튼만 렌더링한다."""
    return render(request, 'recommend/index.html')


def search(request):
    """검색어와 유사한 게임을 추천한다."""
    query = request.GET.get('q', '').strip()
    if not query:
        return render(request, 'recommend/index.html')

    max_len = getattr(settings, 'MAX_QUERY_LENGTH', 100)
    if len(query) > max_len:
        # 긴 검색어는 오타 교정(difflib) 비용만 키울 뿐 쓸모가 없다.
        return render(request, 'recommend/index.html', {
            'error': f'검색어는 {max_len}자 이내로 입력해 주세요.',
        }, status=400)

    # 임베딩 전수 계산 + 오타 교정까지 수행하는 비싼 경로라 IP당 횟수를 제한한다.
    # 제한에 걸린 요청은 로그도 남기지 않는다(로그 테이블 오염 방지).
    if is_rate_limited(request, 'SEARCH'):
        return render(request, 'recommend/index.html', {
            'error': RATE_LIMITED_MESSAGE,
            'query': query,
        }, status=429)

    started_at = time.monotonic()

    owned_appids = {str(g['appid']) for g in _owned_games_for(request.user)}

    try:
        results, error, meta = get_search_recommendations(query, owned_appids)
    except Exception as exc:
        logger.exception('검색 처리 중 오류 (q=%r): %s', query, exc)
        return render(request, 'recommend/index.html', {
            'error': '서버 처리 중 오류가 발생했습니다. 잠시 후 다시 시도해 주세요.',
            'query': query,
        })

    _log_search(request, SearchLog.SOURCE_SEARCH, query,
                meta.get('matched_name'), results, started_at)

    if error:
        # 오타 교정 후보를 함께 넘긴다. index.html에는 예전부터 이걸 표시하는
        # 코드가 있었지만 뷰가 값을 넘기지 않아 동작하지 않았다.
        return render(request, 'recommend/index.html', {
            'error': error,
            'query': query,
            'suggestions': meta.get('suggestions', []),
        })

    return render(request, 'recommend/results.html', {
        'results': results,
        'query': query,
        'matched_name': meta.get('matched_name', ''),
    })


def dashboard(request):
    """플레이 기록 기반 개인화 추천."""
    if not request.user.is_authenticated:
        return redirect('index')

    steam_id = _steam_id_for(request.user)
    if not steam_id:
        return render(request, 'recommend/dashboard.html', {
            'error': '스팀 계정이 연동되지 않았습니다.',
        })

    # 새로고침할 때마다 '우리' Steam API 키로 보유 게임을 조회한다. 스팀 계정은
    # 무료로 만들 수 있으므로 로그인만으로는 남용을 막지 못한다. 사용자당 제한.
    if is_rate_limited(request, 'DASHBOARD', ident=f'user:{request.user.pk}'):
        return render(request, 'recommend/dashboard.html', {
            'error': RATE_LIMITED_MESSAGE,
        }, status=429)

    started_at = time.monotonic()
    games = get_user_owned_games(steam_id)

    if not games:
        return render(request, 'recommend/dashboard.html', {
            'error': '보유한 게임이 없거나 스팀 프로필이 비공개로 설정되어 있습니다.',
        })

    try:
        results, error = get_dashboard_recommendations(games)
    except Exception as exc:
        logger.exception('대시보드 처리 중 오류 (steamid=%s): %s', steam_id, exc)
        return render(request, 'recommend/dashboard.html', {
            'error': '서버 처리 중 오류가 발생했습니다. 잠시 후 다시 시도해 주세요.',
        })

    _log_search(request, SearchLog.SOURCE_DASHBOARD, '', '', results, started_at)

    if error:
        return render(request, 'recommend/dashboard.html', {'error': error})

    return render(request, 'recommend/dashboard.html', {
        'results': results,
        'owned_count': len(games),
    })


@require_GET
def track_click(request):
    """추천 카드 클릭을 기록하고 스팀 상점으로 보낸다.

    이 기록이 있어야 CTR과 '순위별 클릭 분포'를 볼 수 있다. 오프라인
    지표(Precision@10)만으로는 실제 추천 품질을 확인할 수 없기 때문이다.

    보안
      - 리다이렉트 주소를 요청에서 받지 않고 app_id(숫자)로 서버가 직접
        조립한다. 임의 URL을 받으면 오픈 리다이렉트 취약점이 된다.
      - 비인증 GET이므로 누구나 호출할 수 있다. 예전에는 요청에 담긴 게임명과
        순위를 그대로 저장해서, 스크립트 한 줄로 CTR 지표를 마음대로 조작하고
        DB를 키울 수 있었다. 이제
          * 게임명은 요청이 아니라 카탈로그에서 가져오고(존재하지 않는 app_id는 기록 안 함)
          * source / rank는 허용 범위로 검증하며
          * IP당 횟수를 넘으면 기록을 생략한다.
        리다이렉트 자체는 제한하지 않는다. 사용자가 상점에 가는 것은 막을 이유가 없다.
    """
    app_id = request.GET.get('app_id', '').strip()

    if app_id.isdigit() and not is_rate_limited(request, 'CLICK'):
        _record_click(request, app_id)

    if app_id.isdigit():
        return redirect(STEAM_STORE_URL.format(app_id=app_id))
    return redirect('index')


def _record_click(request, app_id):
    # 모듈 경유로 호출한다. 이름으로 import하면 테스트에서 카탈로그를 바꿔 끼울 수 없다.
    catalog = services.get_catalog()
    if catalog is None:
        return
    pos = catalog.appid_to_index.get(app_id)
    if pos is None:
        # 추천 목록에 나올 수 없는 게임 -> 조작된 요청으로 보고 기록하지 않는다.
        return

    source = request.GET.get('source', SearchLog.SOURCE_SEARCH)
    if source not in _VALID_SOURCES:
        source = SearchLog.SOURCE_SEARCH

    try:
        rank = int(request.GET.get('rank', 0) or 0)
    except (TypeError, ValueError):
        rank = 0
    top_n = getattr(settings, 'RECOMMENDER', {}).get('TOP_N', 6)
    if not 0 <= rank <= top_n:
        rank = 0   # 실제 추천 목록 범위를 벗어난 순위는 '알 수 없음'으로 취급

    try:
        RecommendationClick.objects.create(
            user=request.user if request.user.is_authenticated else None,
            source=source,
            query=request.GET.get('q', '')[:255],
            game_name=catalog.names[pos][:255],
            app_id=app_id[:32],
            rank=rank,
        )
    except Exception as exc:
        logger.warning('클릭 로그 기록 실패: %s', exc)
