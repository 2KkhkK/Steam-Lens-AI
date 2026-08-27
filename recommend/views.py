"""HTTP 계층.

이 파일은 요청 파싱 / 인증 확인 / 렌더링만 담당한다. 추천 계산은
services.py에 있고, 여기서는 그 결과를 템플릿에 넘겨줄 뿐이다.
"""

import logging
import time

from allauth.socialaccount.models import SocialAccount
from django.shortcuts import redirect, render
from django.views.decorators.http import require_GET

from .models import RecommendationClick, SearchLog
from .services import get_dashboard_recommendations, get_search_recommendations
from .utils import get_user_owned_games

logger = logging.getLogger(__name__)

STEAM_STORE_URL = 'https://store.steampowered.com/app/{app_id}/'


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

    보안: 리다이렉트 주소를 요청에서 받지 않고 app_id(숫자)로 서버가 직접
    조립한다. 임의 URL을 파라미터로 받으면 오픈 리다이렉트 취약점이 된다.
    """
    app_id = request.GET.get('app_id', '').strip()
    game_name = request.GET.get('name', '').strip()

    try:
        RecommendationClick.objects.create(
            user=request.user if request.user.is_authenticated else None,
            source=request.GET.get('source', SearchLog.SOURCE_SEARCH)[:16],
            query=request.GET.get('q', '')[:255],
            game_name=game_name[:255],
            app_id=app_id[:32],
            rank=min(int(request.GET.get('rank', 0) or 0), 32767),
        )
    except (ValueError, TypeError):
        pass
    except Exception as exc:
        logger.warning('클릭 로그 기록 실패: %s', exc)

    if app_id.isdigit():
        return redirect(STEAM_STORE_URL.format(app_id=app_id))
    return redirect('index')
