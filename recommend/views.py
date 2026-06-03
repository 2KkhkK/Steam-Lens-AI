from django.shortcuts import render, redirect
from allauth.socialaccount.models import SocialAccount
from .utils import get_user_owned_games
from .services import get_search_recommendations, get_dashboard_recommendations

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
    
    if not query:
        return render(request, 'recommend/index.html')

    try:
        owned_appids = set()
        if request.user.is_authenticated:
            try:
                steam_account = SocialAccount.objects.get(user=request.user, provider='steam')
                games = get_user_owned_games(steam_account.uid)
                owned_appids = set([str(g['appid']) for g in games])
            except SocialAccount.DoesNotExist:
                pass

        results, error = get_search_recommendations(query, owned_appids)
        if error:
            return render(request, 'recommend/index.html', {'error': error})

    except Exception as e:
        print(f"❌ Critical Error: {e}")
        return render(request, 'recommend/index.html', {'error': "서버 처리 중 오류가 발생했습니다."})

    return render(request, 'recommend/results.html', {'results': results, 'query': query})

def dashboard(request):
    """
    로그인한 스팀 유저의 플레이 기록을 바탕으로 개인화된 게임을 추천해 주는 대시보드 뷰입니다.
    """
    if not request.user.is_authenticated:
        return redirect('/')

    try:
        steam_account = SocialAccount.objects.get(user=request.user, provider='steam')
        steam_id = steam_account.uid
    except SocialAccount.DoesNotExist:
        return render(request, 'recommend/dashboard.html', {'error': '스팀 계정이 연동되지 않았습니다.'})

    games = get_user_owned_games(steam_id)

    if not games:
        return render(request, 'recommend/dashboard.html', {'error': '보유한 게임이 없거나 프로필이 비공개입니다.'})

    games.sort(key=lambda x: x.get('playtime_forever', 0), reverse=True)
    
    top_games = games[:10]
    top_appids = [str(g['appid']) for g in top_games]
    owned_appids = set([str(g['appid']) for g in games])

    try:
        results, error = get_dashboard_recommendations(top_appids, owned_appids)
        if error:
            return render(request, 'recommend/dashboard.html', {'error': error})
    except Exception as e:
        print(f"❌ Critical Error: {e}")
        return render(request, 'recommend/dashboard.html', {'error': "서버 처리 중 오류가 발생했습니다."})

    return render(request, 'recommend/dashboard.html', {'results': results})