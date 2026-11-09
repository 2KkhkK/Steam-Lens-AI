from django.conf import settings
from django.contrib import admin
from django.urls import include, path

from recommend.ratelimit import throttled_admin_login

urlpatterns = [
    # 관리자 로그인은 admin.site.urls보다 '먼저' 매칭되도록 앞에 둔다.
    # 같은 주소(<ADMIN_URL>login/)를 실패 제한이 걸린 뷰가 가로챈다.
    path(f'{settings.ADMIN_URL}login/', throttled_admin_login),
    # 경로는 DJANGO_ADMIN_URL로 바꾼다. 기본 'admin/'은 스캐너가 가장 먼저 두드리는 주소다.
    path(settings.ADMIN_URL, admin.site.urls),
    path('accounts/', include('allauth.urls')),
    path('', include('recommend.urls')),
]
