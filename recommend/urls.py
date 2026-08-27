from django.urls import path

from . import views

urlpatterns = [
    path('', views.index, name='index'),                    # 메인 페이지
    path('search/', views.search, name='search'),           # 검색 결과
    path('dashboard/', views.dashboard, name='dashboard'),  # 맞춤형 대시보드
    path('click/', views.track_click, name='track_click'),  # 클릭 기록 후 스팀으로 이동
]
