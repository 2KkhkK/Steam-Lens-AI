from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),               # 메인 페이지 (http://127.0.0.1:8000/)
    path('search/', views.search, name='search'),     # 결과 페이지 (http://127.0.0.1:8000/search/)
    path('dashboard/', views.dashboard, name='dashboard'), # 맞춤형 대시보드 페이지
]