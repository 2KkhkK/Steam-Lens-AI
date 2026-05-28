from django.contrib import admin
from django.urls import path, include  # include를 반드시 추가하세요!

urlpatterns = [
    path('admin/', admin.site.urls),
    path('accounts/', include('allauth.urls')),
    path('', include('recommend.urls')), # 어떤 주소도 입력하지 않고 접속했을 때 recommend 앱의 설정을 따름
]