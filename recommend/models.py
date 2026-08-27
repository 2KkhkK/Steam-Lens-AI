"""추천 품질을 사후에 측정하기 위한 행동 로그 모델.

기존에는 모델이 하나도 없어서 "추천 시스템인데 피드백 루프가 없다"는 구조적
한계가 있었다. 무엇을 검색했는지, 어떤 추천을 실제로 클릭했는지 남지 않으면
오프라인 지표(Precision@10) 말고는 품질을 검증할 방법이 없다.

이 두 테이블이 쌓이면
  - CTR@6 같은 온라인 지표를 계산할 수 있고,
  - 장기적으로는 (유저 × 게임) 상호작용 행렬을 만들어
    협업 필터링으로 확장하는 발판이 된다.
"""

from django.conf import settings
from django.db import models


class SearchLog(models.Model):
    """검색창에 입력된 쿼리 한 건."""

    SOURCE_SEARCH = 'search'
    SOURCE_DASHBOARD = 'dashboard'
    SOURCE_CHOICES = [
        (SOURCE_SEARCH, '검색'),
        (SOURCE_DASHBOARD, '대시보드'),
    ]

    # 비로그인 사용자도 검색하므로 null 허용.
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name='search_logs',
    )
    source = models.CharField(max_length=16, choices=SOURCE_CHOICES, default=SOURCE_SEARCH)
    query = models.CharField(max_length=255, blank=True)
    # 검색어가 실제로 어떤 게임에 매칭됐는지. 오타 교정 효과를 보려면 필요하다.
    matched_game = models.CharField(max_length=255, blank=True)
    result_count = models.PositiveIntegerField(default=0)
    # 응답 지연 추적용. 외부 API 병렬화 전후를 비교할 수 있다.
    duration_ms = models.PositiveIntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True, db_index=True)

    class Meta:
        ordering = ['-created_at']
        indexes = [models.Index(fields=['query', 'created_at'])]

    def __str__(self):
        return f'[{self.source}] {self.query or "(빈 검색어)"} -> {self.result_count}건'


class RecommendationClick(models.Model):
    """추천 카드를 눌러 스팀 상점으로 넘어간 기록.

    rank(추천 목록에서 몇 번째였는지)를 함께 남기는 것이 핵심이다.
    이게 있어야 "상위에 배치한 추천이 실제로 더 많이 눌리는가"를 확인할 수 있고,
    nDCG 같은 순위 지표를 실제 사용자 행동으로 검증할 수 있다.
    """

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name='recommendation_clicks',
    )
    source = models.CharField(
        max_length=16,
        choices=SearchLog.SOURCE_CHOICES,
        default=SearchLog.SOURCE_SEARCH,
    )
    query = models.CharField(max_length=255, blank=True)
    game_name = models.CharField(max_length=255)
    app_id = models.CharField(max_length=32, blank=True)
    rank = models.PositiveSmallIntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True, db_index=True)

    class Meta:
        ordering = ['-created_at']
        indexes = [models.Index(fields=['game_name', 'created_at'])]

    def __str__(self):
        return f'{self.game_name} (rank {self.rank})'
