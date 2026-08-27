from django.contrib import admin

from .models import RecommendationClick, SearchLog


@admin.register(SearchLog)
class SearchLogAdmin(admin.ModelAdmin):
    list_display = ('created_at', 'source', 'query', 'matched_game', 'result_count', 'duration_ms', 'user')
    list_filter = ('source', 'created_at')
    search_fields = ('query', 'matched_game')
    date_hierarchy = 'created_at'
    readonly_fields = ('created_at',)


@admin.register(RecommendationClick)
class RecommendationClickAdmin(admin.ModelAdmin):
    list_display = ('created_at', 'source', 'game_name', 'rank', 'query', 'user')
    list_filter = ('source', 'rank', 'created_at')
    search_fields = ('game_name', 'query')
    date_hierarchy = 'created_at'
    readonly_fields = ('created_at',)
