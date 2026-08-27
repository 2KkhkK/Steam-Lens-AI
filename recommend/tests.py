"""유닛 테스트.

    python manage.py test recommend

추천 로직의 핵심은 외부 의존이 없는 순수 함수라 테스트하기 쉬운데도
예전에는 테스트가 0개였다. 여기서는
  - similarity.py의 계산 정확성
  - services.py의 랭킹/프로필 로직 (가짜 카탈로그 주입)
  - utils.py의 외부 API 실패 시 폴백 (requests를 mock)
  - views의 기본 동작
을 검증한다.
"""

import unittest.mock as mock

import numpy as np
import pandas as pd
from django.core.cache import cache
from django.test import SimpleTestCase, TestCase
from django.urls import reverse

from . import services
from . import similarity as sim
from . import utils
from .models import RecommendationClick, SearchLog


# =============================================================================
# similarity.py — 순수 계산
# =============================================================================
class ParseTagsTests(SimpleTestCase):
    def test_dict_string_with_votes(self):
        tags = sim.parse_tags("{'Action': 12345, 'Indie': 500}")
        self.assertEqual(tags, {'action': 12345, 'indie': 500})

    def test_apostrophe_in_tag_name(self):
        """예전 정규식 r"'([^']+)'\\s*:" 은 이 입력에서 깨졌다."""
        tags = sim.parse_tags('{"Assassin\'s Creed": 10, "Stealth": 5}')
        self.assertIn("assassin's creed", tags)
        self.assertEqual(tags["assassin's creed"], 10)

    def test_nan_and_empty(self):
        self.assertEqual(sim.parse_tags(float('nan')), {})
        self.assertEqual(sim.parse_tags(None), {})
        self.assertEqual(sim.parse_tags(''), {})
        self.assertEqual(sim.parse_tags('nan'), {})
        self.assertEqual(sim.parse_tags('{}'), {})

    def test_plain_comma_separated(self):
        self.assertEqual(sim.parse_tags('Action, RPG'), {'action': 1, 'rpg': 1})

    def test_list_form(self):
        self.assertEqual(sim.parse_tags("['Action', 'RPG']"), {'action': 1, 'rpg': 1})

    def test_case_and_whitespace_normalised(self):
        self.assertEqual(sim.parse_tags("{'  AcTiOn  ': 3}"), {'action': 3})


class JaccardTests(SimpleTestCase):
    def test_identical_sets(self):
        s = {'action', 'rpg'}
        self.assertEqual(sim.jaccard(s, s), 1.0)

    def test_disjoint_sets(self):
        self.assertEqual(sim.jaccard({'action'}, {'puzzle'}), 0.0)

    def test_partial_overlap(self):
        # 교집합 1개 {action}, 합집합 3개 {action, rpg, puzzle}
        self.assertAlmostEqual(sim.jaccard({'action', 'rpg'}, {'action', 'puzzle'}), 1 / 3)

    def test_empty_input(self):
        self.assertEqual(sim.jaccard(set(), {'action'}), 0.0)
        self.assertEqual(sim.jaccard(set(), set()), 0.0)


class WeightedJaccardTests(SimpleTestCase):
    def test_identical_is_one(self):
        tags = {'action': 100, 'rpg': 50}
        self.assertAlmostEqual(sim.weighted_jaccard(tags, tags), 1.0)

    def test_scale_invariant(self):
        """인기작(수만 표)과 인디게임(수십 표)을 공정하게 비교해야 한다."""
        popular = {'action': 50000, 'rpg': 25000}
        niche = {'action': 50, 'rpg': 25}
        self.assertAlmostEqual(sim.weighted_jaccard(popular, niche), 1.0)

    def test_dominant_tag_matters_more(self):
        """대표 태그가 겹치는 쪽이 소수 태그만 겹치는 쪽보다 높아야 한다."""
        target = {'souls-like': 1000, 'indie': 10}
        strong = {'souls-like': 900, 'indie': 5}
        weak = {'puzzle': 1000, 'indie': 900}
        self.assertGreater(
            sim.weighted_jaccard(target, strong),
            sim.weighted_jaccard(target, weak),
        )

    def test_empty(self):
        self.assertEqual(sim.weighted_jaccard({}, {'action': 1}), 0.0)


class ScoreTests(SimpleTestCase):
    def test_hybrid_score_formula(self):
        # 0.5 * (1 + 0.2*3 + 0.9*2) = 0.5 * 3.4 = 1.7
        self.assertAlmostEqual(
            sim.hybrid_score(0.5, 0.2, 0.9, w_tag=3.0, w_meta=2.0), 1.7
        )

    def test_hybrid_score_can_exceed_one(self):
        """정렬용 점수는 1을 넘을 수 있다. 그래서 표시용과 분리해야 한다."""
        self.assertGreater(sim.hybrid_score(0.9, 1.0, w_tag=3.0), 1.0)

    def test_display_percent_is_bounded(self):
        self.assertEqual(sim.display_percent(0.0), 0)
        self.assertEqual(sim.display_percent(1.0), 100)
        self.assertEqual(sim.display_percent(0.734), 73)
        self.assertEqual(sim.display_percent(-0.5), 0)     # 음의 코사인
        self.assertEqual(sim.display_percent(5.0), 100)    # 범위 초과 방어
        self.assertEqual(sim.display_percent(float('nan')), 0)


class VectorTests(SimpleTestCase):
    def test_l2_normalize_unit_length(self):
        mat = sim.l2_normalize(np.array([[3.0, 4.0], [1.0, 0.0]]))
        np.testing.assert_allclose(np.linalg.norm(mat, axis=1), [1.0, 1.0], atol=1e-6)

    def test_l2_normalize_handles_zero_vector(self):
        """0벡터를 0으로 나누면 NaN이 되어 이후 계산 전체가 오염된다."""
        mat = sim.l2_normalize(np.array([[0.0, 0.0], [3.0, 4.0]]))
        self.assertFalse(np.isnan(mat).any())

    def test_top_k_matches_full_sort(self):
        """argpartition 최적화가 argsort와 같은 결과를 내야 한다."""
        rng = np.random.default_rng(0)
        scores = rng.random(5000)
        fast = sim.top_k_indices(scores, 10)
        slow = np.argsort(scores)[::-1][:10]
        np.testing.assert_array_equal(fast, slow)

    def test_top_k_respects_exclude(self):
        scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
        result = sim.top_k_indices(scores, 3, exclude={0, 2})
        np.testing.assert_array_equal(result, [1, 3, 4])

    def test_top_k_when_k_exceeds_size(self):
        scores = np.array([0.1, 0.9])
        np.testing.assert_array_equal(sim.top_k_indices(scores, 10), [1, 0])

    def test_top_k_empty(self):
        self.assertEqual(len(sim.top_k_indices(np.array([]), 5)), 0)


class MetricTests(SimpleTestCase):
    def test_precision_at_k(self):
        self.assertAlmostEqual(sim.precision_at_k([1, 0, 1, 0, 0], 5), 0.4)

    def test_ndcg_rewards_ordering(self):
        """Precision은 같지만 정답이 위에 있는 쪽이 nDCG가 높아야 한다."""
        good = [1, 1, 0, 0, 0]
        bad = [0, 0, 0, 1, 1]
        self.assertEqual(sim.precision_at_k(good, 5), sim.precision_at_k(bad, 5))
        self.assertGreater(sim.ndcg_at_k(good, 5), sim.ndcg_at_k(bad, 5))

    def test_ndcg_perfect_is_one(self):
        self.assertAlmostEqual(sim.ndcg_at_k([1, 1, 1], 3), 1.0)

    def test_ndcg_all_zero(self):
        self.assertEqual(sim.ndcg_at_k([0, 0, 0], 3), 0.0)

    def test_diversity_identical_items_is_zero(self):
        vectors = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
        self.assertAlmostEqual(sim.intra_list_diversity(vectors), 0.0, places=5)

    def test_diversity_orthogonal_items_is_one(self):
        vectors = np.array([[1.0, 0.0], [0.0, 1.0]])
        self.assertAlmostEqual(sim.intra_list_diversity(vectors), 1.0, places=5)


class EmbeddingTextTests(SimpleTestCase):
    def test_field_order_puts_tags_before_description(self):
        """모델의 256토큰 제한 때문에 장르/태그가 앞에 와야 한다."""
        text = sim.build_embedding_text('Action', "{'Souls-like': 9}", 'A long description')
        self.assertTrue(text.startswith('Genres: Action.'))
        self.assertLess(text.index('Tags:'), text.index('Description:'))

    def test_nan_fields_become_empty(self):
        text = sim.build_embedding_text(float('nan'), None, 'desc')
        self.assertEqual(text, 'Genres: . Tags: . Description: desc')


# =============================================================================
# services.py — 가짜 카탈로그를 주입해 랭킹 로직만 검증
# =============================================================================
def make_fake_catalog():
    """실제 데이터 없이 랭킹 로직을 시험하기 위한 최소 카탈로그.

    임베딩을 손으로 만들어 '어떤 게임이 서로 가까운지'를 통제한다.
      0 Souls Game A / 1 Souls Game B : 거의 같은 방향 (액션RPG)
      2 Farm Sim                      : 직교 방향 (힐링)
      3 Puzzle Box                    : 또 다른 직교 방향
    """
    df = pd.DataFrame([
        {'Name': 'Souls Game A', 'About_the_game': 'A hard action rpg in a dark world.',
         'Price': 59.99, 'Metacritic_score': 96, 'Genres': 'Action, RPG',
         'Tags': "{'Souls-like': 1000, 'Difficult': 800, 'Dark Fantasy': 600}",
         'Image_URL': 'https://cdn.akamai.steamstatic.com/steam/apps/1001/header.jpg'},
        {'Name': 'Souls Game B', 'About_the_game': 'Another punishing action rpg adventure.',
         'Price': 49.99, 'Metacritic_score': 90, 'Genres': 'Action, RPG',
         'Tags': "{'Souls-like': 900, 'Difficult': 700, 'Dark Fantasy': 500}",
         'Image_URL': 'https://cdn.akamai.steamstatic.com/steam/apps/1002/header.jpg'},
        {'Name': 'Farm Sim', 'About_the_game': 'Grow crops and relax in a cozy village.',
         'Price': 14.99, 'Metacritic_score': 89, 'Genres': 'Simulation',
         'Tags': "{'Farming Sim': 1000, 'Relaxing': 900, 'Cozy': 700}",
         'Image_URL': 'https://cdn.akamai.steamstatic.com/steam/apps/1003/header.jpg'},
        {'Name': 'Puzzle Box', 'About_the_game': 'Solve intricate spatial puzzles.',
         'Price': 9.99, 'Metacritic_score': 85, 'Genres': 'Puzzle',
         'Tags': "{'Puzzle': 1000, 'Singleplayer': 400}",
         'Image_URL': 'https://cdn.akamai.steamstatic.com/steam/apps/1004/header.jpg'},
    ])
    df['App_ID'] = df['Image_URL'].map(utils.extract_app_id)

    embeddings = sim.l2_normalize(np.array([
        [1.0, 0.0, 0.0],
        [0.98, 0.2, 0.0],   # Souls A와 매우 가까움
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32))

    tag_dicts = [sim.parse_tags(t) for t in df['Tags'].tolist()]
    return services.Catalog(df, embeddings, tag_dicts, 'fake')


class FakeCatalogMixin:
    """서비스가 쓰는 전역 카탈로그를 가짜로 바꿔치기한다."""

    def setUp(self):
        super().setUp()
        cache.clear()
        self.catalog = make_fake_catalog()
        patcher = mock.patch.object(services, 'get_catalog', return_value=self.catalog)
        patcher.start()
        self.addCleanup(patcher.stop)

        # 외부 API와 번역은 네트워크를 타므로 전부 막는다.
        for target, value in (
            ('get_steam_price_info', ('₩ 10,000', '', 0, False)),
            ('get_historical_low', '₩ 5,000'),
        ):
            p = mock.patch.object(services, target, return_value=value)
            p.start()
            self.addCleanup(p.stop)

        p = mock.patch.object(services, 'translate_text', side_effect=lambda t, **kw: t)
        p.start()
        self.addCleanup(p.stop)


class FindGameTests(FakeCatalogMixin, SimpleTestCase):
    def test_exact_match(self):
        idx, suggestions = services.find_game_index(self.catalog, 'Souls Game A')
        self.assertEqual(idx, 0)
        self.assertEqual(suggestions, [])

    def test_case_insensitive(self):
        idx, _ = services.find_game_index(self.catalog, 'souls game a')
        self.assertEqual(idx, 0)

    def test_partial_match_prefers_shortest_name(self):
        idx, _ = services.find_game_index(self.catalog, 'Farm')
        self.assertEqual(idx, 2)

    def test_typo_returns_suggestions(self):
        """예전에는 후보를 만들지 않아 index.html의 표시 코드가 죽어 있었다."""
        idx, suggestions = services.find_game_index(self.catalog, 'Soul Gaem A')
        self.assertIsNone(idx)
        self.assertTrue(suggestions)
        self.assertIn('Souls Game A', suggestions)

    def test_no_match_no_suggestion(self):
        idx, suggestions = services.find_game_index(self.catalog, 'zzzzzzzzzz')
        self.assertIsNone(idx)
        self.assertEqual(suggestions, [])


class SearchRecommendationTests(FakeCatalogMixin, SimpleTestCase):
    def test_similar_game_ranked_first(self):
        results, error, meta = services.get_search_recommendations('Souls Game A')
        self.assertIsNone(error)
        self.assertEqual(results[0]['name'], 'Souls Game B')
        self.assertEqual(meta['matched_name'], 'Souls Game A')

    def test_query_game_itself_is_excluded(self):
        """자기 자신이 추천에 섞여 나오면 안 된다."""
        results, _, _ = services.get_search_recommendations('Souls Game A')
        self.assertNotIn('Souls Game A', [r['name'] for r in results])

    def test_owned_games_are_excluded(self):
        owned = {self.catalog.df.iloc[1]['App_ID']}   # Souls Game B 보유
        results, _, _ = services.get_search_recommendations('Souls Game A', owned)
        self.assertNotIn('Souls Game B', [r['name'] for r in results])

    def test_unknown_query_returns_error_and_suggestions(self):
        results, error, meta = services.get_search_recommendations('Soul Gaem A')
        self.assertIsNone(results)
        self.assertIn('찾지 못했습니다', error)
        self.assertTrue(meta['suggestions'])

    def test_display_percent_within_range(self):
        results, _, _ = services.get_search_recommendations('Souls Game A')
        for item in results:
            self.assertGreaterEqual(item['match_percent'], 0)
            self.assertLessEqual(item['match_percent'], 100)

    def test_rank_is_sequential(self):
        results, _, _ = services.get_search_recommendations('Souls Game A')
        self.assertEqual([r['rank'] for r in results], list(range(1, len(results) + 1)))


class UserProfileTests(FakeCatalogMixin, SimpleTestCase):
    def test_playtime_weighting_favours_dominant_game(self):
        """플레이타임을 정렬에만 쓰고 가중치로 쓰지 않던 문제의 회귀 테스트.

        Souls를 2000시간, Farm을 1시간 한 유저의 프로필은
        Souls 쪽에 훨씬 가까워야 한다.
        """
        owned = [
            {'appid': '1001', 'playtime_forever': 120000},  # 2000시간
            {'appid': '1003', 'playtime_forever': 60},      # 1시간
        ]
        profile, core_tags, used = services.build_user_profile(self.catalog, owned)
        self.assertEqual(used, 2)

        profile = profile / np.linalg.norm(profile)
        souls_sim = float(profile @ self.catalog.embeddings[0])
        farm_sim = float(profile @ self.catalog.embeddings[2])
        self.assertGreater(souls_sim, farm_sim)

    def test_core_tags_reflect_playtime(self):
        owned = [
            {'appid': '1001', 'playtime_forever': 120000},
            {'appid': '1003', 'playtime_forever': 60},
        ]
        _, core_tags, _ = services.build_user_profile(self.catalog, owned)
        self.assertGreater(core_tags.get('souls-like', 0), core_tags.get('farming sim', 0))

    def test_unknown_games_are_skipped(self):
        profile, _, used = services.build_user_profile(
            self.catalog, [{'appid': '999999', 'playtime_forever': 100}]
        )
        self.assertIsNone(profile)
        self.assertEqual(used, 0)

    def test_dashboard_excludes_owned(self):
        owned = [{'appid': '1001', 'playtime_forever': 6000}]
        results, error = services.get_dashboard_recommendations(owned)
        self.assertIsNone(error)
        self.assertNotIn('Souls Game A', [r['name'] for r in results])


class TextUtilTests(SimpleTestCase):
    def test_safe_float_strips_currency(self):
        self.assertAlmostEqual(services.safe_float('$19.99'), 19.99)
        self.assertAlmostEqual(services.safe_float('1,299'), 1299.0)

    def test_safe_float_bad_input(self):
        self.assertEqual(services.safe_float('무료'), 0.0)
        self.assertEqual(services.safe_float(None), 0.0)
        self.assertEqual(services.safe_float(float('nan')), 0.0)
        self.assertEqual(services.safe_float('', default=-1), -1)

    def test_clean_game_text(self):
        self.assertEqual(services.clean_game_text('Hello???  world'), 'Hello world')
        self.assertEqual(services.clean_game_text(None), '설명 없음')
        self.assertEqual(services.clean_game_text('   '), '설명 없음')


# =============================================================================
# utils.py — 외부 API를 mock으로 막고 폴백만 검증
# =============================================================================
class CacheKeyTests(SimpleTestCase):
    def test_deterministic(self):
        """hash()는 프로세스마다 값이 달라 재시작하면 캐시가 전부 무효화됐다."""
        self.assertEqual(utils.cache_key('t', 'hello'), utils.cache_key('t', 'hello'))

    def test_distinct_inputs_differ(self):
        self.assertNotEqual(utils.cache_key('t', 'hello'), utils.cache_key('t', 'world'))

    def test_known_value(self):
        # 값이 코드 변경으로 조용히 바뀌면 기존 캐시가 전부 미스가 된다.
        self.assertEqual(
            utils.cache_key('trans_ko', 'hello'),
            'trans_ko_2cf24dba5fb0a30e26e83b2ac5b9e29e',
        )


class AppIdTests(SimpleTestCase):
    def test_extracts_from_header_image(self):
        url = 'https://cdn.akamai.steamstatic.com/steam/apps/1245620/header.jpg'
        self.assertEqual(utils.extract_app_id(url), '1245620')

    def test_missing_returns_empty_string(self):
        """예전에는 NaN이 되어 보유 게임 필터가 조용히 어긋났다."""
        self.assertEqual(utils.extract_app_id('https://example.com/x.jpg'), '')
        self.assertEqual(utils.extract_app_id(None), '')
        self.assertEqual(utils.extract_app_id(float('nan')), '')


class ExternalApiFallbackTests(SimpleTestCase):
    def setUp(self):
        cache.clear()

    @mock.patch('recommend.utils.requests.get', side_effect=Exception('network down'))
    def test_historical_low_falls_back_to_mock(self, _mocked):
        """ITAD가 죽어도 mock_lowest_price.json으로 값을 낸다."""
        with mock.patch.dict(utils.MOCK_DATA, {'Elden Ring': {'currency': 'KRW', 'amount': 45000}}):
            self.assertEqual(utils.get_historical_low('Elden Ring'), '₩ 45,000')

    @mock.patch('recommend.utils.requests.get', side_effect=Exception('network down'))
    def test_historical_low_unknown_game_returns_empty(self, _mocked):
        self.assertEqual(utils.get_historical_low('Nonexistent Game 12345'), '')

    @mock.patch('recommend.utils.requests.get', side_effect=Exception('timeout'))
    def test_price_api_failure_is_safe(self, _mocked):
        """가격 조회 실패가 추천 자체를 막아서는 안 된다."""
        result = utils.get_steam_price_info(
            'https://cdn.akamai.steamstatic.com/steam/apps/1245620/header.jpg'
        )
        self.assertEqual(result, ('Free to Play', '', 0, False))

    @mock.patch('recommend.utils.requests.get')
    def test_price_parsed_with_discount(self, mocked):
        mocked.return_value = mock.Mock(
            raise_for_status=mock.Mock(),
            json=mock.Mock(return_value={
                '1245620': {
                    'success': True,
                    'data': {'price_overview': {
                        'final_formatted': '₩ 32,000',
                        'initial_formatted': '₩ 64,000',
                        'discount_percent': 50,
                    }},
                }
            }),
        )
        price, original, discount, is_discounted = utils.get_steam_price_info(
            'https://cdn.akamai.steamstatic.com/steam/apps/1245620/header.jpg'
        )
        self.assertEqual(price, '₩ 32,000')
        self.assertEqual(original, '₩ 64,000')
        self.assertEqual(discount, 50)
        self.assertTrue(is_discounted)

    @mock.patch('recommend.utils.requests.get')
    def test_price_result_is_cached(self, mocked):
        mocked.return_value = mock.Mock(
            raise_for_status=mock.Mock(),
            json=mock.Mock(return_value={'1245620': {'success': True, 'data': {}}}),
        )
        url = 'https://cdn.akamai.steamstatic.com/steam/apps/1245620/header.jpg'
        utils.get_steam_price_info(url)
        utils.get_steam_price_info(url)
        self.assertEqual(mocked.call_count, 1)

    @mock.patch('recommend.utils.requests.get', side_effect=Exception('down'))
    def test_no_owned_games_without_api_key(self, _mocked):
        with self.settings(STEAM_API_KEY=''):
            self.assertEqual(utils.get_user_owned_games('7656119'), [])


# =============================================================================
# views — DB가 필요한 테스트
# =============================================================================
class ViewTests(FakeCatalogMixin, TestCase):
    def test_index_renders(self):
        response = self.client.get(reverse('index'))
        self.assertEqual(response.status_code, 200)

    def test_search_renders_results(self):
        response = self.client.get(reverse('search'), {'q': 'Souls Game A'})
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Souls Game B')

    def test_search_logs_the_query(self):
        self.client.get(reverse('search'), {'q': 'Souls Game A'})
        log = SearchLog.objects.get()
        self.assertEqual(log.query, 'Souls Game A')
        self.assertEqual(log.matched_game, 'Souls Game A')
        self.assertGreater(log.result_count, 0)

    def test_search_without_query_shows_index(self):
        response = self.client.get(reverse('search'), {'q': '   '})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(SearchLog.objects.count(), 0)

    def test_typo_shows_suggestion_links(self):
        response = self.client.get(reverse('search'), {'q': 'Soul Gaem A'})
        self.assertContains(response, '혹시 이 게임인가요')
        self.assertContains(response, 'Souls Game A')

    def test_dashboard_requires_login(self):
        response = self.client.get(reverse('dashboard'))
        self.assertEqual(response.status_code, 302)

    def test_click_tracking_records_and_redirects(self):
        response = self.client.get(reverse('track_click'), {
            'app_id': '1001', 'name': 'Souls Game A', 'rank': '1', 'source': 'search',
        })
        self.assertEqual(response.status_code, 302)
        self.assertIn('store.steampowered.com/app/1001', response['Location'])

        click = RecommendationClick.objects.get()
        self.assertEqual(click.game_name, 'Souls Game A')
        self.assertEqual(click.rank, 1)

    def test_click_tracking_rejects_non_numeric_app_id(self):
        """리다이렉트 주소를 요청에서 받지 않으므로 오픈 리다이렉트가 불가능하다."""
        response = self.client.get(reverse('track_click'), {
            'app_id': 'https://evil.example.com', 'name': 'X',
        })
        self.assertEqual(response.status_code, 302)
        self.assertNotIn('evil.example.com', response['Location'])
