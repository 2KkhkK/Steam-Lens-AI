"""추천 비즈니스 로직.

구조
    views.py       HTTP 관심사만 (요청 파싱 / 인증 / 렌더)
    services.py    추천 로직          <- 이 파일
    utils.py       외부 API 연동
    similarity.py  순수 계산 (Django 무관, 테스트 가능)

이전 버전 대비 주요 변경
  1. 태그를 매 요청마다 정규식으로 파싱하던 것을 기동 시 1회 파싱으로 변경
  2. argsort(O(N log N) 전수 정렬) -> argpartition(O(N))
  3. 임베딩을 미리 L2 정규화해 코사인 유사도를 단순 내적으로 계산
  4. 외부 API 6건 순차 호출 -> ThreadPoolExecutor 병렬 호출
  5. 캐시 키를 hash() -> sha256 (재시작해도 캐시가 유지된다)
  6. 정렬용 점수와 화면 표시용 점수를 분리
  7. 한글 검색어 지원 + difflib 오타 교정
  8. print() -> logging
"""

import difflib
import logging
import os
import pickle
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from collections import Counter

import numpy as np
import pandas as pd
from django.conf import settings
from django.core.cache import cache

from . import similarity as sim
from .utils import cache_key, extract_app_id, get_historical_low, get_steam_price_info

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(BASE_DIR, 'cleaned_games.csv')
SAMPLE_CSV_PATH = os.path.join(BASE_DIR, 'sample_games.csv')
EMBED_PATH = os.path.join(BASE_DIR, 'steam_embeddings.pkl')

# 임베딩을 즉석 생성해도 괜찮은 최대 행 수(샘플 데이터셋 데모용).
ON_THE_FLY_LIMIT = 3000

_HANGUL_RE = re.compile(r'[가-힣]')


def _conf(key, default=None):
    return getattr(settings, 'RECOMMENDER', {}).get(key, default)


# -----------------------------------------------------------------------------
# 카탈로그 (데이터 + 임베딩) 지연 로딩
# -----------------------------------------------------------------------------
class Catalog:
    """게임 데이터프레임과 임베딩을 한 덩어리로 들고 있는 객체.

    예전에는 모듈 최상단에서 곧바로 pd.read_csv를 실행했다. 그래서
    데이터 파일이 없으면 import만 해도 df=None이 되고, 테스트에서
    services를 불러오는 것만으로 디스크 I/O가 발생했다.
    지금은 처음 필요할 때 한 번만 로딩한다(프로세스당 1회).
    """

    def __init__(self, df, embeddings, tag_dicts, source_path):
        self.df = df
        self.embeddings = embeddings          # L2 정규화 완료 (N, D)
        self.tag_dicts = tag_dicts            # [{태그: 투표수}, ...] 행 순서와 일치
        self.source_path = source_path
        self.names = df['Name'].astype(str).tolist()
        self.lower_names = [n.lower() for n in self.names]
        # App_ID -> 행 인덱스. 대시보드에서 보유 게임을 찾을 때 O(1)로 조회한다.
        self.appid_to_index = {}
        for pos, app_id in enumerate(df['App_ID'].tolist()):
            if app_id and app_id not in self.appid_to_index:
                self.appid_to_index[app_id] = pos

    def __len__(self):
        return len(self.df)


_catalog = None
_catalog_failed = False
_catalog_lock = threading.Lock()


def _resolve_csv_path():
    if os.path.exists(CSV_PATH):
        return CSV_PATH
    # 클론 직후 바로 데모가 뜨도록 소형 샘플로 폴백한다.
    if os.path.exists(SAMPLE_CSV_PATH):
        logger.warning(
            'cleaned_games.csv가 없어 sample_games.csv로 동작합니다. '
            '전체 데이터는 README의 "데이터 준비" 절을 참고하세요.'
        )
        return SAMPLE_CSV_PATH
    return None


def _load_embeddings(df, csv_path):
    """임베딩 파일을 읽는다. 없으면 (샘플처럼 작은 데이터에 한해) 즉석 생성."""
    if os.path.exists(EMBED_PATH):
        with open(EMBED_PATH, 'rb') as fp:
            data = pickle.load(fp)
        embeddings = np.asarray(data['embeddings'], dtype=np.float32)
        if embeddings.shape[0] != len(df):
            raise ValueError(
                f'임베딩 행 수({embeddings.shape[0]})와 CSV 행 수({len(df)})가 다릅니다. '
                'data_check.py -> embed.py 순서로 다시 생성하세요.'
            )
        return embeddings

    if len(df) > ON_THE_FLY_LIMIT:
        raise FileNotFoundError(
            f'{EMBED_PATH} 가 없습니다. `python embed.py`로 먼저 생성하세요.'
        )

    # 샘플 데이터 정도 규모면 그 자리에서 만들어 쓰는 편이 낫다.
    from sentence_transformers import SentenceTransformer  # 무거우므로 지연 import

    logger.warning('임베딩 파일이 없어 %d건을 즉석 생성합니다 (샘플 데이터 모드).', len(df))
    model = SentenceTransformer(_conf('MODEL_NAME', 'all-MiniLM-L6-v2'))
    texts = build_embedding_texts(df)
    embeddings = np.asarray(model.encode(texts, batch_size=32), dtype=np.float32)

    try:
        with open(EMBED_PATH, 'wb') as fp:
            pickle.dump({'names': df['Name'].tolist(), 'embeddings': embeddings}, fp)
    except OSError as exc:
        logger.warning('임베딩 캐시 저장 실패: %s', exc)
    return embeddings


def build_embedding_texts(df):
    """데이터프레임 전체의 임베딩 입력 문자열.

    규칙 자체는 similarity.build_embedding_text 한 곳에만 있다.
    예전처럼 embed.py와 services.py가 각자 문자열을 조립하면,
    한쪽만 고쳤을 때 벡터 공간이 어긋나도 아무 에러가 나지 않는다.
    """
    return [
        sim.build_embedding_text(row.get('Genres'), row.get('Tags'), row.get('About_the_game'))
        for _, row in df.iterrows()
    ]


def get_catalog():
    """카탈로그를 반환한다. 로딩 불가 시 None (호출 측이 사용자 메시지 처리)."""
    global _catalog, _catalog_failed

    if _catalog is not None:
        return _catalog
    if _catalog_failed:
        return None

    with _catalog_lock:
        if _catalog is not None:
            return _catalog
        if _catalog_failed:
            return None

        csv_path = _resolve_csv_path()
        if csv_path is None:
            logger.error(
                '게임 데이터가 없습니다. README의 "데이터 준비"에 따라 '
                'cleaned_games.csv를 생성하거나 sample_games.csv를 두세요.'
            )
            _catalog_failed = True
            return None

        try:
            df = pd.read_csv(csv_path, encoding='utf-8-sig')

            # App_ID는 헤더 이미지 URL에서 뽑는다. 추출 실패 시 NaN이 되어
            # 보유 게임 필터가 조용히 어긋나던 문제가 있어 빈 문자열로 채운다.
            if 'App_ID' not in df.columns:
                df['App_ID'] = df['Image_URL'].map(extract_app_id)
            df['App_ID'] = df['App_ID'].fillna('').astype(str).str.strip()

            missing_appid = int((df['App_ID'] == '').sum())
            if missing_appid:
                logger.warning(
                    'App_ID를 추출하지 못한 게임 %d건 (전체 %d건). '
                    '해당 게임은 보유 여부 필터에서 제외됩니다.',
                    missing_appid, len(df),
                )

            embeddings = _load_embeddings(df, csv_path)
            embeddings = sim.l2_normalize(embeddings)

            # 태그 파싱을 기동 시 1회로 옮긴다. 예전에는 검색 한 번에
            # 후보 1000건을 매번 정규식으로 파싱했다.
            tag_dicts = [sim.parse_tags(raw) for raw in df['Tags'].tolist()]

            _catalog = Catalog(df, embeddings, tag_dicts, csv_path)
            logger.info(
                '카탈로그 로딩 완료: %d개 게임, 임베딩 %s, 출처=%s',
                len(df), embeddings.shape, os.path.basename(csv_path),
            )
            return _catalog

        except Exception as exc:
            logger.exception('카탈로그 로딩 실패: %s', exc)
            _catalog_failed = True
            return None


def reset_catalog():
    """테스트에서 카탈로그를 갈아끼우기 위한 훅."""
    global _catalog, _catalog_failed
    with _catalog_lock:
        _catalog = None
        _catalog_failed = False


# -----------------------------------------------------------------------------
# 텍스트 유틸
# -----------------------------------------------------------------------------
def safe_float(val, default=0.0):
    try:
        if isinstance(val, str):
            val = val.replace('$', '').replace(',', '').strip()
        result = float(val)
    except (ValueError, TypeError):
        return default
    # NaN은 float()을 통과하지만 이후 계산을 전부 오염시킨다.
    return default if result != result else result


def clean_game_text(text):
    if not isinstance(text, str):
        return '설명 없음'
    text = re.sub(r'\?{2,}', '', text)   # 인코딩 깨진 흔적 제거
    text = re.sub(r'\s+', ' ', text)
    cleaned = text.strip()
    return cleaned or '설명 없음'


def _get_translator():
    """deep-translator는 선택 의존성으로 둔다.

    설치되어 있지 않아도 서비스와 테스트가 동작해야 한다(번역만 건너뛴다).
    """
    try:
        # pyrefly: ignore [missing-import]
        from deep_translator import GoogleTranslator
        return GoogleTranslator
    except ImportError:
        logger.info('deep-translator 미설치: 번역 없이 원문을 표시합니다.')
        return None


def translate_text(text, target='ko', max_length=400, ttl=86400 * 7):
    """지연 번역 + 캐싱. 화면에 실제로 보이는 텍스트만 번역한다."""
    if not text or text == '설명 없음' or not _conf('TRANSLATE', True):
        return text

    if len(text) > max_length:
        text = text[:max_length].rstrip() + '…'

    key = cache_key(f'trans_{target}', text)
    cached = cache.get(key)
    if cached is not None:
        return cached

    translator_cls = _get_translator()
    if translator_cls is None:
        return text

    try:
        translated = translator_cls(source='auto', target=target).translate(text)
    except Exception as exc:
        # 번역 실패가 추천 자체를 막아서는 안 된다.
        logger.warning('번역 실패: %s', exc)
        return text

    if not translated:
        return text

    cache.set(key, translated, ttl)
    return translated


def _looks_korean(text):
    return bool(_HANGUL_RE.search(text or ''))


def translate_query_to_english(query):
    """한글 검색어를 영어로 바꿔 본다.

    게임 데이터(Name/설명문)와 임베딩 모델이 모두 영어라, 예전에는
    '엘든 링'으로 검색하면 아무것도 찾지 못했다. 근본 해결은 다국어
    임베딩 모델 교체이고(README 참고), 이건 그 전까지의 임시 처방이다.
    """
    if not _looks_korean(query):
        return query

    key = cache_key('query_en', query)
    cached = cache.get(key)
    if cached is not None:
        return cached

    translated = translate_text(query, target='en', max_length=100, ttl=86400 * 30)
    result = translated or query
    cache.set(key, result, 86400 * 30)
    logger.info('한글 검색어 번역: %r -> %r', query, result)
    return result


# -----------------------------------------------------------------------------
# 게임 찾기
# -----------------------------------------------------------------------------
def find_game_index(catalog, query):
    """검색어에 해당하는 행 인덱스를 찾는다.

    반환: (위치 인덱스 또는 None, 후보 게임명 리스트)
    """
    if not query:
        return None, []

    lowered = query.lower().strip()

    # 1) 완전 일치
    for pos, name in enumerate(catalog.lower_names):
        if name == lowered:
            return pos, []

    # 2) 부분 일치 중 이름이 가장 짧은 것
    #    ("Portal"로 검색하면 "Portal 2"보다 "Portal"이 잡히도록)
    partial = [pos for pos, name in enumerate(catalog.lower_names) if lowered in name]
    if partial:
        best = min(partial, key=lambda p: len(catalog.lower_names[p]))
        return best, []

    # 3) 오타 교정 제안
    #    index.html에는 예전부터 'suggestions'를 표시하는 코드가 있었지만
    #    뷰에서 값을 넘겨주지 않아 동작하지 않는 죽은 코드였다.
    matches = difflib.get_close_matches(query, catalog.names, n=5, cutoff=0.6)
    if not matches:
        matches = difflib.get_close_matches(lowered, catalog.lower_names, n=5, cutoff=0.5)
        matches = [catalog.names[catalog.lower_names.index(m)] for m in matches]

    return None, matches


# -----------------------------------------------------------------------------
# 추천 본체
# -----------------------------------------------------------------------------
def _rank_candidates(catalog, scores, target_tags, exclude_indices,
                     w_tag, w_meta=0.0, pool=None, top_n=None):
    """1차 후보 축소 -> 태그 기반 리랭킹이라는 2단계 랭킹.

    전체에 태그 유사도를 계산하면 느리므로, 빠른 벡터 연산으로 pool개까지
    좁힌 뒤(Retrieval) 그 안에서만 정밀 정렬한다(Ranking).
    """
    pool = pool or _conf('CANDIDATE_POOL', 1000)
    top_n = top_n or _conf('TOP_N', 6)
    weighted = _conf('WEIGHTED_TAGS', True)

    candidate_indices = sim.top_k_indices(scores, pool, exclude=exclude_indices)

    candidates = []
    for i in candidate_indices:
        i = int(i)
        bert_score = float(scores[i])
        tag_score = sim.tag_similarity(target_tags, catalog.tag_dicts[i], weighted=weighted)

        meta_score = 0.0
        if w_meta:
            meta_score = safe_float(catalog.df.iloc[i].get('Metacritic_score', 0)) / 100.0

        candidates.append({
            'idx': i,
            'final_score': sim.hybrid_score(bert_score, tag_score, meta_score, w_tag, w_meta),
            'bert_score': bert_score,
            'tag_score': tag_score,
        })

    candidates.sort(key=lambda c: c['final_score'], reverse=True)
    return candidates[:top_n]


def get_search_recommendations(query, owned_appids=None):
    """검색어 기반 추천.

    반환: (결과 리스트, 에러 메시지, meta dict)
      meta = {'matched_name': 실제로 매칭된 게임명, 'suggestions': 오타 교정 후보}
    """
    owned_appids = owned_appids or set()
    meta = {'matched_name': '', 'suggestions': []}

    catalog = get_catalog()
    if catalog is None:
        return None, '게임 데이터가 준비되지 않았습니다. README의 "데이터 준비"를 확인해 주세요.', meta
    if not query:
        return None, '검색어를 입력해 주세요.', meta

    effective_query = translate_query_to_english(query)

    idx, suggestions = find_game_index(catalog, effective_query)
    if idx is None and effective_query != query:
        idx, suggestions = find_game_index(catalog, query)

    if idx is None:
        meta['suggestions'] = suggestions
        return None, f"'{query}'와 일치하는 게임을 찾지 못했습니다.", meta

    scores = catalog.embeddings @ catalog.embeddings[idx]

    # 자기 자신은 항상 제외한다. 예전에는 검색 경로에서 "argsort()[-1001:-1]"로
    # 맨 뒤 1개를 잘라내 자기 자신을 제거했는데, 이는 '자기 자신이 반드시
    # 1등'이라는 가정에 의존한다. 설명문이 동일한 중복 등록 게임이 있으면
    # 깨지는 방식이라 명시적 제외로 바꿨다.
    exclude = {idx}
    if owned_appids:
        exclude |= {
            catalog.appid_to_index[a] for a in owned_appids
            if a in catalog.appid_to_index
        }

    ranked = _rank_candidates(
        catalog, scores,
        target_tags=catalog.tag_dicts[idx],
        exclude_indices=exclude,
        w_tag=_conf('SEARCH_TAG_WEIGHT', 3.0),
    )

    meta['matched_name'] = catalog.names[idx]
    results = format_recommendation_results(catalog, ranked)
    return results, None, meta


def build_user_profile(catalog, owned_games, limit=None):
    """플레이 기록으로 유저 프로필 벡터와 핵심 태그를 만든다.

    예전에는 상위 10개 게임 벡터의 '단순 평균'이었다. 그래서 2000시간을
    플레이한 게임과 21시간짜리 게임이 취향에 똑같이 1/10씩 기여했다.
    플레이타임에 로그를 씌워 가중 평균한다(로그를 쓰는 이유: 시간 자체를
    가중치로 쓰면 최상위 한 게임이 프로필을 독식한다).

    반환: (프로필 벡터 또는 None, 핵심 태그 dict, 사용된 게임 수)
    """
    limit = limit or _conf('PROFILE_GAMES', 10)

    ranked_games = sorted(
        owned_games,
        key=lambda g: g.get('playtime_forever', 0) or 0,
        reverse=True,
    )[:limit]

    vectors, weights = [], []
    tag_weights = Counter()

    for game in ranked_games:
        app_id = str(game.get('appid', '')).strip()
        pos = catalog.appid_to_index.get(app_id)
        if pos is None:
            continue

        playtime = max(0.0, safe_float(game.get('playtime_forever', 0)))
        weight = float(np.log1p(playtime / 60.0)) + 0.1  # 분 -> 시간, 0시간도 최소 기여

        vectors.append(catalog.embeddings[pos])
        weights.append(weight)

        for tag, votes in catalog.tag_dicts[pos].items():
            tag_weights[tag] += weight * votes

    if not vectors:
        return None, {}, 0

    matrix = np.asarray(vectors, dtype=np.float32)
    weight_arr = np.asarray(weights, dtype=np.float32).reshape(-1, 1)
    profile = (matrix * weight_arr).sum(axis=0) / weight_arr.sum()

    core_tags = dict(tag_weights.most_common(20))
    return profile, core_tags, len(vectors)


def get_dashboard_recommendations(owned_games, owned_appids=None):
    """플레이 기록 기반 개인화 추천."""
    catalog = get_catalog()
    if catalog is None:
        return None, '게임 데이터가 준비되지 않았습니다. README의 "데이터 준비"를 확인해 주세요.'

    owned_appids = owned_appids or {str(g.get('appid', '')) for g in owned_games}

    profile, core_tags, used = build_user_profile(catalog, owned_games)
    if profile is None:
        return None, '보유하신 게임 중 추천 데이터베이스에 등록된 게임이 없습니다.'

    logger.info('프로필 벡터 생성: 보유 %d개 중 %d개 사용', len(owned_games), used)

    norm = np.linalg.norm(profile)
    if norm > 0:
        profile = profile / norm
    scores = catalog.embeddings @ profile.astype(np.float32)

    exclude = {
        catalog.appid_to_index[a] for a in owned_appids
        if a in catalog.appid_to_index
    }

    ranked = _rank_candidates(
        catalog, scores,
        target_tags=core_tags,
        exclude_indices=exclude,
        w_tag=_conf('DASHBOARD_TAG_WEIGHT', 5.0),
        w_meta=_conf('DASHBOARD_META_WEIGHT', 2.0),
    )

    return format_recommendation_results(catalog, ranked), None


# -----------------------------------------------------------------------------
# 결과 조립
# -----------------------------------------------------------------------------
def _enrich(catalog, item, rank):
    """추천 1건에 가격/최저가/번역을 붙인다. 스레드에서 병렬 실행된다."""
    i = item['idx']
    row = catalog.df.iloc[i]

    game_name = str(row['Name'])
    image_url = row.get('Image_URL', '')
    app_id = str(row.get('App_ID', '') or '')

    display_price, original_price, discount_percent, is_discounted = get_steam_price_info(image_url)

    # 스팀이 가격 정보를 안 주더라도 CSV에 정가가 있으면 그걸 쓴다.
    raw_price = safe_float(row.get('Price', 0))
    if display_price == 'Free to Play' and raw_price > 0:
        display_price = f'${raw_price:g}'

    about = translate_text(clean_game_text(row.get('About_the_game', '설명 없음')))

    return {
        'rank': rank,
        'name': game_name,
        'app_id': app_id,
        'about': about,
        'score': int(safe_float(row.get('Metacritic_score', 0))),
        'image': image_url,
        # 정렬에 쓴 점수는 가중치 때문에 1.0을 넘을 수 있어 화면에 그대로
        # 내보내면 "2.34점" 같은 해석 불가능한 값이 된다. 표시는 코사인
        # 유사도 기반 0~100%로, 정렬 근거는 디버깅용으로 따로 보관한다.
        'match_percent': sim.display_percent(item['bert_score']),
        'rank_score': round(item['final_score'], 3),
        'tag_score': round(item['tag_score'], 3),
        'price': display_price,
        'original_price': original_price,
        'discount_percent': discount_percent,
        'is_discounted': is_discounted,
        'historical_low': get_historical_low(game_name),
    }


def format_recommendation_results(catalog, candidates):
    """추천 목록에 부가 정보를 붙인다.

    예전에는 for 루프로 6건을 하나씩 처리해서, 캐시가 비어 있으면
    (스팀 가격 + ITAD 2회 + 번역) x 6 = 최대 24회를 순차 대기했다.
    전부 네트워크 I/O 대기라 GIL 영향이 없어, 스레드 풀로 묶으면
    거의 그대로 시간이 나눠진다.
    """
    if not candidates:
        return []

    workers = max(1, min(_conf('ENRICH_WORKERS', 6), len(candidates)))

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(_enrich, catalog, item, rank)
            for rank, item in enumerate(candidates, start=1)
        ]
        results = []
        for future in futures:   # 제출 순서 = 추천 순위 순서
            try:
                results.append(future.result())
            except Exception as exc:
                logger.exception('추천 항목 조립 실패: %s', exc)

    return results
