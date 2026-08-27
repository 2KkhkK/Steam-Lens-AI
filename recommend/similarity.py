"""
추천 엔진의 순수 계산 로직.

이 모듈은 Django / pandas / 외부 API에 의존하지 않는다. 덕분에
  - 웹 서비스(services.py)
  - 오프라인 평가(bulk_evaluate.py)
  - 유닛 테스트(tests.py)
세 곳이 '완전히 동일한' 유사도 정의를 공유한다. 예전에는 services.py와
bulk_evaluate.py가 태그 파싱 정규식을 각자 복사해 갖고 있어서, 한쪽만
고치면 평가 결과와 실제 서비스 동작이 어긋날 수 있었다.
"""

import ast
import math
import re

import numpy as np

# 스팀 tags 컬럼은 "{'Action': 12345, 'Indie': 6789}" 형태의 dict 문자열이다.
# ast.literal_eval이 실패할 때만 쓰는 최후의 수단.
_TAG_FALLBACK_RE = re.compile(r"['\"]([^'\"]+)['\"]\s*:\s*(\d+)")


def parse_tags(raw):
    """태그 문자열을 {태그명(소문자): 투표수} 딕셔너리로 변환한다.

    기존 구현은 정규식 r"'([^']+)'\\s*:" 하나로 처리해서 두 가지 문제가 있었다.
      1. "Assassin's Creed"처럼 태그명에 아포스트로피가 있으면 파싱이 깨진다.
      2. 투표수를 통째로 버려서, 대표 태그와 소수 의견 태그가 동등해진다.
    ast.literal_eval로 정확히 파싱하고 투표수까지 보존한다.
    """
    if raw is None:
        return {}
    # pandas의 NaN은 float이며 자기 자신과 같지 않다 (pandas를 import하지 않고 판별).
    if isinstance(raw, float) and math.isnan(raw):
        return {}

    text = str(raw).strip()
    if not text or text.lower() in ("nan", "none", "{}", "[]"):
        return {}

    parsed = None
    try:
        parsed = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        parsed = None

    tags = {}
    if isinstance(parsed, dict):
        for name, votes in parsed.items():
            key = str(name).strip().lower()
            if not key:
                continue
            try:
                tags[key] = int(votes)
            except (TypeError, ValueError):
                tags[key] = 1
    elif isinstance(parsed, (list, tuple, set)):
        # 투표수 없이 이름만 나열된 형태도 허용한다.
        for name in parsed:
            key = str(name).strip().lower()
            if key:
                tags[key] = 1
    else:
        for name, votes in _TAG_FALLBACK_RE.findall(text):
            key = name.strip().lower()
            if key:
                tags[key] = int(votes)
        if not tags:
            # "Action,Indie,RPG" 같은 단순 쉼표 구분 문자열.
            for name in text.split(","):
                key = name.strip().strip("'\"[]{} ").lower()
                if key:
                    tags[key] = 1
    return tags


def _clean_field(value):
    """NaN / None / 빈 값을 안전하게 빈 문자열로 만든다."""
    if value is None:
        return ''
    if isinstance(value, float) and math.isnan(value):
        return ''
    text = str(value).strip()
    return '' if text.lower() in ('nan', 'none') else text


def build_embedding_text(genres, tags, about):
    """임베딩에 넣을 입력 문자열 (embed.py와 services.py의 단일 기준).

    이 함수는 오프라인 임베딩 생성과 온라인 서비스가 '반드시' 똑같이
    써야 한다. 규칙이 어긋나면 벡터 공간이 달라져 추천이 조용히 망가진다.

    장르/태그를 앞에 두는 것은 의도적이다. all-MiniLM-L6-v2의
    max_seq_length는 256토큰이라 긴 설명문은 뒷부분이 잘려 나가는데,
    가장 변별력 있는 신호를 앞쪽에 배치해 잘림에 대비한다.
    """
    return (
        f'Genres: {_clean_field(genres)}. '
        f'Tags: {_clean_field(tags)}. '
        f'Description: {_clean_field(about)}'
    )


def tag_set(raw):
    """태그 이름만 담긴 집합. 가중치 없는 자카드 계산용."""
    return set(parse_tags(raw).keys())


def jaccard(set1, set2):
    """고전적인 집합 자카드 유사도: |교집합| / |합집합|."""
    if not set1 or not set2:
        return 0.0
    union = len(set1 | set2)
    if union == 0:
        return 0.0
    return len(set1 & set2) / union


def _normalize_votes(tags):
    """게임마다 투표 규모가 달라(인기작은 수만 표) 그대로 비교할 수 없다.
    각 게임 안에서 최대 투표수로 나눠 0~1로 맞춘 뒤 비교한다."""
    if not tags:
        return {}
    peak = max(tags.values())
    if peak <= 0:
        return {k: 1.0 for k in tags}
    return {k: v / peak for k, v in tags.items()}


def weighted_jaccard(tags1, tags2):
    """투표수를 반영한 자카드 유사도: sum(min) / sum(max).

    'Indie'처럼 아무 게임에나 붙는 약한 태그보다, 해당 게임을 실제로 규정하는
    상위 태그가 더 크게 기여한다. 두 인자는 parse_tags()의 결과(dict)여야 한다.
    """
    if not tags1 or not tags2:
        return 0.0
    a = _normalize_votes(tags1)
    b = _normalize_votes(tags2)
    keys = set(a) | set(b)
    num = sum(min(a.get(k, 0.0), b.get(k, 0.0)) for k in keys)
    den = sum(max(a.get(k, 0.0), b.get(k, 0.0)) for k in keys)
    if den <= 0:
        return 0.0
    return num / den


def tag_similarity(tags1, tags2, weighted=True):
    """서비스와 평가가 함께 쓰는 태그 유사도 진입점."""
    if weighted:
        return weighted_jaccard(tags1, tags2)
    return jaccard(set(tags1.keys()), set(tags2.keys()))


def hybrid_score(bert_score, tag_score, meta_score=0.0, w_tag=3.0, w_meta=0.0):
    """최종 정렬 점수.

    주의: 곱셈 구조라 결과가 1.0을 넘을 수 있다. 이 값은 '정렬 전용'이며,
    사용자에게 보여줄 때는 display_percent()를 쓴다.
    """
    return bert_score * (1.0 + tag_score * w_tag + meta_score * w_meta)


def display_percent(bert_score):
    """사용자에게 보여줄 0~100 매칭도.

    정렬용 final_score는 가중치 때문에 최대 4배까지 부풀어 "2.34점" 같은
    해석 불가능한 값이 화면에 나갔다. 표시용으로는 범위가 보장된
    코사인 유사도만 쓴다.
    """
    pct = (float(bert_score) if bert_score == bert_score else 0.0) * 100.0
    return int(round(max(0.0, min(100.0, pct))))


def l2_normalize(matrix):
    """행 단위 L2 정규화.

    미리 정규화해 두면 코사인 유사도가 단순 내적이 되어,
    cosine_similarity() 호출보다 빠르고 메모리 사본도 덜 만든다.
    """
    mat = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


def cosine_scores(query_vec, matrix):
    """정규화된 행렬에 대해 쿼리 벡터 하나의 코사인 유사도를 구한다."""
    q = np.asarray(query_vec, dtype=np.float32).ravel()
    norm = np.linalg.norm(q)
    if norm > 0:
        q = q / norm
    return matrix @ q


def top_k_indices(scores, k, exclude=None):
    """상위 k개 인덱스를 점수 내림차순으로 반환한다.

    기존 코드는 argsort()로 9만 개를 전부 정렬(O(N log N))한 뒤 뒤에서
    1000개만 잘라 썼다. argpartition은 'k번째 경계'만 찾으므로 O(N)이고,
    그 k개만 정렬한다.
    """
    scores = np.asarray(scores)
    n = scores.shape[0]
    if n == 0 or k <= 0:
        return np.array([], dtype=int)

    exclude = set() if exclude is None else set(exclude)
    # 제외 대상까지 고려해 여유분을 확보한다.
    fetch = min(n, k + len(exclude))

    if fetch >= n:
        cand = np.argsort(scores)[::-1]
    else:
        part = np.argpartition(scores, -fetch)[-fetch:]
        cand = part[np.argsort(scores[part])[::-1]]

    if not exclude:
        return cand[:k]
    return np.array([i for i in cand if int(i) not in exclude][:k], dtype=int)


# -----------------------------------------------------------------------------
# 랭킹 품질 지표
# -----------------------------------------------------------------------------
def precision_at_k(relevances, k):
    """상위 k개 중 정답 비율. 순위 내 위치는 구분하지 않는다."""
    top = list(relevances)[:k]
    if not top:
        return 0.0
    return sum(1 for r in top if r > 0) / float(k)


def dcg_at_k(relevances, k):
    return sum(
        (rel / math.log2(rank + 2))
        for rank, rel in enumerate(list(relevances)[:k])
    )


def ndcg_at_k(relevances, k):
    """순위를 반영한 지표. Precision@10은 1등과 10등을 동등하게 보지만
    nDCG는 위쪽에 정답이 올수록 높은 점수를 준다."""
    actual = dcg_at_k(relevances, k)
    ideal = dcg_at_k(sorted(relevances, reverse=True), k)
    if ideal <= 0:
        return 0.0
    return actual / ideal


def intra_list_diversity(vectors):
    """추천 목록 안의 아이템끼리 얼마나 서로 다른가 (1 - 평균 코사인 유사도).

    정확도만 최적화하면 비슷한 게임만 나열되는 필터버블이 생긴다.
    정확도 지표와 반드시 함께 봐야 한다.
    """
    mat = np.asarray(vectors, dtype=np.float32)
    if mat.ndim != 2 or mat.shape[0] < 2:
        return 0.0
    mat = l2_normalize(mat)
    sims = mat @ mat.T
    n = mat.shape[0]
    upper = sims[np.triu_indices(n, k=1)]
    if upper.size == 0:
        return 0.0
    return float(1.0 - upper.mean())
