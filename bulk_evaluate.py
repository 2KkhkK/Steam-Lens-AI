"""추천 성능 오프라인 평가.

    python bulk_evaluate.py                      기본(holdout) 프로토콜로 평가
    python bulk_evaluate.py --protocol all       세 프로토콜 모두 실행
    python bulk_evaluate.py --grid               가중치 그리드 서치
    python bulk_evaluate.py --make-template 40   골든셋 라벨링 템플릿 생성

────────────────────────────────────────────────────────────────────────────
왜 다시 썼는가 — 기존 평가의 치명적 결함
────────────────────────────────────────────────────────────────────────────
이전 버전은 이런 구조였다.

    정답(Hit) 기준 :  태그 자카드 유사도 >= 0.15
    하이브리드 정렬 :  bert_score * (1 + 태그 자카드 * 3.0)

즉 **태그가 겹치도록 정렬한 다음, 태그가 겹치는지로 채점**했다.
하이브리드가 베이스라인을 이기는 것은 모델이 좋아서가 아니라
수학적으로 당연한 결과다. 채점 기준을 최적화 목표로 그대로 쓴
순환논법(circular reasoning)이며, 이 실험은 추천 품질이 좋아졌다는
증거가 되지 못한다.

────────────────────────────────────────────────────────────────────────────
해결: 정렬 신호와 채점 신호를 분리한 3가지 프로토콜
────────────────────────────────────────────────────────────────────────────
[holdout]  (기본, 자동)  신뢰도 ★★☆
    태그 어휘를 해시 기준으로 서로 겹치지 않는 두 묶음 A / B로 나눈다.
    정렬에는 A 태그만, 채점에는 B 태그만 쓴다. 두 묶음은 공유하는 태그가
    하나도 없으므로 태그 수준의 순환논법이 끊긴다.

[genre]    (자동)        신뢰도 ★☆☆
    채점을 Tags가 아닌 Genres 컬럼으로 한다. 정렬(Tags)과 채점(Genres)이
    서로 다른 컬럼이다. 다만 둘 다 임베딩 입력에 들어가 있어 완전히
    독립적이지는 않다. 참고 지표로만 본다.

[golden]   (사람이 라벨링) 신뢰도 ★★★
    eval_goldenset.csv에 사람이 직접 매긴 정답을 읽는다. 유일하게
    "추천이 실제로 좋아졌는가"를 말할 수 있는 프로토콜이다.
        python bulk_evaluate.py --make-template 40
    로 템플릿을 만들고 relevant 열에 0/1을 채워 넣으면 된다.

────────────────────────────────────────────────────────────────────────────
남아 있는 한계 (면접에서 먼저 말할 것)
────────────────────────────────────────────────────────────────────────────
임베딩 입력 문자열 자체가 "Genres: ... Tags: ... Description: ..."이라
장르/태그 정보가 이미 벡터 안에 녹아 있다. 따라서 holdout / genre
프로토콜도 완전한 독립 검증은 아니다. 다만 이 누수는 베이스라인과
하이브리드에 '똑같이' 적용되므로 두 방법의 비교는 공정하다.
완전한 독립 검증은 golden 프로토콜뿐이다.
"""

import argparse
import hashlib
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd

from recommend import similarity as sim

from recommend.console import enable_utf8_output

# 윈도우 cp949 콘솔에서 이모지 출력 시 죽는 것을 막는다.
enable_utf8_output()

CSV_CANDIDATES = ['cleaned_games.csv', 'sample_games.csv']
EMBED_PATH = 'steam_embeddings.pkl'
GOLDEN_PATH = 'eval_goldenset.csv'
TEMPLATE_PATH = 'eval_goldenset.template.csv'

K = 10
CANDIDATE_POOL = 1000


# -----------------------------------------------------------------------------
# 데이터 로딩
# -----------------------------------------------------------------------------
def load_data():
    csv_path = next((p for p in CSV_CANDIDATES if os.path.exists(p)), None)
    if csv_path is None:
        sys.exit(
            '❌ 게임 데이터가 없습니다.\n'
            '   python data_check.py 를 먼저 실행하세요.'
        )
    if not os.path.exists(EMBED_PATH):
        sys.exit(
            f'❌ {EMBED_PATH} 가 없습니다.\n'
            '   python embed.py 를 먼저 실행하세요.'
        )

    print(f'📂 데이터 로딩: {csv_path}')
    df = pd.read_csv(csv_path, encoding='utf-8-sig')

    with open(EMBED_PATH, 'rb') as fp:
        payload = pickle.load(fp)

    embeddings = np.asarray(payload['embeddings'], dtype=np.float32)
    if embeddings.shape[0] != len(df):
        sys.exit(
            f'❌ 임베딩 {embeddings.shape[0]:,}행 vs CSV {len(df):,}행 불일치.\n'
            '   data_check.py -> embed.py 순서로 다시 생성하세요.'
        )

    meta = payload.get('meta', {})
    if meta:
        print(f'   모델: {meta.get("model")} ({meta.get("dim")}차원)')

    # 미리 정규화해 두면 코사인 유사도가 단순 내적이 된다.
    embeddings = sim.l2_normalize(embeddings)

    print(f'🧩 태그 파싱 중... ({len(df):,}건)')
    tag_dicts = [sim.parse_tags(raw) for raw in df['Tags'].tolist()]

    genre_sets = [
        {g.strip().lower() for g in str(raw).split(',') if g.strip()}
        if pd.notna(raw) else set()
        for raw in df.get('Genres', pd.Series([None] * len(df))).tolist()
    ]

    return df, embeddings, tag_dicts, genre_sets


# -----------------------------------------------------------------------------
# 프로토콜별 '정렬용 태그'와 '채점용 라벨' 정의
# -----------------------------------------------------------------------------
def tag_bucket(tag):
    """태그 이름을 해시해 A(0) / B(1) 두 묶음으로 결정적으로 나눈다."""
    digest = hashlib.md5(tag.encode('utf-8')).hexdigest()
    return int(digest[0], 16) % 2


def split_tags(tags):
    """{태그: 투표수}를 (정렬용 A묶음, 채점용 B묶음)으로 분리."""
    rank_tags, label_tags = {}, {}
    for tag, votes in tags.items():
        (rank_tags if tag_bucket(tag) == 0 else label_tags)[tag] = votes
    return rank_tags, label_tags


class Protocol:
    """정렬 신호와 채점 신호를 어떻게 가를지 정의한다."""

    def __init__(self, name, df, tag_dicts, genre_sets, threshold):
        self.name = name
        self.df = df
        self.tag_dicts = tag_dicts
        self.genre_sets = genre_sets
        self.threshold = threshold

    def rank_tags(self, idx):
        """정렬에 쓸 태그."""
        if self.name == 'holdout':
            return split_tags(self.tag_dicts[idx])[0]
        return self.tag_dicts[idx]

    def candidate_rank_tags(self, idx):
        return self.rank_tags(idx)

    def is_relevant(self, target_idx, candidate_idx):
        """채점: 이 추천이 정답인가."""
        if self.name == 'holdout':
            a = split_tags(self.tag_dicts[target_idx])[1]
            b = split_tags(self.tag_dicts[candidate_idx])[1]
            return sim.jaccard(set(a), set(b)) >= self.threshold

        if self.name == 'genre':
            a = self.genre_sets[target_idx]
            b = self.genre_sets[candidate_idx]
            return sim.jaccard(a, b) >= self.threshold

        raise NotImplementedError(self.name)

    def is_evaluable(self, idx):
        """이 게임을 평가 대상으로 삼을 수 있는가."""
        if self.name == 'holdout':
            rank_part, label_part = split_tags(self.tag_dicts[idx])
            return bool(rank_part) and bool(label_part)
        if self.name == 'genre':
            return bool(self.genre_sets[idx]) and bool(self.tag_dicts[idx])
        return True


# -----------------------------------------------------------------------------
# 랭킹
# -----------------------------------------------------------------------------
def rank_once(embeddings, tag_lookup, target_idx, target_tags,
              w_tag, meta_scores=None, w_meta=0.0, pool=CANDIDATE_POOL, k=K):
    """한 게임에 대한 baseline / hybrid 상위 k개를 함께 계산한다.

    예전 스크립트는 baseline을 argsort()[-11:-1] 슬라이싱으로 뽑아
    '자기 자신이 반드시 1등'이라고 가정한 반면, hybrid만 명시적으로
    자기 자신을 제외했다. 비교 조건이 비대칭이었다.
    여기서는 양쪽 모두 같은 exclude 집합을 쓴다.
    """
    scores = embeddings @ embeddings[target_idx]
    exclude = {target_idx}

    baseline = [int(i) for i in sim.top_k_indices(scores, k, exclude=exclude)]

    pool_indices = sim.top_k_indices(scores, pool, exclude=exclude)
    scored = []
    for i in pool_indices:
        i = int(i)
        tag_score = sim.jaccard(set(target_tags), set(tag_lookup(i)))
        meta = (meta_scores[i] if meta_scores is not None else 0.0)
        scored.append((sim.hybrid_score(float(scores[i]), tag_score, meta, w_tag, w_meta), i))

    scored.sort(reverse=True)
    hybrid = [i for _, i in scored[:k]]

    return baseline, hybrid


# -----------------------------------------------------------------------------
# 지표 집계
# -----------------------------------------------------------------------------
class Accumulator:
    def __init__(self, label):
        self.label = label
        self.precision = []
        self.ndcg = []
        self.diversity = []
        self.metacritic = []
        self.recommended = set()

    def add(self, relevances, indices, embeddings, meta_raw):
        self.precision.append(sim.precision_at_k(relevances, K))
        self.ndcg.append(sim.ndcg_at_k(relevances, K))
        self.diversity.append(sim.intra_list_diversity(embeddings[indices]))
        scores = [meta_raw[i] for i in indices if meta_raw[i] > 0]
        if scores:
            self.metacritic.append(float(np.mean(scores)))
        self.recommended.update(indices)

    def summary(self, catalog_size):
        def avg(xs):
            return float(np.mean(xs)) if xs else 0.0
        return {
            'label': self.label,
            'precision': avg(self.precision) * 100,
            'ndcg': avg(self.ndcg) * 100,
            'diversity': avg(self.diversity),
            'metacritic': avg(self.metacritic),
            'coverage': len(self.recommended) / catalog_size * 100,
        }


def print_table(rows):
    header = f'{"방법":<22}{"P@10":>9}{"nDCG@10":>10}{"다양성":>9}{"평균메타":>10}{"커버리지":>10}'
    print(header)
    print('-' * 70)
    for r in rows:
        print(
            f'{r["label"]:<22}'
            f'{r["precision"]:>8.1f}%'
            f'{r["ndcg"]:>9.1f}%'
            f'{r["diversity"]:>9.3f}'
            f'{r["metacritic"]:>10.1f}'
            f'{r["coverage"]:>9.2f}%'
        )


# -----------------------------------------------------------------------------
# 자동 프로토콜 (holdout / genre)
# -----------------------------------------------------------------------------
def run_protocol(name, df, embeddings, tag_dicts, genre_sets,
                 num_samples, threshold, w_tag, seed=42, quiet=False):
    proto = Protocol(name, df, tag_dicts, genre_sets, threshold)

    rng = np.random.default_rng(seed)
    evaluable = [i for i in range(len(df)) if proto.is_evaluable(i)]
    if not evaluable:
        print(f'⚠️  [{name}] 평가 가능한 게임이 없습니다. 건너뜁니다.')
        return None

    size = min(num_samples, len(evaluable))
    samples = rng.choice(evaluable, size=size, replace=False)

    meta_raw = pd.to_numeric(df.get('Metacritic_score', 0), errors='coerce').fillna(0).to_numpy()

    base_acc = Accumulator('① BERT 단독 (baseline)')
    hyb_acc = Accumulator('② 하이브리드 (제안)')

    if not quiet:
        print(f'\n⏳ [{name}] {size}개 샘플 평가 중...')
    start = time.time()

    for count, target_idx in enumerate(samples, start=1):
        target_idx = int(target_idx)
        target_tags = proto.rank_tags(target_idx)

        baseline, hybrid = rank_once(
            embeddings,
            lambda i: proto.candidate_rank_tags(i),
            target_idx, target_tags, w_tag,
        )

        base_rel = [1 if proto.is_relevant(target_idx, i) else 0 for i in baseline]
        hyb_rel = [1 if proto.is_relevant(target_idx, i) else 0 for i in hybrid]

        base_acc.add(base_rel, baseline, embeddings, meta_raw)
        hyb_acc.add(hyb_rel, hybrid, embeddings, meta_raw)

        if count % 20 == 0 and not quiet:
            print(f'   {count}/{size} ...')

    elapsed = time.time() - start
    rows = [base_acc.summary(len(df)), hyb_acc.summary(len(df))]

    if quiet:
        return rows

    print(f'\n{"=" * 70}')
    print(f'📊 프로토콜: {name}   (샘플 {size}개, 정답 임계값 {threshold}, 태그 가중치 {w_tag})')
    print('=' * 70)

    print_table(rows)

    delta = rows[1]['precision'] - rows[0]['precision']
    print(f'\n   P@10 변화: {delta:+.1f}%p')
    if rows[1]['diversity'] < rows[0]['diversity']:
        print(f'   ⚠️  다양성은 오히려 감소했습니다 '
              f'({rows[0]["diversity"]:.3f} → {rows[1]["diversity"]:.3f}). '
              f'정확도와 다양성의 트레이드오프.')
    if rows[1]['metacritic'] > rows[0]['metacritic'] + 1:
        print(f'   ⚠️  추천된 게임의 평균 메타크리틱이 상승했습니다 '
              f'({rows[0]["metacritic"]:.1f} → {rows[1]["metacritic"]:.1f}). '
              f'인기 편향 가능성.')
    print(f'   소요 시간: {elapsed:.1f}초')

    return rows


# -----------------------------------------------------------------------------
# 골든셋 프로토콜
# -----------------------------------------------------------------------------
def run_golden(df, embeddings, tag_dicts, w_tag):
    if not os.path.exists(GOLDEN_PATH):
        print(f'\nℹ️  [golden] {GOLDEN_PATH} 가 없어 건너뜁니다.')
        print(f'   `python bulk_evaluate.py --make-template 40` 으로 템플릿을 만든 뒤')
        print(f'   relevant 열에 0/1을 채우고 {GOLDEN_PATH} 로 저장하세요.')
        print('   사람이 매긴 정답만이 순환논법에서 완전히 자유롭습니다.')
        return None

    golden = pd.read_csv(GOLDEN_PATH, encoding='utf-8-sig')
    required = {'query_game', 'candidate_game', 'relevant'}
    if not required.issubset(golden.columns):
        print(f'⚠️  [golden] 필요한 열이 없습니다: {required}')
        return None

    golden = golden[pd.to_numeric(golden['relevant'], errors='coerce').notna()]
    if golden.empty:
        print('⚠️  [golden] relevant 열이 전부 비어 있습니다. 라벨링이 필요합니다.')
        return None

    name_to_idx = {str(n).lower(): i for i, n in enumerate(df['Name'].tolist())}
    meta_raw = pd.to_numeric(df.get('Metacritic_score', 0), errors='coerce').fillna(0).to_numpy()

    relevant_map = {}
    for query, group in golden.groupby('query_game'):
        positives = {
            str(r['candidate_game']).lower()
            for _, r in group.iterrows()
            if float(r['relevant']) > 0
        }
        relevant_map[str(query).lower()] = positives

    base_acc = Accumulator('① BERT 단독 (baseline)')
    hyb_acc = Accumulator('② 하이브리드 (제안)')

    evaluated = 0
    base_recall, hyb_recall = [], []

    for query_lower, positives in relevant_map.items():
        idx = name_to_idx.get(query_lower)
        if idx is None or not positives:
            continue

        baseline, hybrid = rank_once(
            embeddings, lambda i: tag_dicts[i], idx, tag_dicts[idx], w_tag,
        )

        def relevances(indices):
            return [
                1 if str(df.iloc[i]['Name']).lower() in positives else 0
                for i in indices
            ]

        base_rel = relevances(baseline)
        hyb_rel = relevances(hybrid)

        base_acc.add(base_rel, baseline, embeddings, meta_raw)
        hyb_acc.add(hyb_rel, hybrid, embeddings, meta_raw)
        base_recall.append(sum(base_rel) / len(positives))
        hyb_recall.append(sum(hyb_rel) / len(positives))
        evaluated += 1

    if not evaluated:
        print('⚠️  [golden] 라벨의 query_game이 데이터셋에 하나도 없습니다.')
        return None

    print(f'\n{"=" * 70}')
    print(f'📊 프로토콜: golden   (사람이 라벨링한 쿼리 {evaluated}개)')
    print('=' * 70)
    rows = [base_acc.summary(len(df)), hyb_acc.summary(len(df))]
    print_table(rows)
    print(f'\n   Recall@10 — baseline {np.mean(base_recall) * 100:.1f}% / '
          f'hybrid {np.mean(hyb_recall) * 100:.1f}%')
    print('   ✅ 이 결과만이 정렬 신호와 무관한 독립 검증입니다.')
    return rows


def make_template(df, embeddings, tag_dicts, num_queries, w_tag, seed=42):
    """사람이 라벨링할 후보 쌍 목록을 만든다.

    baseline과 hybrid의 추천 결과를 합집합으로 모아 놓으므로,
    어느 한쪽에 유리하게 치우치지 않은 라벨링이 가능하다.
    """
    rng = np.random.default_rng(seed)
    pool = [i for i in range(len(df)) if tag_dicts[i]]
    size = min(num_queries, len(pool))
    samples = rng.choice(pool, size=size, replace=False)

    rows = []
    for target_idx in samples:
        target_idx = int(target_idx)
        baseline, hybrid = rank_once(
            embeddings, lambda i: tag_dicts[i], target_idx, tag_dicts[target_idx], w_tag,
        )
        # 두 방법의 결과를 섞어 순서를 감춘다(라벨링 편향 방지).
        union = sorted(set(baseline) | set(hybrid))
        for cand in union:
            rows.append({
                'query_game': df.iloc[target_idx]['Name'],
                'candidate_game': df.iloc[cand]['Name'],
                'relevant': '',   # <- 여기에 0 또는 1을 채운다
                'candidate_genres': df.iloc[cand].get('Genres', ''),
            })

    out = pd.DataFrame(rows)
    out.to_csv(TEMPLATE_PATH, index=False, encoding='utf-8-sig')

    print(f'✅ 라벨링 템플릿 생성: {TEMPLATE_PATH}')
    print(f'   쿼리 {size}개 / 후보 쌍 {len(out):,}건')
    print()
    print('   다음 순서로 진행하세요:')
    print(f'   1. {TEMPLATE_PATH} 를 열어 relevant 열에 0(관련없음) 또는 1(관련있음)을 채운다')
    print(f'   2. {GOLDEN_PATH} 이름으로 저장한다')
    print('   3. python bulk_evaluate.py --protocol golden')
    print()
    print('   💡 전부 채울 필요는 없습니다. 30~50쌍만 라벨링해도')
    print('      "독립적인 정답셋으로 검증했다"고 말할 수 있습니다.')


# -----------------------------------------------------------------------------
# 가중치 그리드 서치
# -----------------------------------------------------------------------------
def run_grid(df, embeddings, tag_dicts, genre_sets, num_samples, threshold, seed=42):
    """태그 가중치를 바꿔 가며 최적값을 찾는다.

    3.0이라는 숫자에 근거가 없다는 지적에 대한 답. holdout 프로토콜로
    측정하므로 채점 기준과 정렬 기준이 겹치지 않는다.
    """
    weights = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0]
    print(f'\n🔍 태그 가중치 그리드 서치 (holdout 프로토콜, 샘플 {num_samples}개)')
    print('=' * 70)
    print(f'{"w_tag":>8}{"P@10":>10}{"nDCG@10":>11}{"다양성":>10}')
    print('-' * 70)

    results = []
    for w in weights:
        rows = run_protocol('holdout', df, embeddings, tag_dicts, genre_sets,
                            num_samples, threshold, w, seed, quiet=True)
        if rows is None:
            continue
        hybrid = rows[1]
        results.append((w, hybrid))
        # w_tag=0 이면 사실상 baseline과 같다. 표의 기준점으로 삼는다.
        print(f'{w:>8.1f}{hybrid["precision"]:>9.1f}%'
              f'{hybrid["ndcg"]:>10.1f}%{hybrid["diversity"]:>10.3f}')

    if results:
        best = max(results, key=lambda x: x[1]['ndcg'])
        print(f'\n✅ nDCG@10 기준 최적 가중치: {best[0]}')
        print(f'   .env 의 SEARCH_TAG_WEIGHT 에 이 값을 적용하세요.')
        print('   ⚠️  단, 이 값은 holdout 프로토콜 기준입니다. 골든셋이 준비되면')
        print('      반드시 golden 프로토콜로 재확인하세요.')


# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description='추천 성능 오프라인 평가',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--protocol', default='holdout',
                        choices=['holdout', 'genre', 'golden', 'all'])
    parser.add_argument('--samples', type=int, default=100)
    parser.add_argument('--threshold', type=float, default=0.15)
    parser.add_argument('--w-tag', type=float, default=3.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--grid', action='store_true', help='가중치 그리드 서치')
    parser.add_argument('--make-template', type=int, metavar='N',
                        help='골든셋 라벨링 템플릿 생성 (쿼리 N개)')
    args = parser.parse_args()

    df, embeddings, tag_dicts, genre_sets = load_data()

    if args.make_template:
        make_template(df, embeddings, tag_dicts, args.make_template, args.w_tag, args.seed)
        return

    if args.grid:
        run_grid(df, embeddings, tag_dicts, genre_sets,
                 args.samples, args.threshold, args.seed)
        return

    protocols = ['holdout', 'genre', 'golden'] if args.protocol == 'all' else [args.protocol]

    for name in protocols:
        if name == 'golden':
            run_golden(df, embeddings, tag_dicts, args.w_tag)
        else:
            run_protocol(name, df, embeddings, tag_dicts, genre_sets,
                         args.samples, args.threshold, args.w_tag, args.seed)

    print('\n' + '─' * 70)
    print('※ holdout / genre 는 임베딩 입력에 장르·태그가 포함된 탓에')
    print('  완전한 독립 검증이 아닙니다. 다만 그 누수는 baseline과 hybrid에')
    print('  동일하게 적용되므로 두 방법의 "비교"는 공정합니다.')
    print('  최종 결론은 golden 프로토콜로 내려야 합니다.')


if __name__ == '__main__':
    main()
