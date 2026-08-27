"""원본 스팀 데이터셋을 추천 엔진이 쓸 형태로 정제한다.

    python data_check.py                 games.csv -> cleaned_games.csv
    python data_check.py --inspect       원본 컬럼 목록만 확인
    python data_check.py --sample 300    데모용 소형 샘플까지 함께 생성

이전 버전의 문제
  - 컬럼을 df.iloc[:, 42] 처럼 '번호'로 뽑았다. 데이터셋 버전이 바뀌면
    에러도 없이 엉뚱한 컬럼이 들어온다(조용한 실패).
  - errors='ignore' / on_bad_lines='skip' / dropna()로 데이터가 사라지는데
    몇 건이 사라졌는지 기록이 없었다.
지금은 컬럼을 '이름'으로 찾고, 단계별 손실 건수를 전부 출력한다.
"""

import argparse
import io
import os
import sys

import pandas as pd

from recommend.console import enable_utf8_output

# 윈도우 cp949 콘솔에서 이모지 출력 시 죽는 것을 막는다.
enable_utf8_output()

DEFAULT_INPUT = 'games.csv'
DEFAULT_OUTPUT = 'cleaned_games.csv'
SAMPLE_OUTPUT = 'sample_games.csv'

# 출력 컬럼명 -> 원본에서 찾아볼 후보 이름들(소문자 비교).
# 데이터셋 버전에 따라 이름이 조금씩 달라 여러 후보를 둔다.
COLUMN_MAP = {
    'Name':             ['name', 'title', 'game_name'],
    'About_the_game':   ['about_the_game', 'about the game', 'detailed_description', 'short_description'],
    'Price':            ['price', 'final_price', 'initial_price'],
    'Metacritic_score': ['metacritic_score', 'metacritic score', 'metacritic'],
    'Genres':           ['genres', 'genre'],
    'Tags':             ['tags', 'steamspy_tags', 'popular_tags'],
    'Image_URL':        ['header_image', 'header image', 'image', 'image_url'],
}

REQUIRED = ['Name', 'About_the_game']


def read_raw(path):
    """인코딩이 깨진 원본도 읽어 낸다. 몇 줄이 버려졌는지 함께 알려 준다."""
    if not os.path.exists(path):
        sys.exit(
            f'❌ {path} 파일이 없습니다.\n'
            '   README의 "데이터 준비" 절을 참고해 원본 데이터셋을 먼저 내려받으세요.'
        )

    with open(path, 'rb') as fp:
        raw = fp.read()

    for encoding in ('utf-8', 'utf-8-sig', 'cp949', 'latin-1'):
        try:
            text = raw.decode(encoding)
            print(f'📂 {encoding} 로 디코딩 성공')
            break
        except UnicodeDecodeError:
            continue
    else:
        text = raw.decode('utf-8', errors='ignore')
        print('⚠️  모든 인코딩 시도 실패 → utf-8 + errors="ignore"로 강제 디코딩')

    physical_lines = text.count('\n')

    df = pd.read_csv(io.StringIO(text), on_bad_lines='skip', low_memory=False)

    # 필드 수가 안 맞아 통째로 버려진 줄이 몇 개인지 대략 추정한다.
    skipped = max(0, physical_lines - 1 - len(df))
    print(f'🧐 {len(df):,}행 로드 완료 (파싱 실패로 건너뛴 줄 약 {skipped:,}개)')
    return df


def resolve_columns(df):
    """이름으로 컬럼을 찾는다. 못 찾은 것은 None으로 남긴다."""
    lookup = {str(c).strip().lower(): c for c in df.columns}
    resolved = {}

    for target, candidates in COLUMN_MAP.items():
        found = None
        for cand in candidates:
            if cand in lookup:
                found = lookup[cand]
                break
        resolved[target] = found

    return resolved


def build_clean_frame(df):
    resolved = resolve_columns(df)

    print('\n--- 컬럼 매핑 결과 ---')
    for target, source in resolved.items():
        mark = '✅' if source else ('❌' if target in REQUIRED else '⚠️ ')
        print(f'  {mark} {target:<18} <- {source or "(찾지 못함)"}')

    missing_required = [t for t in REQUIRED if not resolved[t]]
    if missing_required:
        print('\n사용 가능한 컬럼 목록:')
        print('  ' + ', '.join(str(c) for c in df.columns))
        sys.exit(
            f'\n❌ 필수 컬럼을 찾지 못했습니다: {", ".join(missing_required)}\n'
            '   data_check.py의 COLUMN_MAP에 이 데이터셋의 컬럼명을 추가해 주세요.'
        )

    clean = pd.DataFrame()
    for target, source in resolved.items():
        clean[target] = df[source] if source else ''

    print(f'\n🧹 정제 시작 (입력 {len(clean):,}행)')
    before = len(clean)

    # 1) 이름/설명문이 없는 행은 임베딩을 만들 수 없다.
    clean = clean.dropna(subset=['Name', 'About_the_game'])
    print(f'   - 이름/설명 결측 제거: {before - len(clean):,}행 삭제 → {len(clean):,}행')

    # 2) 설명문이 사실상 비어 있는 행 제거
    before = len(clean)
    clean = clean[clean['About_the_game'].astype(str).str.strip().str.len() >= 20]
    print(f'   - 설명문 20자 미만 제거: {before - len(clean):,}행 삭제 → {len(clean):,}행')

    # 3) 이름 중복 제거 (같은 게임이 여러 번 등록된 경우)
    before = len(clean)
    clean = clean.drop_duplicates(subset=['Name'], keep='first')
    print(f'   - 이름 중복 제거: {before - len(clean):,}행 삭제 → {len(clean):,}행')

    # 4) 숫자 컬럼 타입 교정
    clean['Metacritic_score'] = pd.to_numeric(
        clean['Metacritic_score'], errors='coerce'
    ).fillna(0).astype(int)
    clean['Price'] = pd.to_numeric(clean['Price'], errors='coerce').fillna(0.0)

    # 5) App_ID를 미리 뽑아 둔다. 서비스에서 매번 정규식을 돌리지 않도록.
    clean['App_ID'] = (
        clean['Image_URL'].astype(str).str.extract(r'/apps/(\d+)/')[0].fillna('')
    )
    no_appid = int((clean['App_ID'] == '').sum())
    if no_appid:
        print(f'   - App_ID 추출 실패: {no_appid:,}행 (보유 게임 필터에서 제외됨)')

    empty_tags = int(clean['Tags'].astype(str).str.strip().isin(['', 'nan']).sum())
    if empty_tags:
        print(f'   - 태그 없음: {empty_tags:,}행 (하이브리드 리랭킹 효과 없음)')

    return clean.reset_index(drop=True)


def write_sample(clean, path, size):
    """데모용 소형 데이터셋.

    저장소에 전체 데이터를 넣을 수는 없지만, 이게 있으면 클론한 사람이
    곧바로 서버를 띄워 볼 수 있다. 메타크리틱 점수가 높은 순으로 뽑아
    이름이 익숙한 게임이 들어가게 한다.
    """
    ranked = clean.sort_values('Metacritic_score', ascending=False)
    sample = ranked.head(size).sort_values('Name').reset_index(drop=True)
    sample.to_csv(path, index=False, encoding='utf-8-sig')
    print(f'🎁 데모용 샘플 저장: {path} ({len(sample):,}행)')


def main():
    parser = argparse.ArgumentParser(description='스팀 데이터셋 정제')
    parser.add_argument('--input', default=DEFAULT_INPUT, help=f'원본 CSV (기본: {DEFAULT_INPUT})')
    parser.add_argument('--output', default=DEFAULT_OUTPUT, help=f'정제 결과 (기본: {DEFAULT_OUTPUT})')
    parser.add_argument('--sample', type=int, default=0, metavar='N',
                        help=f'상위 N개로 {SAMPLE_OUTPUT}도 함께 생성')
    parser.add_argument('--inspect', action='store_true', help='원본 컬럼 목록만 출력하고 종료')
    args = parser.parse_args()

    df = read_raw(args.input)

    if args.inspect:
        print('\n--- 원본 컬럼 목록 ---')
        for i, col in enumerate(df.columns):
            preview = str(df.iloc[0][col])[:60].replace('\n', ' ') if len(df) else ''
            print(f'  [{i:>3}] {col:<28} | {preview}')
        return

    clean = build_clean_frame(df)

    clean.to_csv(args.output, index=False, encoding='utf-8-sig')
    print(f'\n✅ 저장 완료: {args.output} ({len(clean):,}행)')

    if args.sample:
        write_sample(clean, SAMPLE_OUTPUT, args.sample)

    print('\n💡 다음 단계:  python embed.py')


if __name__ == '__main__':
    main()
