"""정제된 게임 데이터를 Sentence-BERT 벡터로 변환한다.

    python embed.py                              cleaned_games.csv -> steam_embeddings.pkl
    python embed.py --input sample_games.csv     샘플 데이터로 빠르게 생성
    python embed.py --model paraphrase-multilingual-MiniLM-L12-v2
                                                 다국어 모델(한국어 검색 지원)

주의: 이 스크립트는 모델을 '학습'하지 않는다. 사전학습된 모델로
추론(encode)만 수행한다. 예전 로그 문구가 "AI 모델 로드 및 학습 시작"이어서
오해를 부르기 쉬웠다.
"""

import argparse
import os
import pickle
import sys
import time

import pandas as pd

from recommend.console import enable_utf8_output
from recommend.similarity import build_embedding_text

# 윈도우 cp949 콘솔에서 이모지 출력 시 죽는 것을 막는다.
enable_utf8_output()

DEFAULT_INPUT = 'cleaned_games.csv'
DEFAULT_OUTPUT = 'steam_embeddings.pkl'
DEFAULT_MODEL = 'all-MiniLM-L6-v2'


def main():
    parser = argparse.ArgumentParser(description='게임 설명문 임베딩 생성')
    parser.add_argument('--input', default=DEFAULT_INPUT)
    parser.add_argument('--output', default=DEFAULT_OUTPUT)
    parser.add_argument('--model', default=os.environ.get('EMBEDDING_MODEL', DEFAULT_MODEL),
                        help='sentence-transformers 모델 이름')
    parser.add_argument('--batch-size', type=int, default=32)
    args = parser.parse_args()

    if not os.path.exists(args.input):
        sys.exit(
            f'❌ {args.input} 파일이 없습니다.\n'
            '   먼저 `python data_check.py`를 실행해 정제 데이터를 만드세요.'
        )

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        sys.exit(
            '❌ sentence-transformers가 설치되어 있지 않습니다.\n'
            '   pip install -r requirements.txt'
        )

    print(f'📂 데이터 로딩: {args.input}')
    df = pd.read_csv(args.input, encoding='utf-8-sig')

    for column in ('Name', 'About_the_game'):
        if column not in df.columns:
            sys.exit(f'❌ 필수 컬럼 {column}이(가) 없습니다. data_check.py를 다시 실행하세요.')

    print(f'🧩 장르 + 태그 + 설명문을 하나의 입력 문자열로 결합 ({len(df):,}건)')
    # 문자열 조립 규칙은 recommend/similarity.py 한 곳에만 있다.
    # 서비스와 규칙이 어긋나면 벡터 공간이 달라져 추천이 조용히 망가진다.
    texts = [
        build_embedding_text(row.get('Genres'), row.get('Tags'), row.get('About_the_game'))
        for _, row in df.iterrows()
    ]

    print(f'🧠 사전학습 모델 로드: {args.model}  (학습이 아니라 추론입니다)')
    model = SentenceTransformer(args.model)

    max_len = getattr(model, 'max_seq_length', None)
    if max_len:
        print(f'   ℹ️  max_seq_length = {max_len} 토큰. '
              f'이보다 긴 설명문은 뒤쪽이 잘립니다.')
        print(f'      그래서 장르/태그를 문자열 앞쪽에 배치합니다.')

    start = time.time()
    embeddings = model.encode(texts, batch_size=args.batch_size, show_progress_bar=True)
    elapsed = time.time() - start

    print(f'💾 저장: {args.output}')
    with open(args.output, 'wb') as fp:
        pickle.dump({
            'names': df['Name'].astype(str).tolist(),
            'embeddings': embeddings,
            # 메타데이터를 함께 저장한다. 나중에 "이 pkl이 어떤 모델로
            # 만들어졌는지" 알 수 없어 재현이 막히는 일을 방지한다.
            'meta': {
                'model': args.model,
                'dim': int(embeddings.shape[1]),
                'rows': int(embeddings.shape[0]),
                'source_csv': os.path.basename(args.input),
                'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
            },
        }, fp)

    print('-' * 50)
    print(f'✅ 임베딩 완료: {embeddings.shape[0]:,}건 × {embeddings.shape[1]}차원')
    print(f'   소요 시간: {elapsed / 60:.2f}분')
    print('-' * 50)
    print('💡 다음 단계:  python bulk_evaluate.py   또는   python manage.py runserver')


if __name__ == '__main__':
    main()
