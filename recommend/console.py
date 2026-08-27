"""CLI 스크립트용 콘솔 출력 헬퍼.

윈도우 기본 콘솔 코드페이지는 cp949(한국어 환경)라, 스크립트가 출력하는
이모지(📂, ✅ 등)에서 UnicodeEncodeError가 나며 통째로 죽는다.
한글은 cp949에 있어서 괜찮지만 이모지는 없다.

    UnicodeEncodeError: 'cp949' codec can't encode character '\\U0001f4c2'

IDE 터미널(UTF-8)에서는 재현되지 않아 놓치기 쉬운 버그다.
"""

import sys


def enable_utf8_output():
    """표준 출력/에러를 UTF-8로 재설정한다.

    콘솔이 UTF-8을 실제로 렌더링하지 못하더라도 errors='replace' 덕분에
    최소한 프로그램이 죽지는 않는다.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, 'reconfigure', None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding='utf-8', errors='replace')
        except (ValueError, OSError):
            # 파이프로 리다이렉트된 경우 등. 무시하고 진행한다.
            pass
