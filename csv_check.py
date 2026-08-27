"""원본 CSV의 컬럼 구성을 확인하는 보조 스크립트.

data_check.py --inspect 와 같은 일을 한다. 컬럼을 번호로 세던 시절의
잔재이지만, 새 데이터셋을 처음 받았을 때 가장 먼저 실행하게 되는
스크립트라 진입점으로 남겨 둔다.

    python csv_check.py [파일명]
"""

import subprocess
import sys

if __name__ == '__main__':
    target = sys.argv[1] if len(sys.argv) > 1 else 'games.csv'
    sys.exit(
        subprocess.call([sys.executable, 'data_check.py', '--inspect', '--input', target])
    )
