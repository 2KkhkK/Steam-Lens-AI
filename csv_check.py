import pandas as pd

# 새 데이터셋 파일명으로 수정하세요
df = pd.read_csv('games.csv') 

print("--- 🎯 데이터셋 컬럼 매핑 점검 ---")
# 어떤 게임이든 좋으니 첫 번째 행의 데이터를 번호와 함께 출력합니다.
sample = df.iloc[0]

for i, value in enumerate(sample):
    print(f"[{i}번 칸] 컬럼명: {df.columns[i]} | 내용: {value}")