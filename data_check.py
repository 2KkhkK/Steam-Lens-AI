import pandas as pd
import io
import os

# 1. 파일 경로 설정
input_file = 'games.csv'
output_file = 'cleaned_games.csv'

print("📂 파일을 바이너리 모드로 읽어 디코딩을 시도합니다...")

# 2. 최후의 방법: 에러 무시하고 읽기 (Method 3)
if not os.path.exists(input_file):
    print(f"❌ 에러: {input_file} 파일이 없습니다. 파일명을 확인해 주세요.")
else:
    with open(input_file, 'rb') as f:
        # utf-8로 읽되, 에러 나는 바이트는 무시(ignore)합니다.
        content = f.read().decode('utf-8', errors='ignore')

    # 3. 판다스로 변환 (칸 수가 안 맞는 줄은 skip)
    # 
    df = pd.read_csv(io.StringIO(content), on_bad_lines='skip', low_memory=False)
    print(f"🧐 총 {len(df)}개의 행을 성공적으로 불러왔습니다.")

    # 4. 우리가 필요한 컬럼만 추출 (이미 분석한 번호 기반)
    # 컬럼명이 다를 수 있으니 인덱스(번호)로 접근하는 게 가장 안전합니다.
    try:
        clean_df = pd.DataFrame()
        clean_df['Name'] = df.iloc[:, 1]               # name
        clean_df['About_the_game'] = df.iloc[:, 7]    # about_the_game
        clean_df['Price'] = df.iloc[:, 4]              # price
        clean_df['Metacritic_score'] = df.iloc[:, 17] # metacritic_score
        clean_df['Genres'] = df.iloc[:, 28]            # genres
        clean_df['Image_URL'] = df.iloc[:, 10]         # header_image

        # 5. 데이터 '진짜' 정제 (필터링)
        # - 설명(About_the_game)이 비어있으면 AI 학습 불가 -> 삭제
        # - 메타크리틱 점수가 숫자가 아닌 것들(True, False 등) -> 0으로 처리
        print("🧹 데이터 필터링 및 타입 교정 중...")
        
        clean_df = clean_df.dropna(subset=['Name', 'About_the_game'])
        clean_df['Metacritic_score'] = pd.to_numeric(clean_df['Metacritic_score'], errors='coerce').fillna(0)
        
        # 6. 저장 (한글 깨짐 방지를 위해 utf-8-sig 사용)
        clean_df.to_csv(output_file, index=False, encoding='utf-8-sig')
        
        print("-" * 30)
        print(f"✅ 수술 성공! 깨끗한 데이터가 생성되었습니다.")
        print(f"💾 저장된 파일명: {output_file}")
        print(f"📊 최종 데이터 개수: {len(clean_df)}개")
        print("-" * 30)
        print("💡 상위 5개 데이터 미리보기:")
        print(clean_df[['Name', 'Metacritic_score']].head())

    except Exception as e:
        print(f"❌ 데이터 추출 중 에러 발생: {e}")
        print("팁: df.columns를 찍어서 컬럼 개수가 맞는지 확인해 보세요.")