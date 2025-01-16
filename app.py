import streamlit as st
import pandas as pd

# Streamlit 앱 헤더
st.title("사용자 파일 업로드 기반 데이터 처리 앱")

# 파일 업로드 UI
uploaded_file = st.file_uploader("엑셀 파일을 업로드하세요", type=["xlsx", "xls"])

if uploaded_file:
    # 데이터 로드
    @st.cache_data
    def load_data(file):
        df = pd.read_excel(file)
        df["진행 날짜"] = pd.to_datetime(df["진행 날짜"], format='%Y%m%d')  # 날짜 형식 변환
        df["행사등급"] = df["행사등급"].str.strip().str.upper()  # 공백 제거 및 대소문자 표준화
        df["운영몰"] = df["운영몰"].str.strip().str.upper()    # 공백 제거 및 대소문자 표준화
        return df

    # 업로드된 파일을 읽어 데이터프레임 생성
    try:
        df = load_data(uploaded_file)
        st.success("파일이 정상적으로 로드되었습니다!")
    except Exception as e:
        st.error(f"파일을 처리하는 데 실패했습니다: {e}")
        st.stop()

    # 필터링 조건 UI 생성
    st.header("검색 조건")

    # 행사등급 조건
    grades = st.multiselect("행사등급 선택", options=df["행사등급"].dropna().unique().tolist())

    # 운영몰 조건
    malls = st.multiselect("운영몰 선택", options=df["운영몰"].dropna().unique().tolist())

    # 브랜드명 조건
    brands = st.multiselect("브랜드명 선택", options=df["브랜드명"].dropna().unique().tolist())

    # 카테고리 조건
    categories = st.multiselect("카테고리 선택", options=df["카테고리"].dropna().unique().tolist())

    # 세분류 조건
    sub_categories = st.multiselect("세분류 선택", options=df["세분류"].dropna().unique().tolist())

    # 판매가 범위
    min_price, max_price = st.slider(
        "판매가 범위",
        min_value=int(df["판매가"].min()),
        max_value=int(df["판매가"].max()),
        value=(int(df["판매가"].min()), int(df["판매가"].max()))
    )

    # 매출 범위
    min_sales, max_sales = st.slider(
        "매출 범위",
        min_value=int(df["매출"].min()),
        max_value=int(df["매출"].max()),
        value=(int(df["매출"].min()), int(df["매출"].max()))
    )

    # 진행 날짜 범위
    start_date, end_date = st.date_input(
        "진행 날짜 범위",
        value=[df["진행 날짜"].min(), df["진행 날짜"].max()],
        min_value=df["진행 날짜"].min(),
        max_value=df["진행 날짜"].max()
    )

    # 필터링
    filtered_data = df.copy()

    if grades:
        filtered_data = filtered_data[filtered_data["행사등급"].isin(grades)]
    if malls:
        filtered_data = filtered_data[filtered_data["운영몰"].isin(malls)]
    if brands:
        filtered_data = filtered_data[filtered_data["브랜드명"].isin(brands)]
    if categories:
        filtered_data = filtered_data[filtered_data["카테고리"].isin(categories)]
    if sub_categories:
        filtered_data = filtered_data[filtered_data["세분류"].isin(sub_categories)]

    # 가격 및 매출 범위 필터
    filtered_data = filtered_data[
        (filtered_data["판매가"] >= min_price) & 
        (filtered_data["판매가"] <= max_price) &
        (filtered_data["매출"] >= min_sales) & 
        (filtered_data["매출"] <= max_sales)
    ]

    # 날짜 범위 필터
    filtered_data = filtered_data[
        (filtered_data["진행 날짜"] >= pd.Timestamp(start_date)) &
        (filtered_data["진행 날짜"] <= pd.Timestamp(end_date))
    ]

    # 결과 출력
    st.write(f"총 결과: {len(filtered_data)}개")
    if len(filtered_data) > 0:
        st.dataframe(filtered_data)
    else:
        st.warning("조건에 맞는 데이터가 없습니다.")

    # 결과 저장 버튼
    if st.button("결과 저장"):
        if len(filtered_data) > 0:
            output_path = "filtered_results.xlsx"
            filtered_data.to_excel(output_path, index=False)
            st.success(f"결과가 저장되었습니다: {output_path}")
        else:
            st.warning("저장할 데이터가 없습니다.")
else:
    st.info("엑셀 파일을 업로드해주세요.")
