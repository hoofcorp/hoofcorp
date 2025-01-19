import streamlit as st
import pandas as pd
import io
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import matplotlib.pyplot as plt

# Streamlit 앱 헤더
st.title("매출 예측 기반 데이터 처리 앱")

# 파일 업로드 UI
uploaded_file = st.file_uploader("엑셀 파일을 업로드하세요", type=["xlsx", "xls"])

if uploaded_file:
    # 데이터 로드 함수
    @st.cache_data
    def load_data(file):
        df = pd.read_excel(file)
        df["진행 날짜"] = pd.to_datetime(df["진행 날짜"], format='%Y%m%d')  # 날짜 형식 변환
        df["행사등급"] = df["행사등급"].str.strip().str.upper()  # 공백 제거 및 대소문자 표준화
        df["운영몰"] = df["운영몰"].str.strip().str.upper()    # 공백 제거 및 대소문자 표준화
        return df

    try:
        df = load_data(uploaded_file)
        st.success("파일이 정상적으로 로드되었습니다!")
    except Exception as e:
        st.error(f"파일을 처리하는 데 실패했습니다: {e}")
        st.stop()

    # 필터링 조건 UI
    st.header("검색 조건")
    grades = st.multiselect("행사등급 선택", options=df["행사등급"].dropna().unique().tolist())
    malls = st.multiselect("운영몰 선택", options=df["운영몰"].dropna().unique().tolist())
    brands = st.multiselect("브랜드명 선택", options=df["브랜드명"].dropna().unique().tolist())
    categories = st.multiselect("카테고리 선택", options=df["카테고리"].dropna().unique().tolist())
    sub_categories = st.multiselect("세분류 선택", options=df["세분류"].dropna().unique().tolist())
    min_price, max_price = st.slider(
        "판매가 범위",
        min_value=int(df["판매가"].min()),
        max_value=int(df["판매가"].max()),
        value=(int(df["판매가"].min()), int(df["판매가"].max()))
    )
    min_sales, max_sales = st.slider(
        "매출 범위",
        min_value=int(df["매출"].min()),
        max_value=int(df["매출"].max()),
        value=(int(df["매출"].min()), int(df["매출"].max()))
    )
    start_date, end_date = st.date_input(
        "진행 날짜 범위",
        value=[df["진행 날짜"].min(), df["진행 날짜"].max()],
        min_value=df["진행 날짜"].min(),
        max_value=df["진행 날짜"].max()
    )

    # 데이터 필터링
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
    filtered_data = filtered_data[
        (filtered_data["판매가"] >= min_price) & 
        (filtered_data["판매가"] <= max_price) & 
        (filtered_data["매출"] >= min_sales) & 
        (filtered_data["매출"] <= max_sales)
    ]
    filtered_data = filtered_data[
        (filtered_data["진행 날짜"] >= pd.Timestamp(start_date)) &
        (filtered_data["진행 날짜"] <= pd.Timestamp(end_date))
    ]

    # 결과 출력
    st.write(f"총 결과: {len(filtered_data)}개")
    if len(filtered_data) > 0:
        # 총매출과 평균 판매가 계산
        total_sales = filtered_data["매출"].sum()
        avg_price = filtered_data["판매가"].mean()

        st.write(f"**총매출**: {total_sales:,}원")
        st.write(f"**평균 판매가**: {avg_price:,.2f}원")
        st.dataframe(filtered_data)

        # 매출 예측
        st.header("매출 예측")
        periods_to_forecast = st.slider("예측할 개월 수", 1, 24, 12)

        # 월별 매출 데이터 준비
        filtered_data["월"] = filtered_data["진행 날짜"].dt.to_period("M")
        monthly_sales = filtered_data.groupby("월")["매출"].sum().reset_index()
        monthly_sales["월"] = monthly_sales["월"].dt.to_timestamp()

        # 데이터의 계절성 감지
        if len(monthly_sales) >= 24:  # 최소 2개의 계절 주기(24개월)가 있어야 계절성 모델 적용 가능
            seasonal_periods = 12  # 12개월(1년) 단위 계절성
            seasonal_type = "add"  # 계절성을 추가 방식으로 모델링
            st.write("**계절성이 감지되었습니다. 계절성을 반영한 모델을 사용합니다.**")
        else:
            seasonal_periods = None
            seasonal_type = None
            st.write("**데이터가 부족하여 계절성이 제거된 모델을 사용합니다.**")

        # 모델 생성
        try:
            model = ExponentialSmoothing(
                monthly_sales["매출"],
                trend="add",  # 트렌드를 추가 방식으로 모델링
                seasonal=seasonal_type,
                seasonal_periods=seasonal_periods
            )
            model_fit = model.fit()
            forecast = model_fit.forecast(periods_to_forecast)

            # 예측 결과 생성
            forecast_dates = pd.date_range(
                start=monthly_sales["월"].iloc[-1] + pd.offsets.MonthBegin(),
                periods=periods_to_forecast,
                freq="MS"
            )
            forecast_df = pd.DataFrame({
                "예측 날짜": forecast_dates,
                "예상 매출": forecast
            })

            # 결과 시각화
            plt.figure(figsize=(10, 6))
            plt.plot(monthly_sales["월"], monthly_sales["매출"], label="실제 매출")
            plt.plot(forecast_df["예측 날짜"], forecast_df["예상 매출"], label="예상 매출", linestyle="--")
            plt.legend()
            plt.xlabel("날짜")
            plt.ylabel("매출")
            plt.title("매출 예측")
            st.pyplot(plt)

            # 예측 결과 표시
            st.subheader("예측 데이터")
            st.dataframe(forecast_df)

            # 다운로드 버튼 추가
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                forecast_df.to_excel(writer, index=False)
            output.seek(0)

            st.download_button(
                label="예측 결과 다운로드",
                data=output,
                file_name="sales_forecast.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        except Exception as e:
            st.error(f"예측 중 오류가 발생했습니다: {e}")
    else:
        st.warning("조건에 맞는 데이터가 없습니다.")
else:
    st.info("엑셀 파일을 업로드해주세요.")
