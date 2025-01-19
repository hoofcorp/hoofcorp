import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
import plotly.graph_objects as go
from st_aggrid import AgGrid

# 페이지 설정
st.set_page_config(
    page_title="다중 모델 기반 매출 예측 대시보드",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 제목 및 설명
st.title("📊 다중 모델 기반 매출 예측 대시보드")
st.markdown("""
    이 대시보드는 매출 데이터를 필터링하고 다양한 예측 모델을 비교할 수 있도록 설계되었습니다.
""")

# 데이터 업로드
uploaded_file = st.file_uploader("데이터 파일을 업로드하세요 (Excel)", type=["xlsx", "xls"])
if uploaded_file:
    # 데이터 로드 함수
    @st.cache_data
    def load_data(file):
        df = pd.read_excel(file)
        df["진행 날짜"] = pd.to_datetime(df["진행 날짜"], format='%Y%m%d')  # 날짜 형식 변환
        return df

    df = load_data(uploaded_file)

    # 필터링 UI
    with st.sidebar:
        st.header("필터 설정")
        grades = st.multiselect("행사 등급 선택", options=df["행사등급"].dropna().unique().tolist())
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
            "날짜 범위 선택",
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

    # 주요 지표 출력
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="총 매출", value=f"{filtered_data['매출'].sum():,}원")
    with col2:
        st.metric(label="평균 판매가", value=f"{filtered_data['판매가'].mean():,.2f}원")
    with col3:
        st.metric(label="데이터 개수", value=len(filtered_data))

    # 데이터 테이블
    st.subheader("📋 필터링된 데이터")
    AgGrid(filtered_data, height=300, theme="streamlit")

    # 매출 예측 준비
    st.subheader("🔮 다중 모델 기반 매출 예측")
    filtered_data["월"] = filtered_data["진행 날짜"].dt.to_period("M")
    monthly_sales = filtered_data.groupby("월")["매출"].sum().reset_index()
    monthly_sales["월"] = monthly_sales["월"].dt.to_timestamp()

    if not monthly_sales.empty:
        periods_to_forecast = st.slider("예측할 개월 수", 1, 24, 12)

        results = {}

        # 1. Prophet 모델 예측
        if len(monthly_sales) >= 2:  # 최소 2개 이상의 데이터가 있어야 Prophet 실행 가능
            df_prophet = monthly_sales.rename(columns={"월": "ds", "매출": "y"})
            model_prophet = Prophet()
            model_prophet.fit(df_prophet)

            future = model_prophet.make_future_dataframe(periods=periods_to_forecast, freq="MS")
            forecast_prophet = model_prophet.predict(future)
            results["Prophet"] = forecast_prophet[["ds", "yhat"]]

            # Prophet 결과 시각화
            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(x=monthly_sales["월"], y=monthly_sales["매출"], mode="lines", name="실제 매출"))
            fig1.add_trace(go.Scatter(x=forecast_prophet["ds"], y=forecast_prophet["yhat"], mode="lines", name="Prophet 예측"))
            fig1.update_layout(title="Prophet 예측 결과", xaxis_title="날짜", yaxis_title="매출")
            st.plotly_chart(fig1, use_container_width=True)
        else:
            st.warning("📉 Prophet 모델을 실행하기에 데이터가 충분하지 않습니다. 필터를 변경하거나 데이터를 확인하세요.")

        # 2. Holt-Winters 모델 예측
        model_hw = ExponentialSmoothing(monthly_sales["매출"], seasonal="add", seasonal_periods=12, trend="add")
        model_hw_fit = model_hw.fit()
        forecast_hw = model_hw_fit.forecast(periods_to_forecast)
        forecast_dates_hw = pd.date_range(start=monthly_sales["월"].iloc[-1] + pd.offsets.MonthBegin(), periods=periods_to_forecast, freq="MS")
        results["Holt-Winters"] = pd.DataFrame({"날짜": forecast_dates_hw, "예측": forecast_hw.values})

        # Holt-Winters 결과 시각화
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=monthly_sales["월"], y=monthly_sales["매출"], mode="lines", name="실제 매출"))
        fig2.add_trace(go.Scatter(x=forecast_dates_hw, y=forecast_hw, mode="lines", name="Holt-Winters 예측"))
        fig2.update_layout(title="Holt-Winters 예측 결과", xaxis_title="날짜", yaxis_title="매출")
        st.plotly_chart(fig2, use_container_width=True)

        # 3. ARIMA 모델 예측
        model_arima = ARIMA(monthly_sales["매출"], order=(5, 1, 0))
        model_arima_fit = model_arima.fit()
        forecast_arima = model_arima_fit.forecast(steps=periods_to_forecast)
        forecast_dates_arima = pd.date_range(start=monthly_sales["월"].iloc[-1] + pd.offsets.MonthBegin(), periods=periods_to_forecast, freq="MS")
        results["ARIMA"] = pd.DataFrame({"날짜": forecast_dates_arima, "예측": forecast_arima.values})

        # ARIMA 결과 시각화
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=monthly_sales["월"], y=monthly_sales["매출"], mode="lines", name="실제 매출"))
        fig3.add_trace(go.Scatter(x=forecast_dates_arima, y=forecast_arima, mode="lines", name="ARIMA 예측"))
        fig3.update_layout(title="ARIMA 예측 결과", xaxis_title="날짜", yaxis_title="매출")
        st.plotly_chart(fig3, use_container_width=True)

        # 모델 비교 테이블
        comparison = pd.DataFrame({
            "날짜": forecast_dates_hw,
            "Prophet": forecast_prophet["yhat"].iloc[-periods_to_forecast:].values if "Prophet" in results else None,
            "Holt-Winters": forecast_hw.values,
            "ARIMA": forecast_arima.values
        })
        st.write("📊 모델 비교 결과")
        st.write(comparison)

        # 다운로드 버튼
        st.download_button(
            label="모델 비교 결과 다운로드",
            data=comparison.to_csv(index=False).encode("utf-8"),
            file_name="model_comparison.csv",
            mime="text/csv"
        )
    else:
        st.warning("데이터가 부족하여 예측을 수행할 수 없습니다.")
else:
    st.info("엑셀 파일을 업로드하세요.")
