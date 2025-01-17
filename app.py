import subprocess
import streamlit as st
import pandas as pd
import io

# 필요한 패키지 설치
subprocess.run(["pip", "install", "--upgrade", "pip", "setuptools", "wheel"])
subprocess.run(["pip", "install", "mxnet", "gluonts"])

from gluonts.dataset.common import ListDataset
from gluonts.model.deepar import DeepAREstimator
from gluonts.trainer import Trainer
from gluonts.evaluation.backtest import make_evaluation_predictions

# Streamlit 앱 헤더
st.title("사용자 파일 업로드 기반 데이터 처리 앱")

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

    # 업로드된 파일 읽기
    try:
        df = load_data(uploaded_file)
        st.success("파일이 정상적으로 로드되었습니다!")
    except Exception as e:
        st.error(f"파일을 처리하는 데 실패했습니다: {e}")
        st.stop()

    # 필터링 조건 UI
    st.header("검색 조건")
    grades = st.multiselect("행사등급 선택", options=df["행사등급"].drop나().unique().tolist())
    malls = st.multiselect("운영몰 선택", options=df["운영몰"].drop나().unique().tolist())
    brands = st.multiselect("브랜드명 선택", options=df["브랜드명"].drop나().unique().tolist())
    categories = st.multiselect("카테고리 선택", options=df["카테고리"].drop나().unique().tolist())
    sub_categories = st.multiselect("세분류 선택", options=df["세분류"].drop나().unique().tolist())
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

    # 예측을 위한 데이터셋 생성
    train_ds = ListDataset([{ "start": data.index[0], "target": data["매출"] }], freq="M")

    # 딥AR 모델 생성 및 학습
    estimator = DeepAREstimator(freq="M", prediction_length=12, trainer=Trainer(epochs=10))
    predictor = estimator.train(train_ds)

    # 예측
    forecast_it, ts_it = make_evaluation_predictions(train_ds, predictor=predictor, num_samples=100)
    forecasts = list(forecast_it)

    next_year_forecast = forecasts[0].mean
    next_month_forecast = next_year_forecast[0]

    # 결과 출력
    st.write(f"총 결과: {len(filtered_data)}개")
    if len(filtered_data) > 0:
        # 총매출과 평균 판매가 계산
        total_sales = filtered_data["매출"].sum()
        avg_price = filtered_data["판매가"].mean()

        # 계산 결과 출력
        st.write(f"**총매출**: {total_sales:,}원")
        st.write(f"**평균 판매가**: {avg_price:,.2f}원")
        st.write(f"**다음년도 예상 매출**: {next_year_forecast.sum():,}원")
        st.write(f"**다음달 예상 매출**: {next_month_forecast:,}원")

        # 필터링된 데이터 출력
        st.dataframe(filtered_data)

        # 다운로드 버튼 추가
        output = io.BytesIO()
        filtered_data.to_excel(output, index=False, engine='openpyxl')
        output.seek(0)

        st.download_button(
            label="결과 다운로드",
            data=output,
            file_name="filtered_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    else:
        st.warning("조건에 맞는 데이터가 없습니다.")
else:
    st.info("엑셀 파일을 업로드해주세요.")
