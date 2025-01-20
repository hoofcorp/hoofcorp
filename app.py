import streamlit as st
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import plotly.express as px
from st_aggrid import AgGrid
from googleapiclient.discovery import build

# YouTube API 키
YOUTUBE_API_KEY = "AIzaSyAHjsvQRyMnFVsjbFgj02Ws5dXMgnTOD0M"

# YouTube API 클라이언트 초기화
youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)

# YouTube 데이터 가져오기 함수
def fetch_youtube_data(keyword, max_results=10):
    request = youtube.search().list(
        part="snippet",
        q=keyword,
        type="video",
        maxResults=max_results
    )
    response = request.execute()

    # 동영상 ID 추출
    video_ids = [item["id"]["videoId"] for item in response["items"]]
    stats_request = youtube.videos().list(
        part="statistics",
        id=",".join(video_ids)
    )
    stats_response = stats_request.execute()
    stats_dict = {item["id"]: item["statistics"].get("viewCount", "0") for item in stats_response["items"]}

    data = [
        {
            "게시일": item["snippet"]["publishedAt"],
            "채널명": item["snippet"]["channelTitle"],
            "구독자수": stats_dict.get(item["id"]["videoId"], "0"),
            "제목": item["snippet"]["title"],
            "링크": f"https://www.youtube.com/watch?v={item['id']['videoId']}",
            "설명": item["snippet"]["description"],
        }
        for item in response.get("items", [])
    ]
    return data

# 페이지 설정
st.set_page_config(
    page_title="매출 예측 및 YouTube 데이터 대시보드",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 제목 및 설명
st.title("📊 매출 예측 및 YouTube 데이터 대시보드")
st.markdown("""
    이 대시보드는 매출 데이터를 분석하고 예측하며, YouTube 데이터를 연동하여 다양한 정보를 제공합니다.
""")

# 파일 업로드
uploaded_file = st.file_uploader("매출 데이터를 업로드하세요 (Excel)", type=["xlsx", "xls"])

if uploaded_file:
    # 데이터 로드 함수
    @st.cache_data
    def load_data(file):
        df = pd.read_excel(file)
        df["진행 날짜"] = pd.to_datetime(df["진행 날짜"], format='%Y%m%d')  # 날짜 형식 변환
        return df

    df = load_data(uploaded_file)

    # 데이터 필터링 UI
    with st.sidebar:
        st.header("매출 데이터 필터 설정")
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

    # YouTube 데이터 필터 설정
    with st.sidebar:
        st.header("YouTube 데이터 필터 설정")
        youtube_keyword = st.text_input("🔍 YouTube 검색 키워드 입력", placeholder="예: 신제품, 매출")
        max_results = st.slider("YouTube 검색 결과 개수", min_value=1, max_value=50, value=10)

    # 매출 데이터 필터링
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
    st.subheader("📋 필터링된 매출 데이터")
    AgGrid(filtered_data, height=300, theme="streamlit")

    # 매출 시각화 및 예측
    st.subheader("📈 매출 추이 시각화")
    monthly_sales = filtered_data.copy()
    monthly_sales["월"] = monthly_sales["진행 날짜"].dt.to_period("M")
    monthly_sales = monthly_sales.groupby("월")["매출"].sum().reset_index()
    monthly_sales["월"] = monthly_sales["월"].dt.to_timestamp()

    if not monthly_sales.empty:
        fig = px.line(monthly_sales, x="월", y="매출", title="월별 매출 추이", labels={"매출": "매출(원)", "월": "날짜"})
        st.plotly_chart(fig, use_container_width=True)

        # 매출 예측
        st.subheader("🔮 매출 예측")
        if len(monthly_sales) >= 2:
            periods_to_forecast = st.slider("예측할 개월 수", 1, 24, 12)
            try:
                model = ExponentialSmoothing(
                    monthly_sales["매출"],
                    trend="add",
                    seasonal="add" if len(monthly_sales) >= 24 else None,
                    seasonal_periods=12 if len(monthly_sales) >= 24 else None,
                )
                model_fit = model.fit()
                forecast = model_fit.forecast(periods_to_forecast)

                forecast_dates = pd.date_range(
                    start=monthly_sales["월"].iloc[-1] + pd.offsets.MonthBegin(),
                    periods=periods_to_forecast,
                    freq="MS"
                )
                forecast_df = pd.DataFrame({"예측 날짜": forecast_dates, "예상 매출": forecast})

                # 예측 그래프
                forecast_fig = px.line(
                    forecast_df, x="예측 날짜", y="예상 매출", title="예상 매출 추이", labels={"예상 매출": "매출(원)", "예측 날짜": "날짜"}
                )
                forecast_fig.add_scatter(x=monthly_sales["월"], y=monthly_sales["매출"], mode="lines", name="실제 매출")
                st.plotly_chart(forecast_fig, use_container_width=True)

                # 예측 결과 다운로드
                st.download_button(
                    label="📥 예측 결과 다운로드",
                    data=forecast_df.to_csv(index=False).encode("utf-8"),
                    file_name="forecast.csv",
                    mime="text/csv"
                )
            except Exception as e:
                st.error(f"예측 중 오류가 발생했습니다: {e}")
        else:
            st.warning("데이터가 부족하여 매출 예측을 수행할 수 없습니다.")

    # YouTube 데이터 검색
    if youtube_keyword:
        st.subheader(f"🔍 YouTube 검색 결과 - '{youtube_keyword}'")
        youtube_data = fetch_youtube_data(youtube_keyword, max_results)
        youtube_df = pd.DataFrame(youtube_data)

        # 링크를 클릭 가능한 HTML 형식으로 변환
        youtube_df["링크"] = youtube_df["링크"].apply(
            lambda x: f'<a href="{x}" target="_blank">동영상 보기</a>'
        )

        # 데이터 표시 (Streamlit HTML 렌더링 사용)
        st.write(
            youtube_df[["게시일", "채널명", "구독자수", "제목", "링크", "설명"]].to_html(escape=False, index=False),
            unsafe_allow_html=True
        )

        # YouTube 데이터 다운로드 버튼
        st.download_button(
            label="📥 YouTube 데이터 다운로드 (CSV)",
            data=youtube_df.to_csv(index=False).encode("utf-8"),
            file_name=f"{youtube_keyword}_youtube_results.csv",
            mime="text/csv"
        )

else:
    st.info("📤 데이터를 업로드하거나 YouTube 검색을 시작하세요.")
