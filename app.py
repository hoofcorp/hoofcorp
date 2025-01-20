import streamlit as st
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import plotly.express as px
from st_aggrid import AgGrid, GridOptionsBuilder
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
            "구독자수": int(stats_dict.get(item["id"]["videoId"], "0")),
            "제목": item["snippet"]["title"],
            "링크": f'<a href="https://www.youtube.com/watch?v={item["id"]["videoId"]}" target="_blank">동영상 보기</a>',
            "설명": item["snippet"]["description"],
        }
        for item in response.get("items", [])
    ]
    return data

# 페이지 설정
st.set_page_config(
    page_title="매출 예측 및 YouTube 트렌드 분석",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 제목 및 설명
st.title("📊 매출 예측 및 YouTube 트렌드 분석")
st.markdown("""
    이 대시보드는 매출 데이터를 분석 및 예측하고, YouTube 데이터를 활용하여 트렌드와 통찰을 제공합니다.
""")

# 탭 생성
tab1, tab2 = st.tabs(["💼 매출 데이터 분석", "📈 YouTube 트렌드 분석"])

with tab1:
    st.header("💼 매출 데이터 분석")

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

            min_price, max_price = st.slider(
                "판매가 범위",
                min_value=int(df["판매가"].min()),
                max_value=int(df["판매가"].max()),
                value=(int(df["판매가"].min()), int(df["판매가"].max()))
            )

            start_date, end_date = st.date_input(
                "날짜 범위 선택",
                value=[df["진행 날짜"].min(), df["진행 날짜"].max()],
                min_value=df["진행 날짜"].min(),
                max_value=df["진행 날짜"].max()
            )

        # 매출 데이터 필터링
        filtered_data = df.copy()
        if grades:
            filtered_data = filtered_data[filtered_data["행사등급"].isin(grades)]
        if malls:
            filtered_data = filtered_data[filtered_data["운영몰"].isin(malls)]
        if brands:
            filtered_data = filtered_data[filtered_data["브랜드명"].isin(brands)]

        filtered_data = filtered_data[
            (filtered_data["판매가"] >= min_price) &
            (filtered_data["판매가"] <= max_price)
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
        st.dataframe(filtered_data)

        # 매출 시각화 및 예측
        st.subheader("📈 매출 추이 시각화")
        monthly_sales = filtered_data.copy()
        monthly_sales["월"] = monthly_sales["진행 날짜"].dt.to_period("M")
        monthly_sales = monthly_sales.groupby("월")["매출"].sum().reset_index()
        monthly_sales["월"] = monthly_sales["월"].dt.to_timestamp()

        if not monthly_sales.empty:
            fig = px.line(monthly_sales, x="월", y="매출", title="월별 매출 추이", labels={"매출": "매출(원)", "월": "날짜"})
            st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.header("📈 YouTube 트렌드 분석")

    # YouTube 데이터 검색
    youtube_keyword = st.text_input("🔍 YouTube 검색 키워드 입력", placeholder="예: 신제품, 매출")
    max_results = st.slider("YouTube 검색 결과 개수", min_value=1, max_value=50, value=10)

    if youtube_keyword:
        youtube_data = fetch_youtube_data(youtube_keyword, max_results)
        youtube_df = pd.DataFrame(youtube_data)

        # 키워드 분석
        st.subheader("🧩 키워드 분석 (워드 클라우드)")
        text = " ".join(youtube_df["제목"])
        wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        st.pyplot(plt)

        # 조회수 트렌드 시각화
        st.subheader("📊 시간에 따른 조회수 변화")
        youtube_df["게시일"] = pd.to_datetime(youtube_df["게시일"])
        trend_data = youtube_df.groupby(youtube_df["게시일"].dt.date).agg({"구독자수": "sum", "링크": "count"}).reset_index()
        fig = px.line(trend_data, x="게시일", y="구독자수", title="시간에 따른 구독자 변화")
        st.plotly_chart(fig, use_container_width=True)

        st.download_button(
            label="📥 YouTube 데이터 다운로드 (CSV)",
            data=youtube_df.to_csv(index=False).encode("utf-8"),
            file_name=f"{youtube_keyword}_youtube_trend.csv",
            mime="text/csv"
        )
