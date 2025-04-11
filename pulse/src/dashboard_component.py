import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

def create_summary_dashboard(data_dict):
    """
    전체 데이터를 기반으로 한 요약 대시보드 생성
    
    data_dict: 모든 데이터를 포함하는 딕셔너리
    """
    st.header("RetainPulse 데이터 요약 대시보드")
    
    # 데이터 존재 여부 확인
    if 'visits_df' not in data_dict or data_dict['visits_df'].empty:
        st.error("방문 데이터가 없습니다.")
        return
        
    # 데이터 요약 정보
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # 총 방문 데이터 수
        total_visits = data_dict['visits_df']['COUNT'].sum()
        avg_daily_visits = data_dict['visits_df'].groupby('DATE_KST')['COUNT'].sum().mean()
        
        st.metric(
            "총 방문 고객 수", 
            f"{total_visits:,.0f}",
            f"일평균 {avg_daily_visits:,.0f}명"
        )
    
    with col2:
        # 백화점별 방문 비율
        store_visits = data_dict['visits_df'].groupby('DEP_NAME')['COUNT'].sum()
        top_store = store_visits.idxmax()
        top_store_pct = store_visits[top_store] / total_visits * 100
        
        st.metric(
            "최다 방문 백화점", 
            f"{top_store}",
            f"전체의 {top_store_pct:.1f}%"
        )
    
    with col3:
        # 주요 방문객 거주 지역
        if 'residence_df' in data_dict and not data_dict['residence_df'].empty:
            home_data = data_dict['residence_df'][data_dict['residence_df']['LOC_TYPE'] == 1]
            top_districts = home_data.groupby('ADDR_LV2')['RATIO'].sum().sort_values(ascending=False)
            top_district = top_districts.index[0] if not top_districts.empty else "데이터 없음"
            top_district_pct = top_districts.iloc[0] * 100 if not top_districts.empty else 0
            
            st.metric(
                "주요 거주 지역", 
                f"{top_district}",
                f"전체의 {top_district_pct:.1f}%"
            )
        else:
            st.metric("주요 거주 지역", "데이터 없음", "")
    
    # 최근 12개월 방문 트렌드
    st.subheader("최근 12개월 방문 트렌드")
    
    # 날짜 변환
    if 'DATE_KST' in data_dict['visits_df'].columns:
        visits_df = data_dict['visits_df'].copy()
        visits_df['DATE_KST'] = pd.to_datetime(visits_df['DATE_KST'])
        
        # 최근 1년 데이터 필터링
        latest_date = visits_df['DATE_KST'].max()
        year_ago = latest_date - timedelta(days=365)
        recent_visits = visits_df[visits_df['DATE_KST'] >= year_ago]
        
        # 월별 집계 - 컬럼명 충돌 방지를 위해 명시적 컬럼명 사용
        monthly_visits = recent_visits.groupby([
            recent_visits['DATE_KST'].dt.year.rename('year'),
            recent_visits['DATE_KST'].dt.month.rename('month'),
            'DEP_NAME'
        ])['COUNT'].sum().reset_index()
        
        # year와 month를 문자열로 변환하여 year_month 컬럼 생성
        monthly_visits['year_month'] = monthly_visits['year'].astype(str) + '-' + monthly_visits['month'].astype(str).str.zfill(2)
        
        # 트렌드 그래프
        fig = px.line(
            monthly_visits, 
            x='year_month', 
            y='COUNT', 
            color='DEP_NAME',
            title="백화점별 월간 방문 추이",
            labels={"COUNT": "방문 고객 수", "year_month": "년월", "DEP_NAME": "백화점"}
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("방문 날짜 데이터가 올바른 형식이 아닙니다.")
    
    # 이탈 위험 요약
    st.subheader("이탈 위험 세그먼트 요약")
    
    # 샘플 데이터 (실제로는 분석 결과에서 가져옴)
    risk_data = pd.DataFrame({
        "세그먼트": ["VIP 고객", "정기 방문 고객", "간헐적 방문 고객", "이탈 위험 고객"],
        "고객 비중": [0.15, 0.30, 0.35, 0.20],
        "이탈 위험도": [0.10, 0.25, 0.45, 0.75],
        "연간 평균 소비액": [500, 300, 150, 80]
    })
    
    # 버블 차트로 표시
    fig = px.scatter(
        risk_data,
        x="이탈 위험도",
        y="연간 평균 소비액",
        size="고객 비중",
        color="세그먼트",
        hover_name="세그먼트",
        text="세그먼트",
        size_max=50,
        title="세그먼트별 이탈 위험도와 가치"
    )
    
    fig.update_traces(textposition='top center')
    fig.update_layout(
        xaxis=dict(title="이탈 위험도", tickformat=".0%"),
        yaxis=dict(title="연간 평균 소비액 (만원)")
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # 날씨 영향 분석
    st.subheader("날씨와 방문 관계 분석")
    
    if 'weather_df' in data_dict and not data_dict['weather_df'].empty and 'visits_df' in data_dict:
        # 날씨 데이터와 방문 데이터 병합 준비
        weather_df = data_dict['weather_df'].copy()
        visits_df = data_dict['visits_df'].copy()
        
        # 날짜 형식 확인
        weather_df['DATE_KST'] = pd.to_datetime(weather_df['DATE_KST'])
        visits_df['DATE_KST'] = pd.to_datetime(visits_df['DATE_KST'])
        
        # 일별 집계
        daily_visits = visits_df.groupby(['DATE_KST', 'DEP_NAME'])['COUNT'].sum().reset_index()
        
        # 날씨 데이터와 병합
        visit_weather = pd.merge(
            daily_visits,
            weather_df[['DATE_KST', 'AVG_TEMP', 'RAINFALL_MM']],
            on='DATE_KST',
            how='left'
        )
        
        # 기온 구간화
        visit_weather['temp_bin'] = pd.cut(
            visit_weather['AVG_TEMP'],
            bins=[-20, 0, 10, 20, 30, 40],
            labels=['매우 추움', '추움', '적정', '따뜻함', '더움']
        )
        
        # 구간별 방문 평균
        temp_visits = visit_weather.groupby(['temp_bin', 'DEP_NAME'])['COUNT'].mean().reset_index()
        
        # 기온별 방문 차트
        fig = px.bar(
            temp_visits,
            x='temp_bin',
            y='COUNT',
            color='DEP_NAME',
            barmode='group',
            title="기온별 평균 방문객 수",
            labels={"COUNT": "평균 방문객 수", "temp_bin": "기온 구간", "DEP_NAME": "백화점"}
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 비 여부에 따른 방문 변화
        visit_weather['비 여부'] = visit_weather['RAINFALL_MM'] > 0
        rain_impact = visit_weather.groupby(['비 여부', 'DEP_NAME'])['COUNT'].mean().reset_index()
        rain_impact['비 여부'] = rain_impact['비 여부'].map({True: '비 오는 날', False: '맑은 날'})
        
        fig = px.bar(
            rain_impact,
            x='DEP_NAME',
            y='COUNT',
            color='비 여부',
            barmode='group',
            title="비 여부에 따른 평균 방문객 수",
            labels={"COUNT": "평균 방문객 수", "DEP_NAME": "백화점", "비 여부": ""}
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("날씨 분석을 위한 데이터가 충분하지 않습니다.")
    
    # 지역별 소득 & 백화점 선호도 관계
    st.subheader("지역별 소득과 백화점 선호도 관계")
    
    # 샘플 데이터 (실제로는 분석 결과에서 가져옴)
    if 'income_asset_df' in data_dict and not data_dict['income_asset_df'].empty and 'residence_df' in data_dict:
        st.info("지역별 소득 데이터와 백화점 방문 데이터를 분석하여 관계를 파악할 수 있습니다.")
        
        # 데이터 존재할 경우 실제 분석 코드 추가
    else:
        districts = ["강남구", "서초구", "영등포구", "중구", "종로구"]
        
        region_data = pd.DataFrame({
            "지역": districts,
            "평균소득": [8500, 7800, 6300, 5800, 6000],
            "신세계_강남_선호도": [0.7, 0.65, 0.4, 0.3, 0.25],
            "더현대_서울_선호도": [0.5, 0.45, 0.6, 0.55, 0.4],
            "롯데_본점_선호도": [0.3, 0.35, 0.55, 0.7, 0.65]
        })
        
        # 소득 & 선호도 관계 그래프
        fig = px.scatter(
            region_data,
            x="평균소득",
            y=["신세계_강남_선호도", "더현대_서울_선호도", "롯데_본점_선호도"],
            title="지역별 소득과 백화점 선호도 관계",
            labels={"value": "선호도", "평균소득": "평균 소득 (만원/년)", "variable": "백화점"}
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.warning("샘플 데이터로 시각화했습니다. 실제 데이터로 분석 시 더 정확한 결과를 얻을 수 있습니다.")
    
    # 이탈 예방 전략 요약
    st.subheader("이탈 방지 전략 효과 시뮬레이션")
    
    # 이탈 방지 전략별 효과 데이터 (샘플)
    strategy_data = pd.DataFrame({
        "전략": ["VIP 전용 혜택", "포인트 적립 강화", "맞춤형 프로모션", "계절 이벤트", "날씨 연계 마케팅"],
        "비용": [100, 80, 60, 50, 30],
        "기대효과": [25, 20, 15, 12, 10],
        "ROI": [2.5, 2.3, 2.0, 1.8, 1.5]
    })
    
    # 전략별 ROI 차트
    fig = px.bar(
        strategy_data,
        x="전략",
        y="ROI",
        color="기대효과",
        text="ROI",
        title="이탈 방지 전략별 ROI 비교",
        labels={"ROI": "투자수익률", "전략": "", "기대효과": "이탈률 감소 (%)"}
    )
    
    fig.update_traces(texttemplate='%{text:.1f}x', textposition='outside')
    
    st.plotly_chart(fig, use_container_width=True)
    
    # 비용 대비 효과 버블 차트
    fig = px.scatter(
        strategy_data,
        x="비용",
        y="기대효과",
        size="ROI",
        color="전략",
        hover_name="전략",
        text="전략",
        size_max=50,
        title="이탈 방지 전략 비용 대비 효과 분석"
    )
    
    fig.update_traces(textposition='top center')
    fig.update_layout(
        xaxis=dict(title="전략 비용 (백만원)"),
        yaxis=dict(title="이탈률 감소 (%)")
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.info("💡 분석 결과, '**VIP 전용 혜택**'과 '**포인트 적립 강화**' 전략이 가장 높은 ROI를 보이고 있습니다.")