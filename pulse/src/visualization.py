import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns

# 이탈 위험 모니터링 시각화
def display_risk_monitoring(churn_risk, segments, store=None, selected_segment=None, risk_threshold=70):
    st.header("이탈 위험 모니터링")
    
    # 백화점별 세그먼트별 이탈 위험도
    risk_data = []
    for store_name, store_risks in churn_risk['store_risk'].items():
        if store and store != "전체" and store != store_name:
            continue
            
        for segment_name, risk_value in store_risks.items():
            if selected_segment and "전체" not in selected_segment and segment_name not in selected_segment:
                continue
                
            risk_data.append({
                "백화점": store_name,
                "세그먼트": segment_name,
                "이탈_위험도": round(risk_value * 100, 1)  # 0-1 값을 퍼센트로 변환
            })
    
    risk_df = pd.DataFrame(risk_data)
    
    if not risk_df.empty:
        # 이탈 위험도 히트맵
        st.subheader("세그먼트별 이탈 위험도")
        
        fig = px.imshow(
            risk_df.pivot(index="세그먼트", columns="백화점", values="이탈_위험도"),
            text_auto=True,
            color_continuous_scale="RdYlGn_r",
            aspect="auto",
            title="세그먼트별 백화점별 이탈 위험도 (%)"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # 위험 유형별 대응 전략
        st.subheader("이탈 위험 유형별 대응 전략")
        
        risk_tabs = st.tabs(list(churn_risk['risk_types'].keys()))
        
        for i, (risk_type, description) in enumerate(churn_risk['risk_types'].items()):
            with risk_tabs[i]:
                st.write(f"**{risk_type}**: {description}")
                st.write("**권장 대응 전략:**")
                for strategy in churn_risk['prevention_strategies'].get(risk_type, []):
                    st.write(f"- {strategy}")
    else:
        st.info("선택한 조건에 맞는 이탈 위험 데이터가 없습니다.")

# 고객 세그먼트 분석 시각화
def display_segment_analysis(segments, store=None, selected_segment=None):
    st.header("고객 세그먼트 분석")
    
    # 세그먼트 분포 차트
    st.subheader("백화점별 세그먼트 분포")
    
    # 데이터 준비
    segment_data = []
    for store_name, dist in segments['segment_distribution'].items():
        if store and store != "전체" and store != store_name:
            continue
            
        for i, segment_name in enumerate(segments['income_segments']):
            if selected_segment and "전체" not in selected_segment and segment_name not in selected_segment:
                continue
                
            segment_data.append({
                "백화점": store_name,
                "세그먼트": segment_name,
                "비율": dist[i] * 100  # 비율을 퍼센트로 변환
            })
    
    segment_df = pd.DataFrame(segment_data)
    
    if not segment_df.empty:
        # 세그먼트 분포 시각화
        fig = px.bar(
            segment_df,
            x="백화점",
            y="비율",
            color="세그먼트",
            barmode="stack",
            title="백화점별 고객 세그먼트 분포 (%)",
            labels={"비율": "비율 (%)"}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # 세그먼트별 프로필 비교
        st.subheader("세그먼트별 상세 프로필")
        
        # 세그먼트 프로필 데이터 준비
        profile_data = []
        for segment, profile in segments['segment_profiles'].items():
            if selected_segment and "전체" not in selected_segment and segment not in selected_segment:
                continue
                
            profile_data.append({
                "세그먼트": segment,
                "평균_소득": profile["avg_income"],
                "주요_거주지역": ", ".join(profile["main_residence"]),
                "방문_빈도": profile["visit_frequency"],
                "평균_구매액": profile["avg_purchase"],
                "이탈_위험도": f"{profile['churn_risk'] * 100:.1f}%"
            })
        
        profile_df = pd.DataFrame(profile_data)
        st.dataframe(profile_df, use_container_width=True)
        
        # 세그먼트별 특성 비교
        st.subheader("세그먼트별 특성 비교")
        
        # 레이더 차트 데이터 준비
        radar_data = []
        metrics = ["소득 수준", "방문 빈도", "객단가", "충성도", "날씨 민감도"]
        
        # 가상의 데이터로 레이더 차트 구성 (실제로는 데이터에서 추출)
        for segment in segments['income_segments']:
            if selected_segment and "전체" not in selected_segment and segment not in selected_segment:
                continue
                
            if segment == "고소득층":
                values = [0.9, 0.8, 0.9, 0.7, 0.3]
            elif segment == "중상위층":
                values = [0.7, 0.7, 0.7, 0.6, 0.4]
            elif segment == "중산층":
                values = [0.5, 0.5, 0.5, 0.5, 0.6]
            else:  # 일반소비자
                values = [0.3, 0.3, 0.3, 0.4, 0.8]
                
            # 레이더 차트를 닫기 위해 첫 값 반복
            radar_data.append({
                "세그먼트": segment,
                "특성": metrics + [metrics[0]],
                "값": values + [values[0]]
            })
        
        # 레이더 차트 그리기
        fig = go.Figure()
        
        for data in radar_data:
            fig.add_trace(go.Scatterpolar(
                r=data["값"],
                theta=data["특성"],
                fill='toself',
                name=data["세그먼트"]
            ))
            
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            title="세그먼트별 특성 비교"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("선택한 조건에 맞는 세그먼트 데이터가 없습니다.")

# 방문 패턴 분석 시각화
def display_visit_pattern_analysis(visit_patterns, weather_impact, store=None):
    st.header("백화점 방문 패턴 분석")
    
    # 백화점 선택
    stores = visit_patterns['store_trends']['DEP_NAME'].unique()
    selected_store = st.selectbox(
        "백화점 선택",
        ["전체"] + list(stores),
        index=0 if store is None else (list(stores).index(store) + 1 if store in stores else 0)
    )
    
    # 시계열 트렌드 분석
    st.subheader("월별 방문 트렌드")
    
    # 데이터 필터링
    if selected_store != "전체":
        trend_data = visit_patterns['store_trends'][visit_patterns['store_trends']['DEP_NAME'] == selected_store]
    else:
        trend_data = visit_patterns['store_trends']
    
    # 연도와 월을 문자열로 결합
    trend_data['year_month'] = trend_data['year'].astype(str) + '-' + trend_data['month'].astype(str).str.zfill(2)
    
    # 시계열 차트
    fig = px.line(
        trend_data,
        x='year_month',
        y='avg_daily_visits',
        color='DEP_NAME' if selected_store == "전체" else None,
        title="월별 평균 일일 방문객 수",
        labels={"avg_daily_visits": "평균 일일 방문객", "year_month": "년월", "DEP_NAME": "백화점"}
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # 요일별 방문 패턴
    st.subheader("요일별 방문 패턴")
    
    # 데이터 필터링
    if selected_store != "전체":
        dow_data = visit_patterns['dow_patterns'][visit_patterns['dow_patterns']['DEP_NAME'] == selected_store]
    else:
        dow_data = visit_patterns['dow_patterns']
    
    # 요일 이름 매핑
    dow_names = {
        0: '월요일', 1: '화요일', 2: '수요일', 
        3: '목요일', 4: '금요일', 5: '토요일', 6: '일요일'
    }
    dow_data['요일'] = dow_data['dayofweek'].map(dow_names)
    
    # 요일별 차트
    fig = px.bar(
        dow_data,
        x='요일',
        y='avg_visits',
        color='DEP_NAME' if selected_store == "전체" else None,
        title="요일별 평균 방문객 수",
        labels={"avg_visits": "평균 방문객", "요일": "요일", "DEP_NAME": "백화점"},
        category_orders={"요일": [dow_names[i] for i in range(7)]}
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # 계절별 방문 패턴
    st.subheader("계절별 방문 패턴")
    
    # 데이터 필터링
    if selected_store != "전체":
        season_data = visit_patterns['seasonal_patterns'][visit_patterns['seasonal_patterns']['DEP_NAME'] == selected_store]
    else:
        season_data = visit_patterns['seasonal_patterns']
    
    # 계절 순서 설정
    season_order = ['Winter', 'Spring', 'Summer', 'Fall']
    
    # 계절별 차트
    fig = px.bar(
        season_data,
        x='season',
        y='avg_visits',
        color='DEP_NAME' if selected_store == "전체" else None,
        title="계절별 평균 방문객 수",
        labels={"avg_visits": "평균 방문객", "season": "계절", "DEP_NAME": "백화점"},
        category_orders={"season": season_order}
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # 날씨와 방문 관계
    st.subheader("날씨와 방문 관계")
    
    # 데이터 필터링
    if selected_store != "전체":
        weather_data = weather_impact['weather_impact'][weather_impact['weather_impact']['DEP_NAME'] == selected_store]
    else:
        weather_data = weather_impact['weather_impact']
    
    # 기온별 방문 패턴
    temp_data = weather_data.groupby(['DEP_NAME', 'temp_range']).agg(
        avg_visits=('avg_visits', 'mean')
    ).reset_index()
    
    # 기온별 차트
    fig = px.bar(
        temp_data,
        x='temp_range',
        y='avg_visits',
        color='DEP_NAME' if selected_store == "전체" else None,
        title="기온별 평균 방문객 수",
        labels={"avg_visits": "평균 방문객", "temp_range": "기온 범위", "DEP_NAME": "백화점"},
        category_orders={"temp_range": ['매우 추움', '추움', '적정', '따뜻함', '더움']}
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # 강수량별 방문 패턴
    rain_data = weather_data.groupby(['DEP_NAME', 'rain_range']).agg(
        avg_visits=('avg_visits', 'mean')
    ).reset_index()
    
    # 강수량별 차트
    fig = px.bar(
        rain_data,
        x='rain_range',
        y='avg_visits',
        color='DEP_NAME' if selected_store == "전체" else None,
        title="강수량별 평균 방문객 수",
        labels={"avg_visits": "평균 방문객", "rain_range": "강수량 범위", "DEP_NAME": "백화점"},
        category_orders={"rain_range": ['맑음', '약한 비', '보통 비', '강한 비', '폭우']}
    )
    st.plotly_chart(fig, use_container_width=True)

# 주거지 및 소득 분석 시각화
def display_residence_income_analysis(residence_workplace, income_assets, property_prices, store=None):
    st.header("주거지 및 소득 분석")
    
    # 백화점 선택
    stores = residence_workplace['home_distribution']['DEP_NAME'].unique()
    selected_store = st.selectbox(
        "백화점 선택",
        ["전체"] + list(stores),
        index=0 if store is None else (list(stores).index(store) + 1 if store in stores else 0),
        key="residence_store_select"
    )
    
    # 주거지 분포 분석
    st.subheader("주요 고객 주거지 분포")
    
    # 데이터 필터링
    if selected_store != "전체":
        home_data = residence_workplace['top_home_areas'][residence_workplace['top_home_areas']['DEP_NAME'] == selected_store]
    else:
        home_data = residence_workplace['top_home_areas']
    
    # 주거지 차트
    fig = px.bar(
        home_data,
        x='ADDR_LV3',  # 법정동 수준
        y='ratio',
        color='DEP_NAME' if selected_store == "전체" else None,
        title="백화점별 주요 고객 거주지 (상위 10개 동)",
        labels={"ratio": "고객 비율", "ADDR_LV3": "거주 지역 (동)", "DEP_NAME": "백화점"}
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # 소득 분포 분석 (데이터가 있을 경우)
    if 'income_distribution' in income_assets and not income_assets['income_distribution'].empty:
        st.subheader("지역별 소득 분포")
        
        # 소득 분포 차트
        income_data = income_assets['income_distribution']
        
        fig = px.bar(
            income_data,
            x='DONG_NAME',
            y='est_income',
            title="지역별 추정 평균 소득 (만원/년)",
            labels={"est_income": "추정 평균 소득 (만원/년)", "DONG_NAME": "지역 (동)"}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # 부동산 시세 분석
    st.subheader("지역별 부동산 시세")
    
    # 구별 시세 차트
    fig = px.bar(
        property_prices['district_prices'],
        x='SGG',
        y=['avg_jeonse', 'avg_meme'],
        title="구별 평균 전세/매매 시세 (만원/평)",
        labels={"value": "가격 (만원/평)", "SGG": "자치구", "variable": "유형"},
        barmode='group'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # 부동산 시세와 방문 관계 분석 (가상 데이터)
    st.subheader("부동산 시세와 백화점 방문 관계")
    
    # 가상 데이터 생성
    price_visit_data = pd.DataFrame({
        "지역": ["서초구", "강남구", "영등포구", "중구", "종로구"],
        "평균_매매가": [7000, 8000, 5000, 4500, 4800],
        "신세계_강남_방문비율": [0.25, 0.30, 0.15, 0.10, 0.05],
        "더현대_서울_방문비율": [0.15, 0.20, 0.25, 0.15, 0.10],
        "롯데_본점_방문비율": [0.10, 0.15, 0.20, 0.30, 0.25]
    })
    
    # 부동산 시세와 방문 관계 차트
    fig = px.scatter(
        price_visit_data,
        x="평균_매매가",
        y="신세계_강남_방문비율" if selected_store == "신세계 강남" or selected_store == "전체" else
           "더현대_서울_방문비율" if selected_store == "더현대 서울" else
           "롯데_본점_방문비율",
        size="평균_매매가",
        color="지역",
        text="지역",
        title=f"부동산 시세와 {selected_store if selected_store != '전체' else '백화점'} 방문 관계",
        labels={
            "평균_매매가": "평균 매매가 (만원/평)",
            "신세계_강남_방문비율": "신세계 강남 방문 비율",
            "더현대_서울_방문비율": "더현대 서울 방문 비율",
            "롯데_본점_방문비율": "롯데 본점 방문 비율"
        }
    )
    fig.update_traces(textposition='top center')
    st.plotly_chart(fig, use_container_width=True)

# 맞춤형 대응 전략 시각화
def display_retention_strategies(churn_risk, segments, store=None, selected_segment=None):
    st.header("맞춤형 고객 유지 전략")
    
    # 세그먼트별 이탈 방지 전략
    st.subheader("세그먼트별 이탈 방지 전략")
    
    # 세그먼트 목록 확인 (동적으로 처리)
    available_segments = segments.get('income_segments', [])
    if not available_segments:
        st.warning("세그먼트 정보가 없습니다.")
        return
    
    # 세그먼트 탭
    segment_tabs = st.tabs(available_segments)
    
    # 세그먼트별 전략 - 모든 가능한 세그먼트 이름 포함
    segment_strategies = {
        # APP_CONFIG에 정의된 세그먼트 (실제 앱에서 사용)
        "VIP 고객": [
            "개인화된 VIP 서비스 강화",
            "프리미엄 이벤트 초청",
            "한정판 상품 우선 구매권",
            "프라이빗 쇼핑 경험 제공"
        ],
        "정기 방문 고객": [
            "충성도 프로그램 혜택 확대",
            "맞춤형 상품 추천 서비스",
            "멤버십 등급 상향 기회 제공",
            "정기 방문 보상 프로그램"
        ],
        "간헐적 방문 고객": [
            "방문 빈도 증대 인센티브",
            "시즌 행사 특별 초대",
            "온라인-오프라인 연계 서비스",
            "리마인더 마케팅 강화"
        ],
        "이탈 위험 고객": [
            "즉각적인 할인 혜택 제공",
            "재방문 인센티브 강화",
            "1:1 고객 상담 서비스",
            "서비스 불만족 사항 해결"
        ],
        # feature_eng.py에서 참조되는 세그먼트 (이전 코드와의 호환성 유지)
        "고소득층": [
            "개인화된 VIP 서비스 강화",
            "프리미엄 이벤트 초청",
            "한정판 상품 우선 구매권",
            "프라이빗 쇼핑 경험 제공"
        ],
        "중상위층": [
            "충성도 프로그램 혜택 확대",
            "프리미엄 브랜드 특별 할인",
            "다양한 제휴 혜택 강화",
            "맞춤형 상품 추천 서비스"
        ],
        "중산층": [
            "가성비 높은 제품 라인업 강화",
            "시즌 세일 사전 안내",
            "가족 단위 이벤트 및 프로모션",
            "포인트 적립률 상향"
        ],
        "일반소비자": [
            "입문 가격대 상품 확대",
            "첫 구매 인센티브 강화",
            "실용적 혜택 중심 마케팅",
            "대중교통 연계 할인 혜택"
        ]
    }
    
    # 세그먼트별 프로모션 효과 매핑 (동적으로 대응)
    promotion_effects_mapping = {
        # APP_CONFIG에 정의된 세그먼트
        "VIP 고객": {
            "VIP 혜택": 0.7,  # 30% 감소
            "포인트 적립": 0.9,  # 10% 감소
            "할인": 0.85,  # 15% 감소
            "이벤트 초대": 0.75,  # 25% 감소
            "무료 배송": 0.95  # 5% 감소
        },
        "정기 방문 고객": {
            "VIP 혜택": 0.8,
            "포인트 적립": 0.75,
            "할인": 0.7,
            "이벤트 초대": 0.85,
            "무료 배송": 0.9
        },
        "간헐적 방문 고객": {
            "VIP 혜택": 0.9,
            "포인트 적립": 0.8,
            "할인": 0.65,
            "이벤트 초대": 0.85,
            "무료 배송": 0.75
        },
        "이탈 위험 고객": {
            "VIP 혜택": 0.95,
            "포인트 적립": 0.85,
            "할인": 0.6,
            "이벤트 초대": 0.9,
            "무료 배송": 0.75
        },
        # feature_eng.py에서 참조되는 세그먼트
        "고소득층": {
            "VIP 혜택": 0.7,
            "포인트 적립": 0.9,
            "할인": 0.85,
            "이벤트 초대": 0.75,
            "무료 배송": 0.95
        },
        "중상위층": {
            "VIP 혜택": 0.8,
            "포인트 적립": 0.75,
            "할인": 0.7,
            "이벤트 초대": 0.85,
            "무료 배송": 0.9
        },
        "중산층": {
            "VIP 혜택": 0.9,
            "포인트 적립": 0.8,
            "할인": 0.65,
            "이벤트 초대": 0.85,
            "무료 배송": 0.75
        },
        "일반소비자": {
            "VIP 혜택": 0.95,
            "포인트 적립": 0.85,
            "할인": 0.6,
            "이벤트 초대": 0.9,
            "무료 배송": 0.75
        }
    }
    
    # 기본 프로모션 효과 (세그먼트가 정의되지 않은 경우 사용)
    default_promotion_effects = {
        "VIP 혜택": 0.85,
        "포인트 적립": 0.8,
        "할인": 0.7,
        "이벤트 초대": 0.85,
        "무료 배송": 0.85
    }
    
    # 각 세그먼트 탭의 내용 표시
    for i, segment in enumerate(available_segments):
        with segment_tabs[i]:
            if selected_segment and "전체" not in selected_segment and segment not in selected_segment:
                st.info(f"{segment} 세그먼트가 선택되지 않았습니다.")
                continue
            
            # 세그먼트 프로필 안전하게 접근
            segment_profile = segments.get('segment_profiles', {}).get(segment, {})
            
            st.write(f"### {segment} 맞춤 전략")
            
            # 프로필 정보 안전하게 표시
            avg_income = segment_profile.get('avg_income', '정보 없음')
            st.write(f"평균 소득: {avg_income}")
            
            main_residence = segment_profile.get('main_residence', ['정보 없음'])
            if isinstance(main_residence, list):
                st.write(f"주요 거주지역: {', '.join(main_residence)}")
            else:
                st.write(f"주요 거주지역: {main_residence}")
            
            st.write(f"방문 빈도: {segment_profile.get('visit_frequency', '정보 없음')}")
            st.write(f"평균 구매액: {segment_profile.get('avg_purchase', '정보 없음')}")
            
            st.write("#### 이탈 방지 전략")
            strategies = segment_strategies.get(segment, ["맞춤 전략 정보가 없습니다."])
            for strategy in strategies:
                st.write(f"- {strategy}")
            
            # 세그먼트별 이탈 위험 및 프로모션 효과 시뮬레이션
            st.write("#### 이탈 위험 및 프로모션 효과 시뮬레이션")
            
            # 프로모션 선택
            promotion_type = st.selectbox(
                "프로모션 유형",
                ["VIP 혜택", "포인트 적립", "할인", "이벤트 초대", "무료 배송"],
                key=f"promotion_{segment}"
            )
            
            promotion_intensity = st.slider(
                "프로모션 강도 (%)",
                min_value=5,
                max_value=30,
                value=15,
                step=5,
                key=f"intensity_{segment}"
            )
            
            # 이탈 위험 고객 비율 (기본)
            base_risk = segment_profile.get('churn_risk', 0.3) * 100  # 기본값 0.3 (30%)
            
            # 효과 계산 - 동적으로 세그먼트 효과 매핑
            # 세그먼트별 효과를 찾거나 기본값 사용
            segment_effects = promotion_effects_mapping.get(segment, default_promotion_effects)
            effect_ratio = segment_effects.get(promotion_type, 0.8)  # 기본값 0.8 (20% 감소)
            
            # 프로모션 강도에 따른 조정
            adjusted_effect = effect_ratio + (1 - effect_ratio) * (promotion_intensity - 15) / 30
            adjusted_effect = max(0.5, min(0.95, adjusted_effect))  # 효과 제한
            
            # 최종 이탈 위험
            reduced_risk = base_risk * adjusted_effect
            
            # 결과 표시
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric(
                    "기존 이탈 위험도",
                    f"{base_risk:.1f}%"
                )
            
            with col2:
                st.metric(
                    "프로모션 후 이탈 위험도",
                    f"{reduced_risk:.1f}%",
                    f"-{base_risk - reduced_risk:.1f}%"
                )
            
            # ROI 계산 (안전하게 처리)
            customer_count = 1000  # 가정: 세그먼트당 1000명
            
            # 세그먼트별 비용 계수 매핑
            cost_factor = 1.0  # 기본값
            if segment in ["VIP 고객", "고소득층"]:
                cost_factor = 2.0
            elif segment in ["정기 방문 고객", "중상위층"]:
                cost_factor = 1.5
            elif segment in ["간헐적 방문 고객", "중산층"]:
                cost_factor = 1.0
            elif segment in ["이탈 위험 고객", "일반소비자"]:
                cost_factor = 0.5
            
            promotion_cost = promotion_intensity * customer_count * cost_factor
            
            saved_customers = customer_count * (base_risk / 100) * (1 - adjusted_effect)
            
            # 고객 가치 계산 (안전하게)
            avg_purchase_str = str(segment_profile.get('avg_purchase', '10만원'))
            try:
                # 문자열에서 숫자 추출 시도
                avg_purchase_value = float(avg_purchase_str.replace('만원', '').replace('~', '-').split('-')[0])
            except (ValueError, IndexError):
                avg_purchase_value = 10  # 기본값 10만원
            
            customer_value = avg_purchase_value * 12  # 연간 가치
            benefit = saved_customers * customer_value
            
            # ROI 계산 (0으로 나누기 방지)
            roi = (benefit - promotion_cost) / promotion_cost if promotion_cost > 0 else 0
            
            # ROI 표시
            st.write("#### 프로모션 ROI 시뮬레이션")
            
            roi_cols = st.columns(3)
            
            with roi_cols[0]:
                st.metric("예상 비용 (만원)", f"{promotion_cost/10000:.1f}")
            
            with roi_cols[1]:
                st.metric("예상 수익 (만원)", f"{benefit/10000:.1f}")
            
            with roi_cols[2]:
                st.metric("ROI", f"{roi:.1f}x", delta="양호" if roi > 1 else "주의")
    
    # 이탈 방지 캠페인 계획
    st.subheader("이탈 방지 캠페인 계획")
    
    campaign_options = {
        "시즌별 캠페인": ["봄", "여름", "가을", "겨울"],
        "날씨 연계 캠페인": ["우천 시", "한파 시", "무더위 시", "미세먼지 심각 시"],
        "특별 이벤트": ["명절", "블랙프라이데이", "연말", "새해"],
        "생애 주기 이벤트": ["생일", "결혼기념일", "회원가입 기념일"]
    }
    
    campaign_type = st.selectbox(
        "캠페인 유형",
        list(campaign_options.keys())
    )
    
    campaign_timing = st.selectbox(
        "캠페인 시점",
        campaign_options.get(campaign_type, ["선택하세요"])
    )
    
    # 타겟 세그먼트 - 기본값 없이 설정
    target_segments = st.multiselect(
        "타겟 세그먼트",
        available_segments,
        default=[]  # 기본값 없음
    )
    
    promotion_methods = st.multiselect(
        "프로모션 방법",
        ["이메일", "SMS", "앱 푸시", "우편물", "백화점 내 안내"],
        default=["이메일", "SMS"]  # 기본 선택값 (일반적인 선택지라 남겨둠)
    )
    
    # 캠페인 계획 생성 버튼
    if st.button("캠페인 계획 생성"):
        st.write("### 캠페인 실행 계획")
        
        # 캠페인 이름 생성
        campaign_name = f"{campaign_timing} {campaign_type.replace('캠페인', '').strip()} 고객 유지 캠페인"
        
        st.write(f"**캠페인명:** {campaign_name}")
        st.write(f"**대상 세그먼트:** {', '.join(target_segments) if target_segments else '선택된 세그먼트 없음'}")
        st.write(f"**전달 방법:** {', '.join(promotion_methods)}")
        
        # 세그먼트별 맞춤 내용
        if target_segments:
            st.write("**세그먼트별 맞춤 내용:**")
            for segment in target_segments:
                # 세그먼트별 오퍼 매핑 (앱에서 사용하는 세그먼트와 feature_eng.py 세그먼트 모두 지원)
                segment_offer = ""
                if segment in ["VIP 고객", "고소득층"]:
                    segment_offer = "프리미엄 VIP 초대장 및 사은품"
                elif segment in ["정기 방문 고객", "중상위층"]:
                    segment_offer = "추가 포인트 적립 및 특별 쿠폰"
                elif segment in ["간헐적 방문 고객", "중산층"]:
                    segment_offer = "시즌 할인 및 사은품"
                elif segment in ["이탈 위험 고객", "일반소비자"]:
                    segment_offer = "특별 할인 및 이벤트 참여 기회"
                else:
                    segment_offer = "맞춤형 혜택 및 서비스"
                
                st.write(f"- {segment}: {segment_offer}")
        else:
            st.write("**세그먼트별 맞춤 내용:** 세그먼트를 선택하세요.")
        
        # 일정 표시
        st.write("**실행 일정:**")
        st.write("- 기획 완료: D-14일")
        st.write("- 콘텐츠 제작: D-10일")
        st.write("- 시스템 세팅: D-7일")
        st.write("- 테스트: D-5일")
        st.write("- 발송: D-day")
        st.write("- 효과 측정: D+7일, D+14일, D+30일")
        
        # 예상 효과 계산
        if target_segments:
            avg_risk_reduction = 0
            for segment in target_segments:
                # 세그먼트별 기본 효과 매핑
                if segment in ["VIP 고객", "고소득층"]:
                    base_effect = 0.3
                elif segment in ["정기 방문 고객", "중상위층"]:
                    base_effect = 0.25
                elif segment in ["간헐적 방문 고객", "중산층"]:
                    base_effect = 0.2
                elif segment in ["이탈 위험 고객", "일반소비자"]:
                    base_effect = 0.15
                else:
                    base_effect = 0.2  # 기본값
                
                avg_risk_reduction += base_effect
            
            # 평균 계산 (0으로 나누기 방지)
            avg_risk_reduction = avg_risk_reduction / len(target_segments) if target_segments else 0.2
            
            st.write("**예상 효과:**")
            st.write(f"- 이탈 위험 감소: {avg_risk_reduction*100:.1f}%")
            st.write(f"- 방문 빈도 증가: {avg_risk_reduction*50:.1f}%")
            st.write(f"- 객단가 증가: {avg_risk_reduction*30:.1f}%")
        else:
            st.write("**예상 효과:** 세그먼트를 선택하세요.")