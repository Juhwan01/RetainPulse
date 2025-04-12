import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import snowflake.connector
from snowflake.connector.pandas_tools import write_pandas

# 로컬 모듈 임포트
from src.data_loader import (
    get_snowflake_connection, load_department_store_visits, load_residence_workplace_data,
    load_weather_data, load_floating_population_data, load_income_asset_data,
    load_card_spending_data, load_apartment_price_data, load_population_data,
    load_female_child_data, load_administrative_boundary, load_region_master_data,
    generate_sample_data
)
from src.feature_eng import (
    analyze_visit_patterns, analyze_residence_workplace, analyze_weather_impact,
    analyze_income_assets, analyze_property_prices, create_customer_segments,
    predict_churn_risk, prepare_integrated_modeling_dataset
)
from src.visualization import (
    display_advanced_risk_monitoring, display_segment_analysis,  # 업데이트된 시각화 함수 사용
    display_visit_pattern_analysis, display_residence_income_analysis,
    display_retention_strategies
)
# 새로 추가한 대시보드 요약 컴포넌트 임포트
from src.dashboard_component import create_summary_dashboard
from config import SNOWFLAKE_CONFIG, LOPLAT_CONFIG, RESIDENCE_CONFIG, WEATHER_CONFIG, SPH_CONFIG, DATAKNOWS_CONFIG, APP_CONFIG

# 앱 제목 설정
st.set_page_config(
    page_title="RetainPulse - 백화점 고객 이탈 예측 및 방지 시스템",
    page_icon="🛍️",
    layout="wide"
)

# 앱 헤더
st.title("RetainPulse - 백화점 고객 이탈 예측 및 방지 시스템")
st.write("백화점 고객 행동 데이터를 활용한 이탈 예측 및 맞춤형 고객 유지 전략 시스템")

# 세션 상태 초기화
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'analysis_done' not in st.session_state:
    st.session_state.analysis_done = False

# 사이드바 - 필터 옵션
st.sidebar.header("필터 옵션")
selected_store = st.sidebar.selectbox(
    "백화점 선택", ["전체"] + APP_CONFIG["stores"]
)

selected_segment = st.sidebar.multiselect(
    "고객 세그먼트", 
    ["전체"] + APP_CONFIG["segments"]
)

risk_threshold = st.sidebar.slider(
    "이탈 위험도 임계값", 0, 100, APP_CONFIG["default_risk_threshold"]
)

# 데이터 로드 함수
@st.cache_resource
def load_all_data(use_sample=True):
    if use_sample:
        # 샘플 데이터 생성 (실제 데이터 없을 경우)
        sample_data = generate_sample_data()
        return {
            'sample_data': True,
            'visits_df': sample_data['visits_df'],
            'residence_df': sample_data['residence_df'],
            'weather_df': sample_data['weather_df'],
            'floating_population_df': sample_data.get('floating_population_df', pd.DataFrame()),
            'income_asset_df': sample_data.get('income_asset_df', pd.DataFrame()),
            'card_spending_df': sample_data.get('card_spending_df', pd.DataFrame()),
            'apt_price_df': sample_data.get('apt_price_df', pd.DataFrame()),
            'population_df': sample_data.get('population_df', pd.DataFrame()),
            'female_child_df': sample_data.get('female_child_df', pd.DataFrame()),
            'risk_customers': sample_data['risk_customers']
        }
    else:
        # 각 데이터베이스에 맞는 Snowflake 연결 생성
        loplat_conn = get_snowflake_connection(LOPLAT_CONFIG)
        residence_conn = get_snowflake_connection(RESIDENCE_CONFIG)
        weather_conn = get_snowflake_connection(WEATHER_CONFIG)
        sph_conn = get_snowflake_connection(SPH_CONFIG)
        dataknows_conn = get_snowflake_connection(DATAKNOWS_CONFIG)
        
        # 데이터 로드 - 각 데이터 소스에 맞는 연결 사용
        # 1. 백화점 방문 데이터 (LOPLAT)
        visits_df = load_department_store_visits(
            loplat_conn, 
            APP_CONFIG["date_range"]["start"], 
            APP_CONFIG["date_range"]["end"], 
            selected_store if selected_store != "전체" else None
        )
        
        # 2. 주거지 & 근무지 데이터 (LOPLAT)
        residence_df = load_residence_workplace_data(residence_conn)
        
        # 3. 날씨 데이터 (LOPLAT)
        weather_df = load_weather_data(weather_conn, APP_CONFIG["date_range"]["start"], APP_CONFIG["date_range"]["end"])
        
        # 4. 유동인구 데이터 (SPH - SKT)
        floating_population_df = load_floating_population_data(sph_conn)
        
        # 5. 자산소득 데이터 (SPH - KCB)
        income_asset_df = load_income_asset_data(sph_conn)
        
        # 6. 카드소비내역 데이터 (SPH - 신한카드)
        card_spending_df = load_card_spending_data(sph_conn)
        
        # 7. 아파트 평균 시세 데이터 (DataKnows)
        apartment_price_df = load_apartment_price_data(dataknows_conn)
        
        # 8. 인구 데이터 (DataKnows)
        population_df = load_population_data(dataknows_conn)
        
        # 9. 20~40세 여성 및 영유아 인구 데이터 (DataKnows)
        female_child_df = load_female_child_data(dataknows_conn)
        
        # 10. 행정동경계 데이터 (SPH)
        admin_boundary_df = load_administrative_boundary(sph_conn)
        
        # 11. 지역 마스터 데이터 (SPH - 추가)
        region_master_df = load_region_master_data(sph_conn)
        
        return {
            'sample_data': False,
            'visits_df': visits_df,
            'residence_df': residence_df,
            'weather_df': weather_df,
            'floating_population_df': floating_population_df,
            'income_asset_df': income_asset_df,
            'card_spending_df': card_spending_df,
            'apartment_price_df': apartment_price_df,
            'population_df': population_df,
            'female_child_df': female_child_df,
            'admin_boundary_df': admin_boundary_df,
            'region_master_df': region_master_df
        }

# 메인 프로세스
def main():
    # 탭 설정
    tab0, tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "대시보드 요약", "이탈 위험 모니터링", "고객 세그먼트 분석", 
        "방문 패턴 분석", "주거지 및 소득 분석", "맞춤형 대응 전략"
    ])
    
    # 데이터 로드 버튼
    if not st.session_state.data_loaded:
        use_sample = st.checkbox("샘플 데이터 사용 (Snowflake 연결 없이 테스트)", value=True)
        
        if st.button("데이터 로드 및 분석 시작"):
            try:
                with st.spinner("데이터를 로드하고 분석 중입니다..."):
                    data = load_all_data(use_sample)
                    
                    # 세션 상태에 데이터 저장
                    for key, value in data.items():
                        st.session_state[key] = value
                    
                    # 데이터 분석 수행
                    if 'visits_df' in data and 'residence_df' in data and 'weather_df' in data:
                        # 1. 방문 패턴 분석
                        visit_patterns = analyze_visit_patterns(data['visits_df'])
                        st.session_state.visit_patterns = visit_patterns
                        
                        # 2. 주거지 및 근무지 분석
                        residence_workplace = analyze_residence_workplace(data['residence_df'])
                        st.session_state.residence_workplace = residence_workplace
                        
                        # 3. 날씨 영향 분석
                        weather_impact = analyze_weather_impact(data['visits_df'], data['weather_df'])
                        st.session_state.weather_impact = weather_impact
                        
                        # 4. 소득 및 자산 분석 (데이터가 있는 경우)
                        if 'income_asset_df' in data and not data['income_asset_df'].empty:
                            income_assets = analyze_income_assets(data['income_asset_df'], data['residence_df'])
                            st.session_state.income_assets = income_assets
                        else:
                            st.session_state.income_assets = {'income_distribution': pd.DataFrame()}
                        
                        # 5. 부동산 시세 분석 (데이터가 있는 경우)
                        if 'apt_price_df' in data and not data['apt_price_df'].empty:
                            property_prices = analyze_property_prices(data['apt_price_df'], data['residence_df'])
                            st.session_state.property_prices = property_prices
                        else:
                            st.session_state.property_prices = {
                                'district_prices': pd.DataFrame({'SGG': ['서초구', '강남구', '영등포구', '중구'], 
                                                                'avg_jeonse': [5000, 5500, 4000, 3800],
                                                                'avg_meme': [8000, 9000, 6500, 6000]})
                            }
                        
                        # 6. 고객 세그먼트 생성
                        segments = create_customer_segments(
                            visit_patterns, residence_workplace, weather_impact, 
                            st.session_state.income_assets, st.session_state.property_prices
                        )
                        st.session_state.segments = segments
                        
                        # 7. 이탈 위험 분석 - 개선된 데이터 기반 이탈 예측 사용
                        churn_risk = predict_churn_risk(segments, visit_patterns, weather_impact)
                        st.session_state.churn_risk = churn_risk
                        
                        st.session_state.analysis_done = True
                    
                    st.session_state.data_loaded = True
                
                st.success("데이터 로드 및 분석이 완료되었습니다!")
                st.rerun()
                
            except Exception as e:
                # 오류가 발생하면 자세한 오류 내용을 표시하고 샘플 데이터로 전환하지 않음
                st.error(f"데이터 로드 중 오류가 발생했습니다: {str(e)}")
                st.error("Snowflake 연결 정보를 확인하거나 샘플 데이터를 사용하세요.")
                
                # 오류 상세 내용 표시
                with st.expander("오류 상세 정보"):
                    import traceback
                    st.code(traceback.format_exc())
    
    # 데이터 로드 및 분석이 완료된 경우 각 탭에 내용 표시
    if st.session_state.data_loaded and st.session_state.analysis_done:
        # 대시보드 요약 탭
        with tab0:
            # 모든 데이터를 딕셔너리로 전달
            data_dict = {
                'visits_df': st.session_state.visits_df,
                'residence_df': st.session_state.residence_df,
                'weather_df': st.session_state.weather_df,
                'floating_population_df': st.session_state.get('floating_population_df', pd.DataFrame()),
                'income_asset_df': st.session_state.get('income_asset_df', pd.DataFrame()),
                'card_spending_df': st.session_state.get('card_spending_df', pd.DataFrame()),
                'apt_price_df': st.session_state.get('apt_price_df', pd.DataFrame()),
                'segments': st.session_state.segments if 'segments' in st.session_state else None,
                'churn_risk': st.session_state.churn_risk if 'churn_risk' in st.session_state else None
            }
            create_summary_dashboard(data_dict)
            
        # 이탈 위험 모니터링 탭 - 새로운 고급 모니터링 함수 사용
        with tab1:
            display_advanced_risk_monitoring(
                st.session_state.churn_risk,
                selected_store,
                selected_segment,
                risk_threshold
            )
        
        # 고객 세그먼트 분석 탭
        with tab2:
            display_segment_analysis(
                st.session_state.segments,
                selected_store,
                selected_segment
            )
        
        # 방문 패턴 분석 탭
        with tab3:
            display_visit_pattern_analysis(
                st.session_state.visit_patterns,
                st.session_state.weather_impact,
                selected_store if selected_store != "전체" else None
            )
        
        # 주거지 및 소득 분석 탭
        with tab4:
            display_residence_income_analysis(
                st.session_state.residence_workplace,
                st.session_state.income_assets,
                st.session_state.property_prices,
                selected_store if selected_store != "전체" else None
            )
        
        # 맞춤형 대응 전략 탭
        with tab5:
            display_retention_strategies(
                st.session_state.churn_risk,
                st.session_state.segments,
                selected_store,
                selected_segment
            )
    else:
        if not st.session_state.data_loaded:
            st.info("시작하려면 '데이터 로드 및 분석 시작' 버튼을 클릭하세요.")
            st.image("https://img.freepik.com/premium-vector/shop-logo-template-with-shopping-bag_23-2148720533.jpg", width=300)
            
if __name__ == "__main__":
    main()