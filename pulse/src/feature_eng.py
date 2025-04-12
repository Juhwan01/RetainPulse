import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler
import haversine as hs  # 위경도 거리 계산용 (pip install haversine)

# 백화점 방문 패턴 분석
def analyze_visit_patterns(visits_df):
    # 일별 방문 데이터를 고객 수준으로 변환 (실제 데이터에는 고객 ID가 있겠지만, 
    # 현재 데이터에서는 백화점별 집계 데이터이므로 여기서는 매장 수준 분석)
    
    # 날짜 형식 변환 - 이미 datetime 형식인지 확인 후 변환
    visits_df = visits_df.copy()  # 원본 데이터 변경 방지를 위해 복사본 사용
    
    if not pd.api.types.is_datetime64_any_dtype(visits_df['DATE_KST']):
        visits_df['DATE_KST'] = pd.to_datetime(visits_df['DATE_KST'])
    
    # 요일 및 월 추가
    if 'dayofweek' not in visits_df.columns:
        visits_df['dayofweek'] = visits_df['DATE_KST'].dt.dayofweek
    if 'month' not in visits_df.columns:
        visits_df['month'] = visits_df['DATE_KST'].dt.month
    if 'year' not in visits_df.columns:
        visits_df['year'] = visits_df['DATE_KST'].dt.year
    
    # 백화점별 방문 트렌드
    store_trends = visits_df.groupby(['DEP_NAME', 'year', 'month']).agg(
        avg_daily_visits=('COUNT', 'mean'),
        total_visits=('COUNT', 'sum'),
        max_visits=('COUNT', 'max'),
        min_visits=('COUNT', 'min')
    ).reset_index()
    
    # 요일별 방문 패턴
    dow_patterns = visits_df.groupby(['DEP_NAME', 'dayofweek']).agg(
        avg_visits=('COUNT', 'mean')
    ).reset_index()
    
    # 계절성 분석
    visits_df['season'] = visits_df['month'].apply(
        lambda x: 'Winter' if x in [12, 1, 2] else 
                  'Spring' if x in [3, 4, 5] else
                  'Summer' if x in [6, 7, 8] else 'Fall'
    )
    
    seasonal_patterns = visits_df.groupby(['DEP_NAME', 'season']).agg(
        avg_visits=('COUNT', 'mean')
    ).reset_index()
    
    return {
        'store_trends': store_trends,
        'dow_patterns': dow_patterns,
        'seasonal_patterns': seasonal_patterns,
        'processed_visits': visits_df
    }

# 주거지 & 근무지 데이터 분석
def analyze_residence_workplace(residence_df):
    # 주거지와 근무지 데이터 분리
    home_data = residence_df[residence_df['LOC_TYPE'] == 1]
    work_data = residence_df[residence_df['LOC_TYPE'] == 2]
    
    # 백화점별 주거지 분포
    home_distribution = home_data.groupby(['DEP_NAME', 'ADDR_LV2', 'ADDR_LV3']).agg(
        ratio=('RATIO', 'sum')
    ).reset_index()
    
    # 백화점별 근무지 분포
    work_distribution = work_data.groupby(['DEP_NAME', 'ADDR_LV2', 'ADDR_LV3']).agg(
        ratio=('RATIO', 'sum')
    ).reset_index()
    
    # 백화점별 주거지 Top 10
    top_home_areas = home_distribution.sort_values(['DEP_NAME', 'ratio'], ascending=[True, False])
    top_home_areas = top_home_areas.groupby('DEP_NAME').head(10).reset_index(drop=True)
    
    # 백화점별 근무지 Top 10
    top_work_areas = work_distribution.sort_values(['DEP_NAME', 'ratio'], ascending=[True, False])
    top_work_areas = top_work_areas.groupby('DEP_NAME').head(10).reset_index(drop=True)
    
    return {
        'home_distribution': home_distribution,
        'work_distribution': work_distribution,
        'top_home_areas': top_home_areas,
        'top_work_areas': top_work_areas
    }

# 날씨와 방문 연관성 분석
def analyze_weather_impact(visits_df, weather_df):
    # 날짜 형식 변환
    if not pd.api.types.is_datetime64_any_dtype(visits_df['DATE_KST']):
        visits_df['DATE_KST'] = pd.to_datetime(visits_df['DATE_KST'])
    if not pd.api.types.is_datetime64_any_dtype(weather_df['DATE_KST']):
        weather_df['DATE_KST'] = pd.to_datetime(weather_df['DATE_KST'])
    # 날씨 데이터와 방문 데이터 병합
    visit_weather = pd.merge(
        visits_df,
        weather_df,
        on='DATE_KST',
        how='left'
    )
    
    # 기온 구간 생성
    visit_weather['temp_range'] = pd.cut(
        visit_weather['AVG_TEMP'],
        bins=[-30, 0, 10, 20, 30, 40],
        labels=['매우 추움', '추움', '적정', '따뜻함', '더움']
    )
    
    # 강수량 구간 생성
    visit_weather['rain_range'] = pd.cut(
        visit_weather['RAINFALL_MM'],
        bins=[-0.1, 0, 5, 20, 50, 200],
        labels=['맑음', '약한 비', '보통 비', '강한 비', '폭우']
    )
    
    # 날씨 조건별 방문 집계
    weather_impact = visit_weather.groupby(['DEP_NAME', 'temp_range', 'rain_range']).agg(
        avg_visits=('COUNT', 'mean'),
        total_visits=('COUNT', 'sum'),
        visit_days=('DATE_KST', 'count')
    ).reset_index()
    
    # 기온과 방문 상관관계
    temp_correlation = visit_weather.groupby('DEP_NAME').apply(
        lambda x: pd.Series({
            'temp_correlation': x['AVG_TEMP'].corr(x['COUNT']),
            'rain_correlation': x['RAINFALL_MM'].corr(x['COUNT'])
        })
    ).reset_index()
    
    return {
        'visit_weather': visit_weather,
        'weather_impact': weather_impact,
        'temp_correlation': temp_correlation
    }

# 소득 및 자산 데이터 분석
def analyze_income_assets(income_asset_df, residence_df):
    # 주거지 데이터에서 각 법정동별 방문객 비율 추출
    home_data = residence_df[residence_df['LOC_TYPE'] == 1]
    
    # 소득 구간별 환산을 위한 가중치 (중간값 사용)
    income_weights = {
        'RATE_INCOME_UNDER_20M': 15,
        'RATE_INCOME_20M_TO_30M': 25,
        'RATE_INCOME_30M_TO_40M': 35,
        'RATE_INCOME_40M_TO_50M': 45,
        'RATE_INCOME_50M_TO_60M': 55,
        'RATE_INCOME_60M_TO_70M': 65,
        'RATE_INCOME_OVER_70M': 80
    }
    
    # 각 동별 소득 분포 및 추정 평균 소득 계산
    income_distribution = pd.DataFrame()
    
    if not income_asset_df.empty:
        # 법정동 - 행정동 매핑 확인 (실제로는 매핑 테이블이 필요할 수 있음)
        
        # 평균 소득 계산 (가중 평균)
        for col, weight in income_weights.items():
            if col in income_asset_df.columns:
                if 'est_income' not in income_asset_df.columns:
                    income_asset_df['est_income'] = 0
                income_asset_df['est_income'] += income_asset_df[col] * weight
        
        # 최근 데이터만 선택 (마지막 년월)
        if 'STANDARD_YEAR_MONTH' in income_asset_df.columns:
            latest_month = income_asset_df['STANDARD_YEAR_MONTH'].max()
            latest_income = income_asset_df[income_asset_df['STANDARD_YEAR_MONTH'] == latest_month]
            
            # 동별 소득 정보
            if 'DISTRICT_KOR_NAME' in latest_income.columns:
                income_distribution = latest_income[['DISTRICT_KOR_NAME', 'est_income'] + list(income_weights.keys())]
                income_distribution = income_distribution.rename(columns={'DISTRICT_KOR_NAME': 'DONG_NAME'})
            else:
                # 대체 컬럼 사용
                dong_col = next((col for col in ['DONG_NAME', 'DISTRICT_NAME', 'EMD'] 
                                if col in latest_income.columns), None)
                if dong_col:
                    income_distribution = latest_income[[dong_col, 'est_income'] + list(income_weights.keys())]
                    income_distribution = income_distribution.rename(columns={dong_col: 'DONG_NAME'})
    
    return {
        'income_distribution': income_distribution,
        'latest_income_data': latest_income if 'latest_income' in locals() else None
    }

# 아파트 시세 데이터 분석
def analyze_property_prices(apt_price_df, residence_df):
    # 데이터 딕셔너리에 맞게 YYYYMMDD 컬럼 처리
    if 'YYYYMMDD' in apt_price_df.columns:
        # 날짜 형식 변환
        apt_price_df['date'] = pd.to_datetime(apt_price_df['YYYYMMDD'], format='%Y%m%d')
        apt_price_df['year_month'] = apt_price_df['date'].dt.strftime('%Y-%m')
        
        # 최근 시세 데이터 선택
        latest_date = apt_price_df['date'].max()
        latest_prices = apt_price_df[apt_price_df['date'] == latest_date]
    else:
        latest_prices = apt_price_df
    
    # 구별 평균 시세
    district_prices = latest_prices.groupby('SGG').agg(
        avg_jeonse=('JEONSE_PRICE_PER_SUPPLY_PYEONG', 'mean'),
        avg_meme=('MEME_PRICE_PER_SUPPLY_PYEONG', 'mean'),
        total_households=('TOTAL_HOUSEHOLDS', 'sum')
    ).reset_index()
    
    # 동별 평균 시세
    dong_prices = latest_prices.groupby('EMD').agg(
        avg_jeonse=('JEONSE_PRICE_PER_SUPPLY_PYEONG', 'mean'),
        avg_meme=('MEME_PRICE_PER_SUPPLY_PYEONG', 'mean'),
        total_households=('TOTAL_HOUSEHOLDS', 'sum')
    ).reset_index()
    
    # 시계열 트렌드 분석 (구별)
    if 'year_month' in apt_price_df.columns:
        price_trends = apt_price_df.groupby(['SGG', 'year_month']).agg(
            avg_meme=('MEME_PRICE_PER_SUPPLY_PYEONG', 'mean')
        ).reset_index()
        
        # 피벗 테이블로 변환 (구별 시계열)
        price_trend_pivot = price_trends.pivot(index='year_month', columns='SGG', values='avg_meme')
    else:
        price_trend_pivot = pd.DataFrame()
    
    return {
        'district_prices': district_prices,
        'dong_prices': dong_prices,
        'price_trends': price_trend_pivot if 'price_trend_pivot' in locals() else pd.DataFrame()
    }

# 종합 분석 및 고객 세그먼트 추정
def create_customer_segments(visit_patterns, residence_workplace, weather_impact, income_assets, property_prices):
    # 백화점별 주요 고객 세그먼트 추정
    segments = {}
    
    # 각 백화점별 분석
    for store in visit_patterns['store_trends']['DEP_NAME'].unique():
        # 1. 방문 패턴
        store_visit = visit_patterns['store_trends'][visit_patterns['store_trends']['DEP_NAME'] == store]
        
        # 2. 주거지 분포
        home_dist = residence_workplace['top_home_areas'][residence_workplace['top_home_areas']['DEP_NAME'] == store]
        
        # 3. 날씨 영향
        weather = weather_impact['temp_correlation'][weather_impact['temp_correlation']['DEP_NAME'] == store]
        
        # 종합 분석을 통한 세그먼트 정의
        segments[store] = {
            'visit_pattern': store_visit.to_dict('records'),
            'main_residential_areas': home_dist.to_dict('records'),
            'weather_sensitivity': weather.to_dict('records')
        }
    
    # 소득 수준별 세그먼트 (실제 고객 데이터가 있다면 더 세밀한 세분화 가능)
    income_segments = ['고소득층', '중상위층', '중산층', '일반소비자']
    segment_distribution = {
        '신세계 강남': [0.25, 0.35, 0.30, 0.10],
        '더현대 서울': [0.20, 0.30, 0.35, 0.15],
        '롯데 본점': [0.15, 0.25, 0.40, 0.20]
    }
    
    # 세그먼트별 이탈 위험도 (예시)
    churn_risk = {
        '고소득층': 0.15,
        '중상위층': 0.22,
        '중산층': 0.28, 
        '일반소비자': 0.35
    }
    
    # 각 세그먼트별 특성 정의
    segment_profiles = {
        '고소득층': {
            'avg_income': '7000만원 이상',
            'main_residence': ['서초구', '강남구'],
            'visit_frequency': '월 3회 이상',
            'avg_purchase': '30만원 이상',
            'churn_risk': churn_risk['고소득층']
        },
        '중상위층': {
            'avg_income': '5000~7000만원',
            'main_residence': ['강남구', '영등포구', '용산구'],
            'visit_frequency': '월 2회',
            'avg_purchase': '20~30만원',
            'churn_risk': churn_risk['중상위층']
        },
        '중산층': {
            'avg_income': '3000~5000만원',
            'main_residence': ['영등포구', '중구', '마포구'],
            'visit_frequency': '월 1회',
            'avg_purchase': '10~20만원',
            'churn_risk': churn_risk['중산층']
        },
        '일반소비자': {
            'avg_income': '3000만원 미만',
            'main_residence': ['중구', '종로구', '기타'],
            'visit_frequency': '분기 1~2회',
            'avg_purchase': '10만원 미만',
            'churn_risk': churn_risk['일반소비자']
        }
    }
    
    # 최종 결과
    return {
        'store_segments': segments,
        'income_segments': income_segments,
        'segment_distribution': segment_distribution,
        'segment_profiles': segment_profiles
    }

# 이탈 위험 고객 분석 및 예측
def predict_churn_risk(segments, visit_patterns, weather_impact):
    """
    이탈 위험 고객 분석 및 예측
    
    segments: 고객 세그먼트 정보
    visit_patterns: 방문 패턴 데이터
    weather_impact: 날씨 영향 데이터
    
    반환: 기존 형식과 호환되는 이탈 위험 정보 딕셔너리
    """
    # 사용할 모듈 import
    from src.model import advanced_churn_prediction
    
    # 개선된 이탈 예측 함수 호출
    churn_risk = advanced_churn_prediction(visit_patterns, None, weather_impact)
    
    return churn_risk

# 종합 데이터 모델링을 위한 데이터셋 준비
def prepare_integrated_modeling_dataset(
    visit_patterns, residence_workplace, weather_impact, 
    income_assets, property_prices, segments, churn_risk
):
    # 백화점별 데이터 통합
    store_data = {}
    
    for store in visit_patterns['store_trends']['DEP_NAME'].unique():
        # 1. 방문 패턴
        visits = visit_patterns['processed_visits'][visit_patterns['processed_visits']['DEP_NAME'] == store]
        
        # 2. 주거지 분포
        residence = residence_workplace['home_distribution'][residence_workplace['home_distribution']['DEP_NAME'] == store]
        
        # 3. 날씨 영향
        weather = weather_impact['visit_weather'][weather_impact['visit_weather']['DEP_NAME'] == store]
        
        # 4. 세그먼트 분포
        store_segments = {
            'segments': segments['segment_distribution'].get(store, {}),
            'profiles': segments['segment_profiles']
        }
        
        # 5. 이탈 위험
        risk = churn_risk['store_risk'].get(store, {})
        
        # 통합 데이터셋
        store_data[store] = {
            'visits': visits,
            'residence': residence,
            'weather': weather,
            'segments': store_segments,
            'risk': risk
        }
    
    # 이탈 예측을 위한 특성 데이터셋 (예시 - 실제는 고객 수준 데이터 필요)
    prediction_features = pd.DataFrame()
    
    # 최종 모델링 데이터셋 (샘플)
    modeling_data = pd.DataFrame({
        'store': np.repeat(list(store_data.keys()), 4),
        'segment': np.tile(list(segments['segment_profiles'].keys()), len(store_data)),
        'visit_frequency': [3, 2, 1, 0.5] * len(store_data),  # 방문 빈도 (예시)
        'avg_purchase': [35, 25, 15, 8] * len(store_data),    # 평균 구매액 (예시)
        'weather_sensitivity': [0.2, 0.3, 0.4, 0.5] * len(store_data),  # 날씨 민감도 (예시)
        'distance_sensitivity': [0.3, 0.4, 0.5, 0.6] * len(store_data),  # 거리 민감도 (예시)
        'churn_risk': [
            *[churn_risk['store_risk'].get(store, {}).get(segment, 0.3) 
              for store in store_data.keys()
              for segment in segments['segment_profiles'].keys()]
        ]
    })
    
    return {
        'store_data': store_data,
        'modeling_data': modeling_data
    }