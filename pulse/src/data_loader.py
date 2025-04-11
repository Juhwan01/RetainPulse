import pandas as pd
import snowflake.connector
from snowflake.connector.pandas_tools import write_pandas
import streamlit as st
from datetime import datetime, timedelta
import numpy as np

# Snowflake 연결 함수
def get_snowflake_connection(config):
    return snowflake.connector.connect(
        user=config["user"],
        password=config["password"],
        account=config["account"],
        warehouse=config["warehouse"],
        database=config["database"],
        schema=config["schema"]
    )

# 1. 백화점 방문 데이터 로드 (LOPLAT)
@st.cache_data(ttl=3600)
def load_department_store_visits(conn, start_date="2021-01-01", end_date="2024-12-31", store=None):
    # 데이터 딕셔너리에 따라 테이블명 수정
    query = f"""
    SELECT 
        DATE_KST, 
        DEP_NAME, 
        COUNT
    FROM SNOWFLAKE_STREAMLIT_HACKATHON_LOPLAT_DEPARTMENT_STORE_DATA
    WHERE DATE_KST BETWEEN '{start_date}' AND '{end_date}'
    """
    
    if store and store != "전체":
        query += f" AND DEP_NAME = '{store}'"
        
    return pd.read_sql(query, conn)

# 2. 주거지 & 근무지 데이터 로드 (LOPLAT)
@st.cache_data(ttl=3600)
def load_residence_workplace_data(conn):
    # 데이터 딕셔너리에 따라 테이블명 수정
    query = """
    SELECT 
        ADDR_LV1, ADDR_LV2, ADDR_LV3, 
        DEP_NAME, 
        LOC_TYPE, 
        RATIO
    FROM SNOWFLAKE_STREAMLIT_HACKATHON_LOPLAT_HOME_OFFICE_RATIO
    """
    return pd.read_sql(query, conn)

# 3. 날씨 데이터 로드 (LOPLAT)
@st.cache_data(ttl=3600)
def load_weather_data(conn, start_date="2021-01-01", end_date="2024-12-31"):
    # 데이터 딕셔너리에 따라 테이블명 수정
    query = f"""
    SELECT 
        DATE_KST, 
        CITY, 
        AVG_TEMP, 
        RAINFALL_MM
    FROM SNOWFLAKE_STREAMLIT_HACKATHON_LOPLAT_SEOUL_TEMPERATURE_RAINFALL
    WHERE DATE_KST BETWEEN '{start_date}' AND '{end_date}'
    """
    return pd.read_sql(query, conn)

# 4. 유동인구 데이터 로드 (SPH - SKT)
@st.cache_data(ttl=3600)
def load_floating_population_data(conn):
    query = """
    SELECT 
        PROVINCE_CODE, CITY_CODE, DISTRICT_CODE, 
        STANDARD_YEAR_MONTH, 
        AGE_GROUP, GENDER, WEEKDAY_WEEKEND, TIME_SLOT,
        RESIDENTIAL_POPULATION, WORKING_POPULATION, VISITING_POPULATION
    FROM FLOATING_POPULATION_INFO
    """
    return pd.read_sql(query, conn)

# 5. 자산소득 데이터 로드 (SPH - KCB)
@st.cache_data(ttl=3600)
def load_income_asset_data(conn):
    # 데이터 딕셔너리에 따른 주요 컬럼 선택
    query = """
    SELECT 
        STANDARD_YEAR_MONTH,
        PROVINCE_CODE, PROVINCE_KOR_NAME,
        CITY_CODE, CITY_KOR_NAME,
        DISTRICT_CODE, DISTRICT_KOR_NAME,
        RATE_INCOME_UNDER_20M, RATE_INCOME_20M_TO_30M, 
        RATE_INCOME_30M_TO_40M, RATE_INCOME_40M_TO_50M,
        RATE_INCOME_50M_TO_60M, RATE_INCOME_60M_TO_70M,
        RATE_INCOME_OVER_70M, TOTAL_USAGE_AMOUNT, TOTAL_CREDIT_CARD_USAGE_AMOUNT
    FROM ASSET_INCOME_INFO
    """
    return pd.read_sql(query, conn)

# 6. 카드소비내역 데이터 로드 (SPH - 신한카드)
@st.cache_data(ttl=3600)
def load_card_spending_data(conn):
    # 데이터 딕셔너리에 따른 주요 컬럼 선택
    query = """
    SELECT 
        STANDARD_YEAR_MONTH,
        PROVINCE_CODE, CITY_CODE, DISTRICT_CODE,
        TOTAL_SALES, TOTAL_COUNT,
        DEPARTMENT_STORE_SALES, DEPARTMENT_STORE_COUNT,
        FOOD_SALES, CLOTHING_ACCESSORIES_SALES, COFFEE_SALES
    FROM CARD_SALES_INFO
    """
    return pd.read_sql(query, conn)

# 7. 아파트 평균 시세 데이터 로드 (DataKnows)
@st.cache_data(ttl=3600)
def load_apartment_price_data(conn):
    # 데이터 딕셔너리에 따라 테이블명 수정
    query = """
    SELECT 
        BJD_CODE, EMD, 
        JEONSE_PRICE_PER_SUPPLY_PYEONG, 
        MEME_PRICE_PER_SUPPLY_PYEONG,
        REGION_LEVEL, SD, SGG,
        TOTAL_HOUSEHOLDS,
        YYYYMMDD
    FROM REGION_APT_RICHGO_MARKET_PRICE_M_H
    """
    return pd.read_sql(query, conn)

# 8. 인구 데이터 로드 (DataKnows)
@st.cache_data(ttl=3600)
def load_population_data(conn):
    # 데이터 딕셔너리에 따라 테이블명 수정
    query = """
    SELECT 
        BJD_CODE, EMD, SD, SGG, REGION_LEVEL,
        TOTAL, MALE, FEMALE,
        AGE_208, AGE_209, AGE_308, AGE_309, AGE_408, AGE_409,
        YYYYMMDD
    FROM REGION_MOIS_POPULATION_GENDER_AGE_M_H
    """
    return pd.read_sql(query, conn)

# 9. 20~40세 여성 및 영유아 인구 데이터 로드 (DataKnows)
@st.cache_data(ttl=3600)
def load_female_child_data(conn):
    # 데이터 딕셔너리에 따라 테이블명 수정
    query = """
    SELECT 
        BJD_CODE, EMD, SD, SGG, REGION_LEVEL,
        TOTAL, FEMALE_20TO40, AGE_UNDER5,
        YYYYMMDD
    FROM REGION_MOIS_POPULATION_AGE_UNDER5_PER_FEMALE_20TO40_M_H
    """
    return pd.read_sql(query, conn)

# 10. 행정동경계 데이터 로드
@st.cache_data(ttl=3600)
def load_administrative_boundary(conn):
    # 데이터 딕셔너리에 따른 테이블 사용
    query = """
    SELECT 
        CODE_ID, CODE_NAME, SUB_CODE, SUB_CODE_NAME,
        SORT_ORDER, USE_YN
    FROM CODE_MASTER
    """
    return pd.read_sql(query, conn)

# 11. 지역 마스터 데이터 로드 (추가)
@st.cache_data(ttl=3600)
def load_region_master_data(conn):
    # 데이터 딕셔너리에 따른 새 함수 추가
    query = """
    SELECT 
        PROVINCE_CODE, PROVINCE_KOR_NAME,
        CITY_CODE, CITY_KOR_NAME,
        DISTRICT_CODE, DISTRICT_KOR_NAME
    FROM M_SCCO_MST
    """
    return pd.read_sql(query, conn)

# 샘플 데이터 생성 함수 (실제 데이터 없을 경우 대체용)
def generate_sample_data():
    # 현재 날짜 기준 샘플 데이터 생성
    current_date = datetime.now()
    stores = ["신세계 강남", "더현대 서울", "롯데 본점"]
    
    # 1. 백화점 방문 데이터 샘플
    date_range = pd.date_range(start='2021-01-01', end='2024-12-31', freq='D')
    
    visits_sample = pd.DataFrame({
        "DATE_KST": np.repeat(date_range, 3),
        "DEP_NAME": np.tile(stores, len(date_range)),
        "COUNT": np.random.randint(1000, 5000, len(date_range) * 3)
    })
    
    # 2. 주거지 & 근무지 데이터 샘플
    districts = ["강남구", "서초구", "영등포구", "중구", "종로구"]
    dongs = ["서초동", "반포동", "여의도동", "명동", "신당동", "역삼동", "삼성동", "종로1가"]
    
    residence_sample_home = pd.DataFrame({
        "ADDR_LV1": "서울특별시",
        "ADDR_LV2": np.random.choice(districts, 100),
        "ADDR_LV3": np.random.choice(dongs, 100),
        "DEP_NAME": np.random.choice(stores, 100),
        "LOC_TYPE": 1,  # 주거지
        "RATIO": np.random.uniform(0.01, 0.2, 100)
    })
    
    residence_sample_work = pd.DataFrame({
        "ADDR_LV1": "서울특별시",
        "ADDR_LV2": np.random.choice(districts, 100),
        "ADDR_LV3": np.random.choice(dongs, 100),
        "DEP_NAME": np.random.choice(stores, 100),
        "LOC_TYPE": 2,  # 근무지
        "RATIO": np.random.uniform(0.01, 0.2, 100)
    })
    
    residence_sample = pd.concat([residence_sample_home, residence_sample_work])
    
    # 3. 날씨 데이터 샘플
    weather_sample = pd.DataFrame({
        "DATE_KST": date_range,
        "CITY": "서울",
        "AVG_TEMP": np.sin(np.linspace(0, 4*np.pi, len(date_range))) * 15 + 15,  # -5 ~ 30도 사이 계절성 온도
        "RAINFALL_MM": np.random.exponential(scale=2, size=len(date_range))  # 지수분포로 강수량 생성
    })
    
    # 비가 오는 날 확률적으로 선택 (약 20%)
    rainy_days = np.random.choice([True, False], len(date_range), p=[0.2, 0.8])
    weather_sample.loc[~rainy_days, "RAINFALL_MM"] = 0
    
    # 4. 유동인구 데이터 샘플 (데이터 딕셔너리에 맞게 조정)
    months = pd.date_range(start='2021-01-01', end='2023-12-31', freq='M').strftime("%Y%m").tolist()
    age_groups = ["20대", "30대", "40대", "50대"]
    genders = ["남성", "여성"]
    time_slots = ["아침", "오전", "점심", "오후", "저녁", "밤"]
    weekday_weekend = ["평일", "주말"]
    
    num_floating_samples = 1000
    
    floating_population_sample = pd.DataFrame({
        "PROVINCE_CODE": ["11"] * num_floating_samples,  # 서울
        "CITY_CODE": np.random.choice([f"11{i:02d}" for i in range(1, 6)], num_floating_samples),
        "DISTRICT_CODE": np.random.choice([f"11{i:05d}" for i in range(1, 20)], num_floating_samples),
        "STANDARD_YEAR_MONTH": np.random.choice(months, num_floating_samples),
        "AGE_GROUP": np.random.choice(age_groups, num_floating_samples),
        "GENDER": np.random.choice(genders, num_floating_samples),
        "WEEKDAY_WEEKEND": np.random.choice(weekday_weekend, num_floating_samples),
        "TIME_SLOT": np.random.choice(time_slots, num_floating_samples),
        "RESIDENTIAL_POPULATION": np.random.randint(1000, 10000, num_floating_samples),
        "WORKING_POPULATION": np.random.randint(1000, 8000, num_floating_samples),
        "VISITING_POPULATION": np.random.randint(500, 5000, num_floating_samples)
    })
    
    # 5. 자산소득 데이터 샘플 (데이터 딕셔너리에 맞게 조정)
    # 길이 문제 해결을 위해 배열 길이를 모두 동일하게 설정
    num_income_samples = len(months) * len(districts)
    
    # 길이가 일치하는 배열 준비
    month_array = np.repeat(months, len(districts))
    district_array = np.tile(districts, len(months))
    
    # 동 이름 배열 준비 (길이 맞추기)
    dong_repeats = int(np.ceil(num_income_samples / len(dongs)))
    dong_array = np.tile(dongs, dong_repeats)[:num_income_samples]
    
    # 구/동 코드 배열 준비
    district_codes = [f"11{i:02d}" for i in range(1, len(districts) + 1)]
    district_code_array = np.repeat(district_codes, len(months))[:num_income_samples]
    
    dong_codes = [f"11{i:05d}" for i in range(1, len(dongs) + 1)]
    dong_code_repeats = int(np.ceil(num_income_samples / len(dong_codes)))
    dong_code_array = np.tile(dong_codes, dong_code_repeats)[:num_income_samples]
    
    income_asset_sample = pd.DataFrame({
        "STANDARD_YEAR_MONTH": month_array,
        "PROVINCE_CODE": ["11"] * num_income_samples,  # 서울
        "PROVINCE_KOR_NAME": ["서울특별시"] * num_income_samples,
        "CITY_CODE": district_code_array,
        "CITY_KOR_NAME": district_array,
        "DISTRICT_CODE": dong_code_array,
        "DISTRICT_KOR_NAME": dong_array,
        "RATE_INCOME_UNDER_20M": np.random.uniform(0.2, 0.4, num_income_samples),
        "RATE_INCOME_20M_TO_30M": np.random.uniform(0.15, 0.3, num_income_samples),
        "RATE_INCOME_30M_TO_40M": np.random.uniform(0.1, 0.2, num_income_samples),
        "RATE_INCOME_40M_TO_50M": np.random.uniform(0.05, 0.15, num_income_samples),
        "RATE_INCOME_50M_TO_60M": np.random.uniform(0.03, 0.1, num_income_samples),
        "RATE_INCOME_60M_TO_70M": np.random.uniform(0.02, 0.07, num_income_samples),
        "RATE_INCOME_OVER_70M": np.random.uniform(0.01, 0.05, num_income_samples),
        "TOTAL_USAGE_AMOUNT": np.random.randint(100000000, 500000000, num_income_samples),
        "TOTAL_CREDIT_CARD_USAGE_AMOUNT": np.random.randint(80000000, 400000000, num_income_samples),
    })
    
    # 6. 카드소비내역 데이터 샘플 (데이터 딕셔너리에 맞게 조정)
    # 소비 데이터도 같은 크기로 생성
    num_card_samples = num_income_samples
    
    card_spending_sample = pd.DataFrame({
        "STANDARD_YEAR_MONTH": month_array[:num_card_samples],
        "PROVINCE_CODE": ["11"] * num_card_samples,  # 서울
        "CITY_CODE": district_code_array[:num_card_samples],
        "DISTRICT_CODE": dong_code_array[:num_card_samples],
        "TOTAL_SALES": np.random.randint(500000000, 2000000000, num_card_samples),
        "TOTAL_COUNT": np.random.randint(50000, 200000, num_card_samples),
        "DEPARTMENT_STORE_SALES": np.random.randint(100000000, 500000000, num_card_samples),
        "DEPARTMENT_STORE_COUNT": np.random.randint(5000, 50000, num_card_samples),
        "FOOD_SALES": np.random.randint(50000000, 200000000, num_card_samples),
        "CLOTHING_ACCESSORIES_SALES": np.random.randint(80000000, 300000000, num_card_samples),
        "COFFEE_SALES": np.random.randint(20000000, 80000000, num_card_samples)
    })
    
    # 7. 아파트 시세 데이터 샘플 (데이터 딕셔너리에 맞게 조정)
    # 60개월 * 동 개수만큼의 샘플 생성
    months_apt = pd.date_range(start='2020-01-01', end='2024-12-31', freq='M').strftime("%Y%m%d").tolist()
    num_apt_samples = len(months_apt) * len(dongs)
    
    # 배열 준비
    month_apt_array = np.repeat(months_apt, len(dongs))
    dong_apt_array = np.tile(dongs, len(months_apt))
    
    # 구 배열 준비 (각 동마다 구 할당)
    sgg_mapping = {}
    for i, dong in enumerate(dongs):
        sgg_mapping[dong] = districts[i % len(districts)]
    
    sgg_array = [sgg_mapping[dong] for dong in dong_apt_array]
    
    # BJD_CODE 배열 준비
    bjd_codes = [f"11{i:05d}" for i in range(1, len(dongs) + 1)]
    bjd_code_array = np.tile(bjd_codes, len(months_apt))
    
    apt_price_sample = pd.DataFrame({
        "BJD_CODE": bjd_code_array,
        "EMD": dong_apt_array,
        "JEONSE_PRICE_PER_SUPPLY_PYEONG": np.random.uniform(1000, 3000, num_apt_samples) * 10000,  # 전세가
        "MEME_PRICE_PER_SUPPLY_PYEONG": np.random.uniform(2500, 7000, num_apt_samples) * 10000,  # 매매가
        "REGION_LEVEL": ["법정동"] * num_apt_samples,
        "SD": ["서울특별시"] * num_apt_samples,
        "SGG": sgg_array,
        "TOTAL_HOUSEHOLDS": np.random.randint(5000, 20000, num_apt_samples),
        "YYYYMMDD": month_apt_array
    })
    
    # 8. 인구 데이터 샘플 (데이터 딕셔너리에 맞게 조정)
    population_sample = pd.DataFrame({
        "BJD_CODE": [f"11{i:05d}" for i in range(1, len(dongs) + 1)],
        "EMD": dongs[:len(dongs)],
        "SD": ["서울특별시"] * len(dongs),
        "SGG": [sgg_mapping[dong] for dong in dongs[:len(dongs)]],
        "REGION_LEVEL": ["법정동"] * len(dongs),
        "TOTAL": np.random.randint(10000, 50000, len(dongs)),
        "MALE": np.random.randint(5000, 25000, len(dongs)),
        "FEMALE": np.random.randint(5000, 25000, len(dongs)),
        "AGE_208": np.random.randint(1000, 5000, len(dongs)),
        "AGE_209": np.random.randint(1000, 5000, len(dongs)),
        "AGE_308": np.random.randint(1000, 5000, len(dongs)),
        "AGE_309": np.random.randint(1000, 5000, len(dongs)),
        "AGE_408": np.random.randint(1000, 5000, len(dongs)),
        "AGE_409": np.random.randint(1000, 5000, len(dongs)),
        "YYYYMMDD": ["20250101"] * len(dongs)
    })
    
    # 9. 20~40세 여성 및 영유아 인구 데이터 샘플
    female_child_sample = pd.DataFrame({
        "BJD_CODE": [f"11{i:05d}" for i in range(1, len(dongs) + 1)],
        "EMD": dongs[:len(dongs)],
        "SD": ["서울특별시"] * len(dongs),
        "SGG": [sgg_mapping[dong] for dong in dongs[:len(dongs)]],
        "REGION_LEVEL": ["법정동"] * len(dongs),
        "TOTAL": np.random.randint(10000, 50000, len(dongs)),
        "FEMALE_20TO40": np.random.randint(3000, 15000, len(dongs)),
        "AGE_UNDER5": np.random.randint(500, 3000, len(dongs)),
        "YYYYMMDD": ["20250101"] * len(dongs)
    })
    
    # 10. 이탈 위험 고객 샘플 데이터
    risk_customers = pd.DataFrame({
        "고객_ID": [f"C{i:05d}" for i in range(10)],
        "거주_지역": np.random.choice([f"{d1} {d2}" for d1, d2 in zip(np.repeat(districts, 2), dongs[:len(districts)*2])], 10),
        "마지막_방문일": [(current_date - timedelta(days=np.random.randint(60, 120))).strftime("%Y-%m-%d") for _ in range(10)],
        "이탈_위험도": np.random.randint(70, 96, 10),
        "선호_백화점": np.random.choice(stores, 10),
        "세그먼트": np.random.choice(["VIP 고객", "정기 방문 고객", "간헐적 방문 고객", "이탈 위험 고객"], 10, p=[0.1, 0.2, 0.3, 0.4])
    })
    
    return {
        'visits_df': visits_sample,
        'residence_df': residence_sample,
        'weather_df': weather_sample,
        'floating_population_df': floating_population_sample,
        'income_asset_df': income_asset_sample,
        'card_spending_df': card_spending_sample,
        'apt_price_df': apt_price_sample,
        'population_df': population_sample,
        'female_child_df': female_child_sample,
        'risk_customers': risk_customers
    }