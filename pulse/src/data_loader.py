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
    query = f"""
    SELECT 
        DATE_KST, 
        DEP_NAME, 
        COUNT
    FROM SNOWFLAKE_STREAMLIT_HACKATHON_DEPARTMENT_STORE_FOOT_TRAFFIC
    WHERE DATE_KST BETWEEN '{start_date}' AND '{end_date}'
    """
    
    if store and store != "전체":
        query += f" AND DEP_NAME = '{store}'"
        
    return pd.read_sql(query, conn)

# 2. 주거지 & 근무지 데이터 로드 (LOPLAT)
@st.cache_data(ttl=3600)
def load_residence_workplace_data(conn):
    query = """
    SELECT 
        ADDR_LV1, ADDR_LV2, ADDR_LV3, 
        DEP_NAME, 
        LOC_TYPE, 
        RATIO
    FROM SNOWFLAKE_STREAMLIT_HACKATHON_RESIDENCE_WORKPLACE
    """
    return pd.read_sql(query, conn)

# 3. 날씨 데이터 로드 (LOPLAT)
@st.cache_data(ttl=3600)
def load_weather_data(conn, start_date="2021-01-01", end_date="2024-12-31"):
    query = f"""
    SELECT 
        DATE_KST, 
        CITY, 
        AVG_TEMP, 
        RAINFALL_MM
    FROM SNOWFLAKE_STREAMLIT_HACKATHON_SEOUL_WEATHER
    WHERE DATE_KST BETWEEN '{start_date}' AND '{end_date}'
    """
    return pd.read_sql(query, conn)

# 4. 유동인구 데이터 로드 (SPH - SKT)
@st.cache_data(ttl=3600)
def load_floating_population_data(conn):
    query = """
    SELECT *
    FROM FLOATING_POPULATION_INFO
    """
    return pd.read_sql(query, conn)

# 5. 자산소득 데이터 로드 (SPH - KCB)
@st.cache_data(ttl=3600)
def load_income_asset_data(conn):
    query = """
    SELECT *
    FROM ASSET_INCOME_INFO
    """
    return pd.read_sql(query, conn)

# 6. 카드소비내역 데이터 로드 (SPH - 신한카드)
@st.cache_data(ttl=3600)
def load_card_spending_data(conn):
    query = """
    SELECT *
    FROM CARD_SALES_INFO
    """
    return pd.read_sql(query, conn)

# 7. 아파트 평균 시세 데이터 로드 (DataKnows)
@st.cache_data(ttl=3600)
def load_apartment_price_data(conn):
    query = """
    SELECT 
        BJD_CODE, EMD, 
        JEONSE_PRICE_PER_SUPPLY_PYEONG, 
        MEME_PRICE_PER_SUPPLY_PYEONG,
        REGION_LEVEL, SD, SGG,
        TOTAL_HOUSEHOLDS,
        YYYYMMDD
    FROM REGION_APT_RICHGO_PRICE
    """
    return pd.read_sql(query, conn)

# 8. 인구 데이터 로드 (DataKnows)
@st.cache_data(ttl=3600)
def load_population_data(conn):
    query = """
    SELECT *
    FROM REGION_MOIS_POPULATION
    """
    return pd.read_sql(query, conn)

# 9. 20~40세 여성 및 영유아 인구 데이터 로드 (DataKnows)
@st.cache_data(ttl=3600)
def load_female_child_data(conn):
    query = """
    SELECT *
    FROM REGION_MOIS_POPULATION_FEMALE_CHILD
    """
    return pd.read_sql(query, conn)

# 10. 행정동경계 데이터 로드
@st.cache_data(ttl=3600)
def load_administrative_boundary(conn):
    query = """
    SELECT *
    FROM CODE_MASTER
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
    
    # 4. 자산소득 데이터 샘플
    income_asset_sample = pd.DataFrame({
        "STANDARD_YEAR_MONTH": np.repeat(pd.date_range(start='2021-01', end='2023-12', freq='M').strftime("%Y%m"), len(districts)),
        "PROVINCE_CODE": "11",  # 서울
        "DONG_CODE": np.tile([f"11{i:03d}" for i in range(1, len(districts) + 1)], 36),  # 36개월
        "DONG_NAME": np.tile(districts, 36),
        "RATE_INCOME_UNDER_20M": np.random.uniform(0.2, 0.4, 36 * len(districts)),
        "RATE_INCOME_20M_TO_30M": np.random.uniform(0.15, 0.3, 36 * len(districts)),
        "RATE_INCOME_30M_TO_40M": np.random.uniform(0.1, 0.2, 36 * len(districts)),
        "RATE_INCOME_40M_TO_50M": np.random.uniform(0.05, 0.15, 36 * len(districts)),
        "RATE_INCOME_50M_TO_60M": np.random.uniform(0.03, 0.1, 36 * len(districts)),
        "RATE_INCOME_60M_TO_70M": np.random.uniform(0.02, 0.07, 36 * len(districts)),
        "RATE_INCOME_OVER_70M": np.random.uniform(0.01, 0.05, 36 * len(districts)),
        "TOTAL_USAGE_AMOUNT": np.random.randint(100000000, 500000000, 36 * len(districts)),
        "TOTAL_CREDIT_CARD_USAGE_AMOUNT": np.random.randint(80000000, 400000000, 36 * len(districts)),
    })
    
    # 5. 아파트 시세 데이터 샘플
    apt_price_sample = pd.DataFrame({
        "BJD_CODE": np.tile([f"11{i:03d}" for i in range(1, len(dongs) + 1)], 60),  # 60개월
        "EMD": np.tile(dongs, 60),
        "JEONSE_PRICE_PER_SUPPLY_PYEONG": np.random.uniform(1000, 3000, 60 * len(dongs)) * 10000,  # 전세가
        "MEME_PRICE_PER_SUPPLY_PYEONG": np.random.uniform(2500, 7000, 60 * len(dongs)) * 10000,  # 매매가
        "REGION_LEVEL": "법정동",
        "SD": "서울특별시",
        "SGG": np.repeat(districts, len(dongs) // len(districts) + 1)[:len(dongs)],
        "TOTAL_HOUSEHOLDS": np.random.randint(5000, 20000, 60 * len(dongs)),
        "YYYYMMDD": np.repeat(pd.date_range(start='2020-01-01', end='2024-12-31', freq='M').strftime("%Y%m%d"), len(dongs))
    })
    
    # 6. 이탈 위험 고객 샘플 데이터
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
        'income_asset_df': income_asset_sample,
        'apt_price_df': apt_price_sample,
        'risk_customers': risk_customers
    }