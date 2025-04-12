# Snowflake 접속 정보 
# 개인 키 파일 경로 설정 (실제 경로로 수정하세요)
PRIVATE_KEY_PATH = "C:/Users/정주환/rsa_key.p8"  # 개인 키 파일 경로

# 1. LOPLAT 데이터 (백화점 방문, 주거지/근무지, 날씨)
LOPLAT_CONFIG = {
    "account": "KGAWUSR-SN57219",
    "user": "wnghks5432",
    "private_key_file": PRIVATE_KEY_PATH,  # 비밀번호 대신 개인 키 파일 사용
    "warehouse": "COMPUTE_WH",
    "database": "DEPARTMENT_STORE_FOOT_TRAFFIC_FOR_SNOWFLAKE_STREAMLIT_HACKATHON",
    "schema": "PUBLIC"
}

# 2. RESIDENTIAL WORKPLACE 데이터
RESIDENCE_CONFIG = {
    "account": "KGAWUSR-SN57219",
    "user": "wnghks5432",
    "private_key_file": PRIVATE_KEY_PATH,  # 비밀번호 대신 개인 키 파일 사용
    "warehouse": "COMPUTE_WH",
    "database": "RESIDENTIAL__WORKPLACE_TRAFFIC_PATTERNS_FOR_SNOWFLAKE_STREAMLIT_HACKATHON",
    "schema": "PUBLIC"
}

# 3. WEATHER 데이터
WEATHER_CONFIG = {
    "account": "KGAWUSR-SN57219",
    "user": "wnghks5432",
    "private_key_file": PRIVATE_KEY_PATH,  # 비밀번호 대신 개인 키 파일 사용
    "warehouse": "COMPUTE_WH",
    "database": "SEOUL_TEMPERATURE__RAINFALL_FOR_SNOWFLAKE_STREAMLIT_HACKATHON",
    "schema": "PUBLIC"
}

# 4. SPH 데이터 (유동인구, 자산소득, 카드소비내역)
SPH_CONFIG = {
    "account": "KGAWUSR-SN57219",
    "user": "wnghks5432",
    "private_key_file": PRIVATE_KEY_PATH,  # 비밀번호 대신 개인 키 파일 사용
    "warehouse": "COMPUTE_WH",
    "database": "SEOUL_DISTRICTLEVEL_DATA_FLOATING_POPULATION_CONSUMPTION_AND_ASSETS",
    "schema": "GRANDATA"
}

# 5. DATAKNOWS 데이터 (아파트 시세, 인구)
DATAKNOWS_CONFIG = {
    "account": "KGAWUSR-SN57219",
    "user": "wnghks5432",
    "private_key_file": PRIVATE_KEY_PATH,  # 비밀번호 대신 개인 키 파일 사용
    "warehouse": "COMPUTE_WH",
    "database": "KOREAN_POPULATION__APARTMENT_MARKET_PRICE_DATA",
    "schema": "HACKATHON_2025Q2"
}

# 통합 Snowflake 접속 정보 (기본 설정)
# 앱에서는 이 설정을 사용하고, 필요시 data_loader.py에서 각 데이터베이스별 설정으로 전환
SNOWFLAKE_CONFIG = {
    "account": "KGAWUSR-SN57219",
    "user": "wnghks5432",
    "private_key_file": PRIVATE_KEY_PATH,  # 비밀번호 대신 개인 키 파일 사용
    "warehouse": "COMPUTE_WH",
    "database": "DEPARTMENT_STORE_FOOT_TRAFFIC_FOR_SNOWFLAKE_STREAMLIT_HACKATHON",  # 기본값
    "schema": "PUBLIC"
}

# 애플리케이션 설정
APP_CONFIG = {
    "title": "RetainPulse - 백화점 고객 이탈 예측 및 방지 시스템",
    "stores": ["신세계 강남", "더현대 서울", "롯데 본점"],
    "segments": ["VIP 고객", "정기 방문 고객", "간헐적 방문 고객", "이탈 위험 고객"],
    "default_risk_threshold": 70,
    "churn_days_threshold": 90,  # 이탈 기준 일수 (90일 이상 미방문)
    "date_range": {
        "start": "2021-01-01",
        "end": "2024-12-31"
    }
}

# 데이터베이스별 테이블 매핑
TABLE_MAPPING = {
    # LOPLAT 데이터
    "department_store": "SNOWFLAKE_STREAMLIT_HACKATHON_LOPLAT_DEPARTMENT_STORE_DATA",
    "residence_workplace": "SNOWFLAKE_STREAMLIT_HACKATHON_LOPLAT_HOME_OFFICE_RATIO",
    "weather": "SNOWFLAKE_STREAMLIT_HACKATHON_LOPLAT_SEOUL_TEMPERATURE_RAINFALL",
    
    # SPH 데이터
    "floating_population": "FLOATING_POPULATION_INFO",
    "income_asset": "ASSET_INCOME_INFO",
    "card_spending": "CARD_SALES_INFO",
    "admin_boundary": "CODE_MASTER", 
    "region_master": "M_SCCO_MST",
    
    # DATAKNOWS 데이터
    "apartment_price": "REGION_APT_RICHGO_MARKET_PRICE_M_H",
    "population": "REGION_MOIS_POPULATION_GENDER_AGE_M_H", 
    "female_child": "REGION_MOIS_POPULATION_AGE_UNDER5_PER_FEMALE_20TO40_M_H"
}