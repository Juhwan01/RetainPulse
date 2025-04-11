# Snowflake 접속 정보
SNOWFLAKE_CONFIG = {
    "account": "your_account",
    "user": "your_username",
    "password": "your_password",  # 실제 프로젝트에서는 환경변수 등으로 관리
    "warehouse": "your_warehouse",
    "database": "your_database",
    "schema": "your_schema"
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