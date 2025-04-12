import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve
from sklearn.cluster import KMeans
import joblib
import os
from datetime import datetime, timedelta

# 기존 함수들 유지 (원본 파일의 함수들)
# ...

# 이탈 예측 모델 (실제 고객 수준 데이터가 있을 경우 사용)
def train_churn_model(feature_data, target_data=None, model_type='rf'):
    """
    실제 고객 수준 데이터로 이탈 예측 모델 학습
    
    feature_data: 피처 데이터프레임
    target_data: 타겟 벡터 (이탈 여부)
    model_type: 모델 타입 ('rf': RandomForest, 'gb': GradientBoosting, 'lr': LogisticRegression)
    """
    # 타겟 데이터가 없는 경우 가상의 타겟 생성
    if target_data is None:
        print("Warning: 타겟 데이터가 없어 무작위 타겟을 생성합니다.")
        target_data = np.random.choice([0, 1], size=len(feature_data), p=[0.7, 0.3])
    
    # 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(feature_data, target_data, test_size=0.3, random_state=42)
    
    # 특성 스케일링
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 모델 선택
    if model_type == 'rf':
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        param_grid = {
            'n_estimators': [50, 100],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5]
        }
    elif model_type == 'gb':
        model = GradientBoostingClassifier(random_state=42)
        param_grid = {
            'n_estimators': [50, 100],
            'learning_rate': [0.01, 0.1],
            'max_depth': [3, 5]
        }
    elif model_type == 'lr':
        model = LogisticRegression(random_state=42, max_iter=1000)
        param_grid = {
            'C': [0.1, 1.0, 10.0],
            'penalty': ['l2']
        }
    else:
        raise ValueError("지원되지 않는 모델 타입입니다. 'rf', 'gb', 또는 'lr'을 사용하세요.")
    
    # 그리드 서치로 최적 파라미터 찾기
    grid_search = GridSearchCV(model, param_grid, cv=3, scoring='roc_auc')
    grid_search.fit(X_train_scaled, y_train)
    
    # 최적 모델
    best_model = grid_search.best_estimator_
    
    # 예측 및 평가
    y_pred = best_model.predict(X_test_scaled)
    y_prob = best_model.predict_proba(X_test_scaled)[:, 1]
    
    # 평가 메트릭
    metrics = {
        'accuracy': (y_pred == y_test).mean(),
        'precision_recall': classification_report(y_test, y_pred),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'auc': roc_auc_score(y_test, y_prob)
    }
    
    # ROC 곡선 데이터
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_data = pd.DataFrame({'fpr': fpr, 'tpr': tpr})
    
    # 정밀도-재현율 곡선 데이터
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    pr_data = pd.DataFrame({'precision': precision, 'recall': recall})
    
    # 특성 중요도
    if hasattr(best_model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'feature': feature_data.columns,
            'importance': best_model.feature_importances_
        }).sort_values('importance', ascending=False)
    else:
        feature_importance = pd.DataFrame({
            'feature': feature_data.columns,
            'importance': best_model.coef_[0]
        }).sort_values('importance', ascending=False)
    
    return {
        'model': best_model,
        'scaler': scaler,
        'metrics': metrics,
        'feature_importance': feature_importance,
        'roc_data': roc_data,
        'pr_data': pr_data,
        'X_test': X_test,
        'X_test_scaled': X_test_scaled
    }

# 로그 기반 이탈 예측 (통계적 접근)
def statistical_churn_prediction(visit_patterns, residence_workplace, weather_impact):
    """
    방문 패턴, 주거지 분포, 날씨 영향 데이터를 기반으로 한 통계적 이탈 예측
    
    visit_patterns: 방문 패턴 데이터
    residence_workplace: 주거지/근무지 데이터
    weather_impact: 날씨 영향 데이터
    """
    # 백화점별 분석
    stores = visit_patterns['store_trends']['DEP_NAME'].unique()
    
    # 결과 저장
    store_predictions = {}
    
    for store in stores:
        # 1. 방문 패턴 분석
        # 최근 3개월 방문 트렌드가 감소하는지 확인
        store_visits = visit_patterns['store_trends'][visit_patterns['store_trends']['DEP_NAME'] == store]
        
        if len(store_visits) > 3:
            recent_trend = store_visits.iloc[-3:]['avg_daily_visits'].values
            trend_decreasing = recent_trend[0] > recent_trend[-1]
        else:
            trend_decreasing = False
        
        # 2. 계절성 확인
        seasonal = visit_patterns['seasonal_patterns'][visit_patterns['seasonal_patterns']['DEP_NAME'] == store]
        if not seasonal.empty:
            current_season = 'Winter'  # 예시, 실제로는 현재 날짜 기반으로 계산
            season_visits = seasonal[seasonal['season'] == current_season]['avg_visits'].values
            season_low = season_visits < seasonal['avg_visits'].mean() if len(season_visits) > 0 else False
        else:
            season_low = False
        
        # 3. 날씨 민감도
        weather_corr = weather_impact['temp_correlation'][weather_impact['temp_correlation']['DEP_NAME'] == store]
        if not weather_corr.empty:
            temp_sensitive = abs(weather_corr['temp_correlation'].values[0]) > 0.3
            rain_sensitive = abs(weather_corr['rain_correlation'].values[0]) > 0.3
        else:
            temp_sensitive = False
            rain_sensitive = False
        
        # 이탈 위험 요소 계산
        risk_factors = {
            'trend_decreasing': trend_decreasing,
            'season_low': season_low,
            'temp_sensitive': temp_sensitive,
            'rain_sensitive': rain_sensitive
        }
        
        # 위험 점수 계산 (0-100)
        risk_score = (
            (30 if trend_decreasing else 0) +
            (25 if season_low else 0) +
            (15 if temp_sensitive else 0) +
            (10 if rain_sensitive else 0)
        )
        
        # 지역별 위험도 추정
        area_risk = {}
        home_dist = residence_workplace['home_distribution'][residence_workplace['home_distribution']['DEP_NAME'] == store]
        
        for _, row in home_dist.iterrows():
            district = row['ADDR_LV2']
            dong = row['ADDR_LV3']
            area = f"{district} {dong}"
            
            # 지역별 위험 점수 (기본 점수 + 지역 특성 가중치)
            # 여기서는 임의의 가중치 적용, 실제로는 지역 특성에 따라 다르게 설정
            area_weight = np.random.uniform(0.8, 1.2)  # 임의의 지역 가중치
            area_risk[area] = min(100, risk_score * area_weight)
        
        store_predictions[store] = {
            'overall_risk_score': risk_score,
            'risk_factors': risk_factors,
            'area_risk': area_risk
        }
    
    return store_predictions

# 세그먼트 기반 이탈 위험 예측
def segment_based_churn_prediction(segments, visit_patterns, weather_impact, income_assets=None, property_prices=None):
    """
    세그먼트 정보를 기반으로 한 이탈 위험 예측
    
    segments: 고객 세그먼트 데이터
    visit_patterns: 방문 패턴 데이터
    weather_impact: 날씨 영향 데이터
    income_assets: 소득/자산 데이터 (선택)
    property_prices: 부동산 시세 데이터 (선택)
    """
    # 세그먼트별 이탈 위험 기준선 (예시)
    base_risk_rates = {
        "고소득층": 0.15,
        "중상위층": 0.22,
        "중산층": 0.28,
        "일반소비자": 0.35
    }
    
    # 백화점별 세그먼트 이탈 위험
    store_segment_risk = {}
    
    for store in visit_patterns['store_trends']['DEP_NAME'].unique():
        segment_risk = {}
        
        # 계절성 분석
        seasonal = visit_patterns['seasonal_patterns'][visit_patterns['seasonal_patterns']['DEP_NAME'] == store]
        
        # 요일별 패턴
        dow = visit_patterns['dow_patterns'][visit_patterns['dow_patterns']['DEP_NAME'] == store]
        
        # 날씨 영향
        weather = weather_impact['temp_correlation'][weather_impact['temp_correlation']['DEP_NAME'] == store]
        
        # 각 세그먼트별 이탈 위험 계산
        for segment_name, base_risk in base_risk_rates.items():
            # 세그먼트별 가중치
            if segment_name == "고소득층":
                seasonal_weight = 1.0  # 계절성 영향 낮음
                weather_weight = 1.1   # 날씨 영향 약간 있음
                economic_weight = 0.9  # 경제적 영향 낮음
            elif segment_name == "중상위층":
                seasonal_weight = 1.1
                weather_weight = 1.2
                economic_weight = 1.0
            elif segment_name == "중산층":
                seasonal_weight = 1.2
                weather_weight = 1.3
                economic_weight = 1.1
            else:  # 일반소비자
                seasonal_weight = 1.3  # 계절성 영향 높음
                weather_weight = 1.4   # 날씨 영향 높음
                economic_weight = 1.2  # 경제적 영향 높음
            
            # 날씨 민감도 계산
            if not weather.empty:
                weather_sensitivity = abs(weather['temp_correlation'].values[0])
                weather_effect = 1.0 + (weather_sensitivity - 0.3) * weather_weight if weather_sensitivity > 0.3 else 1.0
            else:
                weather_effect = 1.0
            
            # 최종 이탈 위험 계산
            risk_rate = base_risk * seasonal_weight * weather_effect * economic_weight
            risk_rate = min(0.95, risk_rate)  # 최대 95%로 제한
            
            segment_risk[segment_name] = risk_rate
        
        store_segment_risk[store] = segment_risk
    
    return store_segment_risk

# 맞춤형 대응 전략 추천
def recommend_retention_strategies(churn_predictions, segments):
    """
    이탈 예측 결과를 기반으로 맞춤형 대응 전략 추천
    
    churn_predictions: 이탈 예측 결과
    segments: 고객 세그먼트 정보
    """
    # 세그먼트별 기본 전략
    base_strategies = {
        "고소득층": [
            "VIP 전용 프라이빗 이벤트 초청",
            "개인 맞춤형 쇼핑 컨시어지 서비스",
            "한정판 상품 우선 구매권 제공",
            "프리미엄 라운지 이용 특권"
        ],
        "중상위층": [
            "회원 등급 상향 기회 제공",
            "포인트 추가 적립 혜택",
            "제휴 브랜드 특별 할인",
            "시즌 사전 쇼핑 초대"
        ],
        "중산층": [
            "실용적 혜택 중심의 프로모션",
            "가족 동반 이벤트 및 할인",
            "적립금 2배 데이 운영",
            "생활 밀착형 서비스 강화"
        ],
        "일반소비자": [
            "첫 구매 고객 특별 할인",
            "온라인-오프라인 연계 서비스",
            "주요 상품군 할인 이벤트",
            "쿠폰북 제공"
        ]
    }
    
    # 이탈 위험 수준별 전략 강도
    risk_levels = {
        "저위험": (0.0, 0.3),
        "중위험": (0.3, 0.6),
        "고위험": (0.6, 1.0)
    }
    
    # 결과 저장
    recommended_strategies = {}
    
    for store, segment_risks in churn_predictions.items():
        store_strategies = {}
        
        for segment, risk in segment_risks.items():
            # 위험 수준 결정
            risk_level = next((level for level, (min_risk, max_risk) in risk_levels.items() 
                              if min_risk <= risk < max_risk), "고위험")
            
            # 기본 전략 선택
            base_strategy = base_strategies.get(segment, [])
            
            # 위험 수준별 추가 전략
            if risk_level == "저위험":
                additional = ["리마인더 커뮤니케이션", "일반 만족도 조사"]
            elif risk_level == "중위험":
                additional = ["개인화된 혜택 제안", "관심 카테고리 맞춤 프로모션", "충성도 프로그램 강조"]
            else:  # 고위험
                additional = ["즉각적인 할인 혜택 제공", "1:1 고객 상담", "긴급 리텐션 프로그램 적용", "VIP 혜택 일시적 제공"]
            
            # 최종 추천 전략
            store_strategies[segment] = {
                "risk_level": risk_level,
                "risk_rate": risk,
                "base_strategies": base_strategy,
                "additional_strategies": additional,
                "priority": "높음" if risk_level == "고위험" else "중간" if risk_level == "중위험" else "낮음"
            }
        
        recommended_strategies[store] = store_strategies
    
    return recommended_strategies

# 모델 저장 및 로드
def save_model(model_dict, model_dir='models'):
    """
    모델과 관련 객체 저장
    """
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    # 모델 저장
    if 'model' in model_dict:
        joblib.dump(model_dict['model'], os.path.join(model_dir, 'churn_model.pkl'))
    
    if 'scaler' in model_dict:
        joblib.dump(model_dict['scaler'], os.path.join(model_dir, 'scaler.pkl'))
    
    # 메타데이터 저장
    metadata = {k: v for k, v in model_dict.items() 
               if k not in ['model', 'scaler', 'X_test', 'X_test_scaled', 'feature_importance']}
    
    joblib.dump(metadata, os.path.join(model_dir, 'model_metadata.pkl'))
    
    # 특성 중요도 저장
    if 'feature_importance' in model_dict:
        model_dict['feature_importance'].to_csv(os.path.join(model_dir, 'feature_importance.csv'), index=False)
    
    return os.path.join(model_dir, 'churn_model.pkl')

def load_model(model_dir='models'):
    """
    저장된 모델과 관련 객체 로드
    """
    model_path = os.path.join(model_dir, 'churn_model.pkl')
    scaler_path = os.path.join(model_dir, 'scaler.pkl')
    metadata_path = os.path.join(model_dir, 'model_metadata.pkl')
    feature_importance_path = os.path.join(model_dir, 'feature_importance.csv')
    
    model_dict = {}
    
    # 모델 로드
    if os.path.exists(model_path):
        model_dict['model'] = joblib.load(model_path)
    
    # 스케일러 로드
    if os.path.exists(scaler_path):
        model_dict['scaler'] = joblib.load(scaler_path)
    
    # 메타데이터 로드
    if os.path.exists(metadata_path):
        metadata = joblib.load(metadata_path)
        model_dict.update(metadata)
    
    # 특성 중요도 로드
    if os.path.exists(feature_importance_path):
        model_dict['feature_importance'] = pd.read_csv(feature_importance_path)
    
    return model_dict

# ------------- 새로 추가된 데이터 기반 이탈 예측 함수 ------------- #

def data_driven_churn_prediction(visits_df, residence_workplace=None, weather_impact=None, purchase_data=None):
    """
    실제 방문 데이터를 기반으로 한 이탈 위험 예측
    
    visits_df: 방문 데이터프레임 (DATE_KST, DEP_NAME, COUNT 등 포함)
    residence_workplace: 주거지/근무지 데이터 (선택)
    weather_impact: 날씨 영향 데이터 (선택)
    purchase_data: 구매 데이터 (선택, 있을 경우 사용)
    """
    import pandas as pd
    import numpy as np
    from datetime import datetime, timedelta
    from sklearn.ensemble import IsolationForest
    from sklearn.linear_model import LinearRegression
    
    # 날짜 형식 변환
    visits_df = visits_df.copy()
    if not pd.api.types.is_datetime64_any_dtype(visits_df['DATE_KST']):
        visits_df['DATE_KST'] = pd.to_datetime(visits_df['DATE_KST'])
    
    # 백화점별 분석 결과 저장
    store_churn_risk = {}
    
    # 현재 날짜 (분석 기준일)
    current_date = visits_df['DATE_KST'].max()
    
    for store in visits_df['DEP_NAME'].unique():
        store_visits = visits_df[visits_df['DEP_NAME'] == store]
        
        # 1. 시계열 기반 이탈 위험 계산
        # 주간 방문 트렌드 계산
        store_visits['week'] = store_visits['DATE_KST'].dt.isocalendar().week
        store_visits['year'] = store_visits['DATE_KST'].dt.isocalendar().year
        
        weekly_visits = store_visits.groupby(['year', 'week']).agg(
            total_visits=('COUNT', 'sum'),
            avg_daily_visits=('COUNT', 'mean')
        ).reset_index()
        
        # 시계열 데이터에 날짜 인덱스 생성
        weekly_visits['date'] = weekly_visits.apply(
            lambda x: datetime.strptime(f"{int(x['year'])}-W{int(x['week'])}-1", "%Y-W%W-%w"), axis=1
        )
        weekly_visits = weekly_visits.set_index('date')
        
        # 트렌드 계산 (3개월 이동평균)
        trend_risk = 0.0
        if len(weekly_visits) >= 12:  # 최소 12주 데이터 필요
            weekly_visits['visits_ma'] = weekly_visits['total_visits'].rolling(window=12).mean()
            
            # 최근 추세 (마지막 12주)
            recent_trend = weekly_visits.iloc[-12:].copy()
            
            if len(recent_trend) > 0:
                # 선형 회귀로 트렌드 기울기 계산
                X = np.array(range(len(recent_trend))).reshape(-1, 1)
                y = recent_trend['total_visits'].values
                
                model = LinearRegression()
                model.fit(X, y)
                slope = model.coef_[0]
                
                # 트렌드 하락 정도에 따른 위험도
                if slope < 0:
                    # 하락폭에 따른 위험도 (0~1 사이로 정규화)
                    # 주간 방문객의 10% 이상 하락시 최대 위험
                    avg_visits = recent_trend['total_visits'].mean()
                    decline_pct = abs(slope * 12) / avg_visits if avg_visits > 0 else 0
                    trend_risk = min(1.0, decline_pct / 0.1)  # 10% 하락시 위험도 1.0
        
        # 2. 방문 주기 기반 위험 계산
        # 최근 3개월과 이전 3개월 비교
        recent_period = store_visits[store_visits['DATE_KST'] > (current_date - timedelta(days=90))]
        previous_period = store_visits[(store_visits['DATE_KST'] <= (current_date - timedelta(days=90))) & 
                                      (store_visits['DATE_KST'] > (current_date - timedelta(days=180)))]
        
        # 방문 빈도 변화 계산
        recent_avg = recent_period['COUNT'].mean() if len(recent_period) > 0 else 0
        previous_avg = previous_period['COUNT'].mean() if len(previous_period) > 0 else 0
        
        # 방문 빈도 감소율
        frequency_decline = 0.0
        if previous_avg > 0:
            frequency_decline = max(0, (previous_avg - recent_avg) / previous_avg)
        
        # 3. 이상치 탐지 기반 이탈 위험 계산
        # 최근 6개월 데이터로 이상치 탐지
        recent_6m = store_visits[store_visits['DATE_KST'] > (current_date - timedelta(days=180))]
        
        # 주요 특성: 요일별 방문 패턴
        day_patterns = recent_6m.groupby(recent_6m['DATE_KST'].dt.dayofweek)['COUNT'].mean().to_dict()
        
        # 일자별 방문 데이터에 요일별 평균과의 차이 계산
        recent_6m['day_of_week'] = recent_6m['DATE_KST'].dt.dayofweek
        recent_6m['expected_visits'] = recent_6m['day_of_week'].map(day_patterns)
        recent_6m['visit_ratio'] = recent_6m['COUNT'] / recent_6m['expected_visits']
        
        # 이상치 탐지 모델 적용 (Isolation Forest)
        anomaly_ratio = 0.0
        if len(recent_6m) > 50:  # 충분한 데이터가 있을 경우
            # 특성 설정: 요일별 예상 대비 실제 방문 비율, 날짜 순서
            X = recent_6m[['visit_ratio']].fillna(1)
            X['date_order'] = range(len(X))  # 최근 날짜에 더 가중치 부여
            
            # 이상치 탐지
            model = IsolationForest(contamination=0.1, random_state=42)
            recent_6m['anomaly'] = model.fit_predict(X)
            
            # 최근 1개월의 이상치 비율
            last_month = recent_6m[recent_6m['DATE_KST'] > (current_date - timedelta(days=30))]
            anomaly_ratio = (last_month['anomaly'] == -1).mean() if len(last_month) > 0 else 0
        
        # 4. 계절성 고려 (전년 동기 대비)
        seasonal_risk = 0.0
        
        if pd.Timestamp(current_date).year > pd.Timestamp(current_date).year - 1:
            # 전년 동기 데이터 (같은 달)
            current_month = pd.Timestamp(current_date).month
            current_year = pd.Timestamp(current_date).year
            
            current_month_data = store_visits[
                (store_visits['DATE_KST'].dt.month == current_month) & 
                (store_visits['DATE_KST'].dt.year == current_year)
            ]
            
            prev_year_month_data = store_visits[
                (store_visits['DATE_KST'].dt.month == current_month) & 
                (store_visits['DATE_KST'].dt.year == current_year - 1)
            ]
            
            # 전년 동기 대비 감소율
            current_avg = current_month_data['COUNT'].mean() if len(current_month_data) > 0 else 0
            prev_year_avg = prev_year_month_data['COUNT'].mean() if len(prev_year_month_data) > 0 else 0
            
            if prev_year_avg > 0:
                yoy_decline = max(0, (prev_year_avg - current_avg) / prev_year_avg)
                seasonal_risk = min(1.0, yoy_decline / 0.15)  # 15% 이상 감소시 최대 위험
        
        # 5. 날씨 영향 고려 (선택적)
        weather_risk = 0.0
        
        if weather_impact is not None:
            temp_corr = weather_impact['temp_correlation']
            store_weather = temp_corr[temp_corr['DEP_NAME'] == store]
            
            if not store_weather.empty:
                # 날씨 민감도에 따른 위험도
                temp_sensitivity = abs(store_weather['temp_correlation'].values[0])
                rain_sensitivity = abs(store_weather['rain_correlation'].values[0])
                
                # 민감도 기준값 (0.3 이상이면 민감하다고 판단)
                weather_risk = max(
                    min(1.0, temp_sensitivity / 0.5),  # 온도 상관관계 0.5 이상이면 최대 위험
                    min(1.0, rain_sensitivity / 0.4)   # 강수량 상관관계 0.4 이상이면 최대 위험
                )
        
        # 6. 주거지-상권 거리 기반 위험도 (선택적)
        distance_risk = 0.0
        
        if residence_workplace is not None:
            home_data = residence_workplace['home_distribution']
            store_home = home_data[home_data['DEP_NAME'] == store]
            
            if not store_home.empty:
                # 주거지 분포에 따른 위험도 (가정: 기존 데이터에는 distance_type 컬럼이 없으므로 추가 계산 필요)
                # 원거리 지역 비율 계산 (임의로 기준 설정)
                far_areas = ['강남구', '서초구']  # 예시: 해당 백화점에서 원거리 지역
                if store == '롯데 본점':
                    far_areas = ['강남구', '서초구', '송파구', '강동구']
                elif store == '신세계 강남':
                    far_areas = ['종로구', '중구', '용산구', '성북구']
                elif store == '더현대 서울':
                    far_areas = ['강남구', '서초구', '송파구', '강동구']
                
                far_areas_ratio = store_home[store_home['ADDR_LV2'].isin(far_areas)]['ratio'].sum()
                distance_risk = min(1.0, far_areas_ratio / 0.3)  # 원거리 고객 30% 이상이면 최대 위험
        
        # 7. 세그먼트별 위험도 계산 (APP_CONFIG 세그먼트 사용)
        # 원래 프로젝트 세그먼트: "VIP 고객", "정기 방문 고객", "간헐적 방문 고객", "이탈 위험 고객"
        segments = {
            'VIP 고객': 0.10,  # 기본 위험도
            '정기 방문 고객': 0.20,
            '간헐적 방문 고객': 0.35,
            '이탈 위험 고객': 0.50
        }
        
        # 각 위험 요소의 가중치 설정
        weights = {
            'trend_risk': 0.25,
            'frequency_decline': 0.20,
            'anomaly_ratio': 0.15,
            'seasonal_risk': 0.15,
            'weather_risk': 0.10,
            'distance_risk': 0.15
        }
        
        # 종합 위험도 계산
        total_risk = (
            trend_risk * weights['trend_risk'] +
            frequency_decline * weights['frequency_decline'] +
            anomaly_ratio * weights['anomaly_ratio'] +
            seasonal_risk * weights['seasonal_risk'] +
            weather_risk * weights['weather_risk'] +
            distance_risk * weights['distance_risk']
        )
        
        # 세그먼트별 위험도 조정
        segment_risks = {}
        for segment, base_risk in segments.items():
            # 세그먼트 특성에 따른 가중치 조정
            if segment == 'VIP 고객':
                segment_weight = 0.7  # VIP는 기본 위험이 낮음
            elif segment == '정기 방문 고객':
                segment_weight = 0.9
            elif segment == '간헐적 방문 고객':
                segment_weight = 1.1
            else:  # 이탈 위험 고객
                segment_weight = 1.3  # 이미 이탈 위험군은 가중치 높음
            
            # 최종 세그먼트별 위험도 계산 (기본 위험 + 데이터 기반 위험)
            adjusted_risk = base_risk + (total_risk * segment_weight * 0.5)
            segment_risks[segment] = min(0.95, adjusted_risk)  # 최대 95%로 제한
        
        store_churn_risk[store] = {
            'segment_risks': segment_risks,
            'overall_risk': total_risk,
            'risk_factors': {
                'trend_risk': trend_risk,
                'frequency_decline': frequency_decline,
                'anomaly_ratio': anomaly_ratio,
                'seasonal_risk': seasonal_risk,
                'weather_risk': weather_risk,
                'distance_risk': distance_risk
            }
        }
    
    # 이탈 위험 유형 분류
    risk_types = {}
    for store, risks in store_churn_risk.items():
        factors = risks['risk_factors']
        # 가장 높은 위험 요소 식별
        max_factor = max(factors.items(), key=lambda x: x[1])
        
        if max_factor[0] == 'trend_risk' and max_factor[1] > 0.5:
            risk_types[store] = '트렌드 하락형'
        elif max_factor[0] == 'frequency_decline' and max_factor[1] > 0.4:
            risk_types[store] = '방문 감소형'
        elif max_factor[0] == 'seasonal_risk' and max_factor[1] > 0.5:
            risk_types[store] = '계절성 이탈형'
        elif max_factor[0] == 'weather_risk' and max_factor[1] > 0.5:
            risk_types[store] = '날씨 민감형'
        elif max_factor[0] == 'distance_risk' and max_factor[1] > 0.5:
            risk_types[store] = '거리 제약형'
        else:
            risk_types[store] = '복합 요인형'
    
    # 고객별 이탈 위험 분류 시뮬레이션 (실제 고객 데이터 없을 경우)
    # 이 부분은 실제 고객 수준 데이터가 있을 때 구현
    at_risk_customers = {}
    risk_thresholds = {
        '낮음': 0.3,
        '중간': 0.5,
        '높음': 0.7
    }
    
    for store, risks in store_churn_risk.items():
        # 이탈 위험이 높은 세그먼트 식별
        high_risk_segments = [
            segment for segment, risk in risks['segment_risks'].items()
            if risk >= risk_thresholds['높음']
        ]
        
        # 이탈 징후 데이터에서 고객 식별 (가상)
        churn_signals = {
            '마지막 방문일 90일 이상': 0.7,
            '방문 빈도 50% 이상 감소': 0.6,
            '객단가 30% 이상 감소': 0.5,
            '특정 카테고리 방문 중단': 0.4,
            '시즌 행사 불참': 0.3
        }
        
        # 가상의 위험 고객 수 (세그먼트별 비율 적용)
        customer_counts = {
            'VIP 고객': 200,
            '정기 방문 고객': 500,
            '간헐적 방문 고객': 800,
            '이탈 위험 고객': 300
        }
        
        at_risk_customers[store] = {
            'high_risk_segments': high_risk_segments,
            'risk_thresholds': risk_thresholds,
            'estimated_at_risk': sum(
                int(count * risks['segment_risks'][segment])
                for segment, count in customer_counts.items()
            ),
            'churn_signals': churn_signals
        }
    
    return {
        'store_churn_risk': store_churn_risk,
        'risk_types': risk_types,
        'at_risk_customers': at_risk_customers
    }


def identify_at_risk_customers(customer_visits, purchase_data=None, churn_threshold_days=90):
    """
    개별 고객 방문 이력을 바탕으로 이탈 위험이 있는 고객 식별
    
    customer_visits: 고객별 방문 이력 DataFrame(CUSTOMER_ID, DATE_KST, DEP_NAME 포함)
    purchase_data: 구매 데이터 (선택, 있을 경우 사용)
    churn_threshold_days: 이탈로 간주할 최소 미방문 일수
    """
    import pandas as pd
    import numpy as np
    from datetime import datetime, timedelta
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    
    # 현재 날짜 (분석 기준일)
    current_date = customer_visits['DATE_KST'].max()
    
    # 1. RFM 분석
    # Recency: 최근 방문일로부터 경과일
    # Frequency: 총 방문 횟수
    # Monetary: 총 구매액 (구매 데이터가 있는 경우)
    
    # 고객별 최근 방문일 계산
    customer_recency = customer_visits.groupby('CUSTOMER_ID')['DATE_KST'].max().reset_index()
    customer_recency['days_since_last_visit'] = (current_date - customer_recency['DATE_KST']).dt.days
    
    # 고객별 방문 빈도 계산
    customer_frequency = customer_visits.groupby('CUSTOMER_ID').size().reset_index(name='visit_count')
    
    # RFM 데이터 병합
    rfm_data = pd.merge(customer_recency, customer_frequency, on='CUSTOMER_ID')
    
    # 구매 데이터가 있는 경우
    if purchase_data is not None:
        customer_monetary = purchase_data.groupby('CUSTOMER_ID')['AMOUNT'].sum().reset_index()
        rfm_data = pd.merge(rfm_data, customer_monetary, on='CUSTOMER_ID', how='left')
        rfm_data['AMOUNT'] = rfm_data['AMOUNT'].fillna(0)
    else:
        rfm_data['AMOUNT'] = 0
    
    # 2. 이탈 위험 스코어 계산
    # Recency에 가장 큰 가중치 (오래된 고객일수록 위험)
    rfm_data['recency_score'] = np.minimum(rfm_data['days_since_last_visit'] / churn_threshold_days, 1.0)
    
    # Frequency 역수 (방문 빈도가 낮을수록 위험)
    max_frequency = rfm_data['visit_count'].max()
    rfm_data['frequency_score'] = 1.0 - (rfm_data['visit_count'] / max_frequency)
    rfm_data['frequency_score'] = rfm_data['frequency_score'].clip(0.0, 1.0)
    
    # Monetary 역수 (구매액이 낮을수록 위험)
    if rfm_data['AMOUNT'].max() > 0:
        max_monetary = rfm_data['AMOUNT'].max()
        rfm_data['monetary_score'] = 1.0 - (rfm_data['AMOUNT'] / max_monetary)
        rfm_data['monetary_score'] = rfm_data['monetary_score'].clip(0.0, 1.0)
    else:
        rfm_data['monetary_score'] = 0.5  # 데이터 없는 경우 중간값
    
    # 종합 이탈 위험 스코어 (가중 평균)
    weights = {'recency': 0.6, 'frequency': 0.3, 'monetary': 0.1}
    rfm_data['churn_risk_score'] = (
        rfm_data['recency_score'] * weights['recency'] +
        rfm_data['frequency_score'] * weights['frequency'] +
        rfm_data['monetary_score'] * weights['monetary']
    )
    
    # 3. 방문 패턴 기반 위험 탐지
    # 고객별 방문 간격 패턴 분석
    visit_patterns = {}
    
    for customer_id in rfm_data['CUSTOMER_ID'].unique():
        customer_data = customer_visits[customer_visits['CUSTOMER_ID'] == customer_id].sort_values('DATE_KST')
        
        if len(customer_data) >= 3:  # 최소 3회 방문 이력 필요
            # 방문 간격 계산
            customer_data['next_visit'] = customer_data['DATE_KST'].shift(-1)
            customer_data['days_between_visits'] = (customer_data['next_visit'] - customer_data['DATE_KST']).dt.days
            
            # 평균 방문 간격 및 표준편차
            visit_intervals = customer_data['days_between_visits'].dropna()
            
            if len(visit_intervals) > 0:
                avg_interval = visit_intervals.mean()
                std_interval = visit_intervals.std() if len(visit_intervals) > 1 else avg_interval * 0.2
                
                # 예상 다음 방문일
                last_visit = customer_data['DATE_KST'].max()
                expected_next_visit = last_visit + timedelta(days=avg_interval)
                
                # 예상 방문일 초과 정도
                days_overdue = (current_date - expected_next_visit).days
                
                # 초과일수가 표준편차의 2배 이상이면 위험
                if std_interval > 0:
                    interval_risk = max(0, days_overdue / (std_interval * 2))
                else:
                    interval_risk = max(0, days_overdue / avg_interval) if avg_interval > 0 else 0
                
                visit_patterns[customer_id] = {
                    'avg_interval': avg_interval,
                    'std_interval': std_interval,
                    'expected_next_visit': expected_next_visit,
                    'days_overdue': days_overdue,
                    'interval_risk': min(1.0, interval_risk)
                }
    
    # 방문 패턴 데이터를 RFM 데이터에 병합
    rfm_data['interval_risk'] = rfm_data['CUSTOMER_ID'].map(
        lambda x: visit_patterns.get(x, {}).get('interval_risk', 0.0)
    )
    
    # 방문 패턴 위험 스코어 반영
    rfm_data['churn_risk_score'] = 0.7 * rfm_data['churn_risk_score'] + 0.3 * rfm_data['interval_risk']
    
    # 4. 클러스터링으로 세그먼트 구분 - 프로젝트 세그먼트에 맞춤
    if len(rfm_data) > 100:  # 충분한 데이터가 있을 경우만 클러스터링
        # 클러스터링을 위한 특성
        clustering_features = [
            'days_since_last_visit',
            'visit_count',
            'churn_risk_score'
        ]
        
        if 'AMOUNT' in rfm_data.columns and rfm_data['AMOUNT'].max() > 0:
            clustering_features.append('AMOUNT')
        
        # 특성 스케일링
        X = rfm_data[clustering_features].copy()
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # K-Means 클러스터링
        kmeans = KMeans(n_clusters=4, random_state=42)
        rfm_data['cluster'] = kmeans.fit_predict(X_scaled)
        
        # 클러스터별 특성 분석
        cluster_profiles = rfm_data.groupby('cluster').agg({
            'days_since_last_visit': 'mean',
            'visit_count': 'mean',
            'churn_risk_score': 'mean',
            'CUSTOMER_ID': 'count'
        }).rename(columns={'CUSTOMER_ID': 'customer_count'})
        
        # 프로젝트 세그먼트에 맞는 클러스터 이름 부여
        cluster_names = {}
        for cluster in cluster_profiles.index:
            profile = cluster_profiles.loc[cluster]
            
            if profile['churn_risk_score'] >= 0.7:
                cluster_names[cluster] = '이탈 위험 고객'
            elif profile['churn_risk_score'] >= 0.4:
                cluster_names[cluster] = '간헐적 방문 고객'
            elif profile['visit_count'] > 10:
                cluster_names[cluster] = 'VIP 고객'
            else:
                cluster_names[cluster] = '정기 방문 고객'
        
        # 클러스터 이름 매핑
        rfm_data['segment'] = rfm_data['cluster'].map(cluster_names)
    else:
        # 간소화된 세그먼트 구분 (충분한 데이터 없을 경우)
        conditions = [
            (rfm_data['churn_risk_score'] >= 0.7),
            (rfm_data['churn_risk_score'] >= 0.4),
            (rfm_data['visit_count'] > 10),
            (True)
        ]
        choices = ['이탈 위험 고객', '간헐적 방문 고객', 'VIP 고객', '정기 방문 고객']
        rfm_data['segment'] = np.select(conditions, choices)
    
    # 5. 이탈 위험 고객 그룹화
    high_risk = rfm_data[rfm_data['churn_risk_score'] >= 0.7].copy()
    medium_risk = rfm_data[(rfm_data['churn_risk_score'] >= 0.4) & (rfm_data['churn_risk_score'] < 0.7)].copy()
    
    # 6. 이탈 원인 추정
    high_risk['churn_reason'] = np.where(
        high_risk['days_since_last_visit'] > churn_threshold_days, '장기 미방문',
        np.where(high_risk['visit_count'] <= 2, '초기 이탈',
                np.where(high_risk['interval_risk'] > 0.7, '방문 주기 이탈', '복합 요인'))
    )
    
    # 중간 위험 고객 이탈 원인
    medium_risk['churn_reason'] = np.where(
        medium_risk['days_since_last_visit'] > (churn_threshold_days * 0.7), '방문 간격 증가',
        np.where(medium_risk['interval_risk'] > 0.5, '방문 주기 변화', '구매 패턴 변화')
    )
    
    # 7. 이탈 위험 고객 대응 전략 추천
    high_risk['recommended_action'] = np.where(
        high_risk['churn_reason'] == '장기 미방문', '즉시 연락 및 특별 할인 제공',
        np.where(high_risk['churn_reason'] == '초기 이탈', '웰컴백 프로모션',
               np.where(high_risk['churn_reason'] == '방문 주기 이탈', '개인 맞춤 프로모션', '종합 리텐션 프로그램'))
    )
    
    medium_risk['recommended_action'] = np.where(
        medium_risk['churn_reason'] == '방문 간격 증가', '리마인더 및 인센티브 제공',
        np.where(medium_risk['churn_reason'] == '방문 주기 변화', '생활 패턴 맞춤 혜택',
                '카테고리 맞춤 프로모션')
    )
    
    # 결과 정리
    results = {
        'all_customers': rfm_data,
        'high_risk_customers': high_risk,
        'medium_risk_customers': medium_risk,
        'risk_distribution': rfm_data.groupby('segment')['CUSTOMER_ID'].count().to_dict(),
        'risk_thresholds': {
            '고위험': 0.7,
            '중위험': 0.4,
            '저위험': 0.0
        }
    }
    
    if 'cluster' in rfm_data.columns:
        results['cluster_profiles'] = cluster_profiles
        results['cluster_names'] = cluster_names
    
    # 방문 간격 패턴 정보도 추가
    results['visit_patterns'] = visit_patterns
    
    return results

# 기존 feature_eng.py의 predict_churn_risk 함수를 개선한 버전
def advanced_churn_prediction(visit_patterns, residence_workplace, weather_impact):
    """
    데이터 기반 이탈 위험 고객 분석 함수 (기존 predict_churn_risk 대체 가능)
    
    visit_patterns: analyze_visit_patterns 함수의 결과
    residence_workplace: analyze_residence_workplace 함수의 결과
    weather_impact: analyze_weather_impact 함수의 결과
    """
    # 데이터 준비
    visits_df = visit_patterns['processed_visits']
    
    # 이탈 위험 예측 모델 실행
    churn_risk = data_driven_churn_prediction(
        visits_df=visits_df,
        residence_workplace=residence_workplace,
        weather_impact=weather_impact
    )
    
    # 기존 함수와 호환되는 형식으로 결과 변환
    store_risk = {}
    for store, risk_data in churn_risk['store_churn_risk'].items():
        store_risk[store] = risk_data['segment_risks']
    
    # 이탈 위험 유형 및 방지 전략
    risk_types = {
        '트렌드 하락형': '시간이 지남에 따라 방문 횟수가 지속적으로 감소하는 패턴',
        '방문 감소형': '최근 방문 빈도가 이전 기간 대비 급격히 감소한 현상',
        '계절성 이탈형': '특정 계절에 방문이 크게 감소하는 계절적 패턴',
        '날씨 민감형': '특정 날씨 조건에 민감하게 반응하여 방문이 감소하는 특성',
        '거리 제약형': '거주지에서 백화점까지의 거리가 방문을 제한하는 요인',
        '복합 요인형': '여러 요인이 복합적으로 작용하여 이탈 위험이 증가하는 상황'
    }
    
    prevention_strategies = {
        '트렌드 하락형': [
            '맞춤형 고객 재활성화 프로그램 도입',
            '장기 미방문 고객 대상 특별 혜택 제공',
            '방문 트렌드 분석 및 원인 파악을 위한 심층 조사'
        ],
        '방문 감소형': [
            '방문 빈도 보상 프로그램 강화',
            '주기적 방문 유도 인센티브 도입',
            '재방문 쿠폰 및 할인 제공'
        ],
        '계절성 이탈형': [
            '비수기 특별 프로모션 및 이벤트 개최',
            '계절별 타겟 마케팅 전략 수립',
            '시즌 전환기 고객 유지 프로그램 운영'
        ],
        '날씨 민감형': [
            '날씨 연동 실시간 프로모션 시스템 도입',
            '우천/한파 시 교통 편의 서비스 제공',
            '날씨 영향이 적은 실내 활동 강화'
        ],
        '거리 제약형': [
            '원거리 고객 교통비 지원 프로그램',
            '온라인-오프라인 옴니채널 전략 강화',
            '지역 기반 픽업 서비스 및 홈 딜리버리 확대'
        ],
        '복합 요인형': [
            '데이터 기반 개인화 전략 수립',
            '세그먼트별 맞춤형 CRM 프로그램 운영',
            '정기적 고객 데이터 분석 및 전략 최적화'
        ]
    }
    
    # 결과 반환
    return {
        'store_risk': store_risk,
        'risk_types': risk_types,
        'prevention_strategies': prevention_strategies,
        'detailed_risk_data': churn_risk  # 추가 세부 정보 제공
    }