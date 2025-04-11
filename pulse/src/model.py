import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve
import joblib
import os

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