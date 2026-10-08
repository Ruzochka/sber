import yaml
import pandas as pd
import numpy as np
from src.data_loader import SberIndexDataLoader
from src.feature_engineering import FeatureEngineer
from src.change_point import ChangePointDetector
from src.models import ModelFactory
from src.evaluation import MetricsCalculator

def main():
    # 1. Загрузка конфигурации
    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    loader = SberIndexDataLoader(config)
    df = loader.load_data()

    cpd = ChangePointDetector(config)
    df = cpd.process_dataframe(
        df, 
        entity_col=config['data']['entity_col'], 
        date_col=config['data']['date_col'], 
        target_col=config['data']['target_col']
    )
    print(f"Найдено точек структурных сдвигов: {df['is_shock_point'].sum()}")

    print("=== 3. Генерация признаков (Feature Engineering) ===")
    fe = FeatureEngineer(config)
    df_features = fe.create_features(df)

    # Очистка от пропусков из-за лагов
    df_clean = df_features.dropna().reset_index(drop=True)

    # 4. Валидация по горизонтам прогнозирования
    horizons = config['forecasting']['horizons']
    results = []

    feature_cols = [c for c in df_clean.columns if c not in [
        config['data']['entity_col'], 
        config['data']['date_col'], 
        config['data']['target_col']
    ]]

    print("\n=== 4. Оценка моделей по горизонтам прогнозирования ===")
    max_date = df_clean[config['data']['date_col']].max()

    for h in horizons:
        cutoff_date = max_date - pd.DateOffset(months=h)
        
        train = df_clean[df_clean[config['data']['date_col']] <= cutoff_date]
        test = df_clean[df_clean[config['data']['date_col']] > cutoff_date]

        if len(test) == 0:
            continue

        X_train, y_train = train[feature_cols], train[config['data']['target_col']]
        X_test, y_test = test[feature_cols], test[config['data']['target_col']]

        # Prophet Baseline
        p_preds = ModelFactory.train_predict_prophet(train, test, config)
        p_metrics = MetricsCalculator.evaluate_all(y_test.values, p_preds)
        results.append({"Model": "Prophet", "Horizon_Months": h, **p_metrics})

        # LightGBM
        lgb_preds = ModelFactory.train_predict_lgb(X_train, y_train, X_test, config)
        lgb_metrics = MetricsCalculator.evaluate_all(y_test.values, lgb_preds)
        results.append({"Model": "LightGBM", "Horizon_Months": h, **lgb_metrics})

        # CatBoost
        cat_preds = ModelFactory.train_predict_catboost(X_train, y_train, X_test, config)
        cat_metrics = MetricsCalculator.evaluate_all(y_test.values, cat_preds)
        results.append({"Model": "CatBoost", "Horizon_Months": h, **cat_metrics})

    # Вывод результатов
    results_df = pd.DataFrame(results)
    print("\nИтоговые результаты моделирования (MAE и R2):")
    print(results_df.to_string(index=False))

    # Сохранение отчета по метрикам
    results_df.to_csv("benchmark_results.csv", index=False)
    print("\nРезультаты сохранены в benchmark_results.csv")

if __name__ == "__main__":
    main()