import os
import joblib

from catboost import CatBoostRegressor

import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

from src.preprocessing import get_forest_pipeline
from config import PROCESSED_DIR, MODELS_DIR

def train_cat_reg_model():
    path = os.path.join(PROCESSED_DIR, "train_features_regression.csv")
    df = pd.read_csv(path)

    y = df['RiskScore']
    X = df.drop(columns=['RiskScore', 'ID'], errors='ignore')

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = CatBoostRegressor(
        iterations=1500,
        learning_rate=0.05,
        depth=6,
        l2_leaf_reg=3.0,
        loss_function='RMSE',
        eval_metric='RMSE',
        random_seed=42,
        verbose=200
    )
    preprocessor = get_forest_pipeline()

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])

    print("Обучается CatBoost Regression...")
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_val)

    print(f"R2-Score: {r2_score(y_val, y_pred)}")
    print(f"MSE-Score: {mean_squared_error(y_val, y_pred)}")
    print(f"RMSE-Score: {np.sqrt(mean_squared_error(y_val, y_pred))}")
    print(f"MAE-Score: {mean_absolute_error(y_val, y_pred)}")

    models_dir = os.path.join(MODELS_DIR, 'regression')
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(pipeline, os.path.join(models_dir, "catboost_reg.joblib"))

if __name__ == "__main__":
    train_cat_reg_model()