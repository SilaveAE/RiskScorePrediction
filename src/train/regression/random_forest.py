import os
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

from src.preprocessing import get_forest_pipeline
from config import PROCESSED_DIR, MODELS_DIR

def train_forest_model():
    path = os.path.join(PROCESSED_DIR, "train_features_regression.csv")
    df = pd.read_csv(path)

    y = df['RiskScore']
    X = df.drop(columns=['RiskScore'])

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=10,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    )
    preprocessor = get_forest_pipeline()

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_val)

    print("Обучается RandomForest...")
    print(f"R2-Score: {r2_score(y_val, y_pred)}")
    print(f"MSE-Score: {mean_squared_error(y_val, y_pred)}")
    print(f"RMSE-Score: {np.sqrt(mean_squared_error(y_val, y_pred))}")
    print(f"MAE-Score: {mean_absolute_error(y_val, y_pred)}")

    models_dir = os.path.join(MODELS_DIR, 'regression')
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(pipeline, os.path.join(models_dir, "random_forest.joblib"))

if __name__ == "__main__":
    train_forest_model()