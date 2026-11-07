import os
import joblib
import pandas as pd
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from config import PROCESSED_DIR
from src.transformer import get_linear_pipeline

def train_linear_model():
    path = os.path.join(PROCESSED_DIR, "train_features_linear.csv")
    df = pd.read_csv(path)

    y = df['RiskScore']
    X = df.drop(columns=['RiskScore'])

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"Обучащая выборка: {X_train.shape} строк")
    print(f"Валидационная выборка: {X_val.shape} строк")

    preprocessor = get_linear_pipeline()

    X_train_proc = preprocessor.fit_transform(X_train)
    X_val_proc = preprocessor.transform(X_val)

    alphas = np.logspace(-3, 3, 100)
    model = RidgeCV(alphas=alphas, cv=5)
    """model = SGDRegressor(
        penalty='l2',
        alpha=0.01,
        max_iter=2000,
        tol=1e-3,
        random_state=42
    )"""
    model.fit(X_train_proc, y_train)

    y_pred = model.predict(X_val_proc)
    print(f"R2-Score: {r2_score(y_val, y_pred)}")
    print(f"MSE-Score: {mean_squared_error(y_val, y_pred)}")
    print(f"RMSE-Score: {np.sqrt(mean_squared_error(y_val, y_pred))}")
    print(f"MAE-Score: {mean_absolute_error(y_val, y_pred)}")

    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)

    joblib.dump(preprocessor, os.path.join(models_dir, "ridge_preproc.joblib"))
    joblib.dump(model, os.path.join(models_dir, "ridge_model.joblib"))

if __name__ == "__main__":
    train_linear_model()