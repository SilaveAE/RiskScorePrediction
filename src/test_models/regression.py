import os
import joblib
import pandas as pd
from config import BASE_DIR, PROCESSED_DIR


REGRESSION_MODELS = {
    "ridge": ("ridge_preproc.joblib", "ridge_model.joblib"),
    "forest": ("preprocessor_forest.joblib", "forest_regressor.joblib"),
}


def predict_test(model_name: str) -> str:
    if model_name not in REGRESSION_MODELS:
        raise ValueError(
            f"Неизвестная модель {model_name!r}. Выберите одну из: "
            f"{', '.join(REGRESSION_MODELS)}."
        )

    test_path = os.path.join(PROCESSED_DIR, "test_features_linear.csv")
    preprocessor_name, model_file_name = REGRESSION_MODELS[model_name]
    preprocessor_path = os.path.join(BASE_DIR, "models", preprocessor_name)
    model_path = os.path.join(BASE_DIR, "models", model_file_name)

    for path in (test_path, preprocessor_path, model_path):
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Не найден файл: {path}. Сначала подготовьте данные и обучите "
                f"модель {model_name}."
            )

    test = pd.read_csv(test_path)

    identifiers = test["ID"].copy() if "ID" in test.columns else None
    features = test.drop(columns=["ID"], errors="ignore")

    preprocessor = joblib.load(preprocessor_path)
    model = joblib.load(model_path)
    features_processed = preprocessor.transform(features)
    risk_score = model.predict(features_processed)

    predictions = pd.DataFrame()
    if identifiers is not None:
        predictions["ID"] = identifiers
    predictions["RiskScore_pred"] = risk_score

    output_dir = os.path.join(BASE_DIR, "data", "predictions", "regression")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(
        output_dir, f"test_predictions_{model_name}.csv"
    )
    predictions.to_csv(output_path, index=False)
    print(f"Предсказания сохранены: {output_path} ({len(predictions)} строк)")
    return output_path


if __name__ == "__main__":
    predict_test("ridge")
