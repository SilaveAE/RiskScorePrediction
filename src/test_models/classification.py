import os

import joblib
import pandas as pd

from config import BASE_DIR, PROCESSED_DIR


def predict_test() -> str:
    test_path = os.path.join(PROCESSED_DIR, "test_features_class.csv")
    preprocessor_path = os.path.join(BASE_DIR, "models", "preprocessor_class.joblib")
    model_path = os.path.join(BASE_DIR, "models", "logistic_model.joblib")

    for path in (test_path, preprocessor_path, model_path):
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Не найден файл: {path}. Сначала подготовьте данные и обучите модель."
            )

    test = pd.read_csv(test_path)
    if test.empty:
        raise ValueError(f"Тестовый набор пуст: {test_path}")

    identifiers = test["ID"].copy() if "ID" in test.columns else None
    features = test.drop(columns=["ID"], errors="ignore")

    preprocessor = joblib.load(preprocessor_path)
    model = joblib.load(model_path)
    features_processed = preprocessor.transform(features)

    class_labels = model.predict(features_processed)
    positive_class_index = list(model.classes_).index(1)
    risk_probability = model.predict_proba(features_processed)[:, positive_class_index]

    predictions = pd.DataFrame()
    if identifiers is not None:
        predictions["ID"] = identifiers
    predictions["isRisk_pred"] = class_labels
    predictions["risk_probability"] = risk_probability

    output_dir = os.path.join(BASE_DIR, "data", "predictions", "classification")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "test_predictions_classification.csv")
    predictions.to_csv(output_path, index=False)
    print(f"Предсказания сохранены: {output_path} ({len(predictions)} строк)")
    return output_path


if __name__ == "__main__":
    predict_test()
