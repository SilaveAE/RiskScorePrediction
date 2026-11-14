import os
import joblib
import pandas as pd
from config import BASE_DIR, PROCESSED_DIR, MODELS_DIR

MODEL_REGISTRY = {
    'regression': {
        'features_file': 'test_features_regression.csv',
        'models': {
            'ridge_cv': 'ridge_cv.joblib',
            'random_forest': 'random_forest.joblib',
            'catboost': 'catboost_reg.joblib'
        },
    },
    'classification': {
        'features_file': 'test_features_classification.csv',
        'models': {
            'log_reg': 'log_reg.joblib',
            'catboost': 'catboost_clf.joblib'
        }
    }
}


def run_inference(task: str, model: str) -> str:
    if task not in MODEL_REGISTRY:
        raise ValueError(f'Неизвестная задача {task}. Доступны {list(MODEL_REGISTRY)}')

    task_cfg = MODEL_REGISTRY[task]
    if model not in task_cfg['models']:
        raise ValueError(f'Модель {model} не найдена для {task}. Доступны {list(task_cfg["models"])}')

    test_path = os.path.join(PROCESSED_DIR, task_cfg['features_file'])
    model_path = os.path.join(MODELS_DIR, task, task_cfg['models'][model])

    for path in (test_path, model_path):
        if not os.path.isfile(path):
            raise FileNotFoundError(f'Файл не найден: {path}')

    df = pd.read_csv(test_path)
    if df.empty:
        raise ValueError(f'Датасет пуст: {test_path}')

    identifiers = df['ID'].copy() if "ID" in df.columns else None
    features = df.drop(columns=['ID'], errors='ignore')

    pipeline = joblib.load(model_path)
    predictions = pd.DataFrame()

    if identifiers is not None:
        predictions['ID'] = identifiers

    if task == 'regression':
        predictions['RiskScore'] = pipeline.predict(features)
    elif task == 'classification':
        predictions['isRisk'] = pipeline.predict(features)
        pos_idx = list(pipeline.classes_).index(1)
        predictions['Probability'] = pipeline.predict_proba(features)[:, pos_idx]

    output_dir = os.path.join(BASE_DIR, 'data', 'predictions', task)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'predictions_{model}.csv')

    predictions.to_csv(output_path, index=False)
    print(f"[{task} | {model}] сохранено {len(predictions)} строк -> {output_path}")
    return output_path