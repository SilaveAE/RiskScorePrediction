import argparse
from src.make_dataset import main as run_make_dataset
from src.feature_engineering import main as run_feature_engineering
from src.train.regression.ridge_cv import train_ridge_model
from src.train.regression.random_forest import train_forest_model
from src.train.regression.catboost_reg import train_cat_reg_model
from src.train.classification.log_reg import train_log_reg_model
from src.train.classification.catboost_clf import train_cat_clf_model
from src.predict import run_inference, MODEL_REGISTRY


def step_data() -> None:
    """
    Очистка сырых данных и генерация признаков.
    """
    print('[1/2] Очистка данных.')
    run_make_dataset()
    print('[2/2] Генерация признаков.')
    run_feature_engineering()


def step_train() -> None:
    """
    Обучение всех моделей регрессии и классификации.
    """
    print("Обучение моделей регрессии.")
    train_ridge_model()
    print()
    train_forest_model()
    print()
    train_cat_reg_model()
    print()

    print("Обучение моделей классификации.")
    train_log_reg_model()
    print()
    train_cat_clf_model()


def step_predict(task: str | None = None, model: str | None = None) -> None:
    """
    Инференс на тестовой выборке.
    """

    if task and model:
        run_inference(task, model)
        return

    print("Запуск инференса")
    for curr_task, cfg in MODEL_REGISTRY.items():
        if task and curr_task != task:
            continue
        for curr_model in cfg["models"]:
            run_inference(curr_task, curr_model)


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Credit Scoring'
    )
    parser.add_argument(
        "command",
        choices=["all", "data", "train", "predict"],
        help='Этапы пайплайна: all, data, train, predict'
    )
    parser.add_argument(
        "--task",
        choices=["regression", "classification"],
        default=None,
        help='Задача для predict'
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help='Имя модели для predict (пр.: ridge_cv, random_forest, log_reg, ...)'
    )

    args = parser.parse_args()

    if args.command == 'all':
        step_data()
        print()
        step_train()
        print()
        step_predict()
    elif args.command == 'data':
        step_data()
    elif args.command == 'train':
        step_train()
    elif args.command == "predict":
        step_predict(task=args.task, model=args.model)

if __name__ == '__main__':
    main()