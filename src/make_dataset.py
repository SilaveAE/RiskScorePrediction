import os
import pandas as pd
from config import RAW_DIR, PROCESSED_DIR, COLS_TO_DROP

def load_clean_data(file_name: str, is_train: bool = True) -> pd.DataFrame:
    """
        Загружаем сырые данные, удаляем дубликаты, очищаем таргет от NA строк 
        и обрезаем от 0 до 100, удаляем мультиколлинеарные признаки.
    """

    #0. Проверка пути
    path = os.path.join(RAW_DIR, file_name)
    if not os.path.exists(path):
        raise FileNotFoundError(f'Файл не найден по пути: {path}')

    #1. Считываем датасет
    df = pd.read_csv(path)

    #2. Удаляем дубликаты
    df.drop_duplicates(inplace=True)

    #3. Если train выборка
    if is_train:
        target = 'RiskScore'
        df.dropna(subset=target, inplace=True)
        df = df[(df[target] >= 0) & (df[target] <= 100)]

    #4. Очистка дублёров
    df.drop(columns=COLS_TO_DROP, errors='ignore', inplace=True)

    return df

def main():
    """
        Очищаем датасеты train и test.
    """
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    print("Обработка train.csv")
    train_df = load_clean_data("train.csv", is_train=True)
    train_df.to_csv(os.path.join(PROCESSED_DIR, 'train_cleaned.csv'), index=False)
    print(f"Первичная обработка train.csv завершена. Файл сохранен как train_cleaned.csv в папку {PROCESSED_DIR}.")


    print("Обработка test.csv")
    test_df = load_clean_data("test.csv", is_train=False)
    test_df.to_csv(os.path.join(PROCESSED_DIR, 'test_cleaned.csv'), index=False)
    print(f"Первичная обработка test.csv завершена. Файл сохранен как test_cleaned.csv в папку {PROCESSED_DIR}.")


if __name__ == "__main__":
    main()