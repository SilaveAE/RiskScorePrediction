import os
import pandas as pd
import numpy as np
from config import PROCESSED_DIR


def build_features(data: pd.DataFrame) -> pd.DataFrame:
    """"
        Применение feature-инжиринга к данным: создаем бинарными признаки
        is_weekend, is_salary, is_vacation, финансовые коэффы
    """
    df = data.copy()

    df['ApplicationDate'] = pd.to_datetime(df['ApplicationDate'])
    df['Year'] = df['ApplicationDate'].dt.year
    df['Month'] = df['ApplicationDate'].dt.month
    df['DayOfWeek'] = df['ApplicationDate'].dt.dayofweek
    df['Day'] = df['ApplicationDate'].dt.day

    df['is_salary'] = df['Day'].isin([6, 7, 25, 26]).astype(int)
    df['is_vacation'] = df['Month'].isin([2, 7]).astype(int)
    df['is_weekend'] = df['DayOfWeek'].isin([4, 5, 6]).astype(int)

    df['LoanToIncomeRatio'] = df['MonthlyLoanPayment'] / (df['MonthlyIncome'] + 1e-5)
    df['RiskPlus'] = df['CreditScore'] * df['InterestRate']
    df['TotalLoanPayments'] = df['MonthlyLoanPayment'] + df['MonthlyDebtPayments']
    df['FreeMoney'] = df['MonthlyIncome'] - df['MonthlyLoanPayment'] - df['MonthlyDebtPayments']


    df['Income_per_person'] = df['MonthlyIncome'] / (df['NumberOfDependents'] + 1)
    df['Savings_to_Income'] = df['SavingsAccountBalance'] / (df['MonthlyIncome'] + 1e-5)
    df['Checking_to_Income'] = df['CheckingAccountBalance'] / (df['MonthlyIncome'] + 1e-5)
    df['NetWorth_to_Income'] = df['NetWorth'] / (df['MonthlyIncome'] * 12 + 1e-5)


    df.drop(columns=['ApplicationDate'], inplace=True)

    return df

def save_task(clean_file: str, prefix: str) -> None:
    """
    Читает очищенный файл и сохраняет наборы признаков для регрессии и классификации.
    """
    path = os.path.join(PROCESSED_DIR, clean_file)
    df = pd.read_csv(path)
    df_features = build_features(df)

    reg_path = os.path.join(PROCESSED_DIR, f"{prefix}_features_regression.csv")
    df_features.to_csv(reg_path, index=False)

    df_clf = df_features.copy()
    if 'RiskScore' in df_clf.columns:
        df_clf['isRisk'] = np.where(df_clf['RiskScore'] >= 50, 1, 0)
        df_clf.drop(columns=['RiskScore'], inplace=True)

    clf_path = os.path.join(PROCESSED_DIR, f"{prefix}_features_classification.csv")
    df_clf.to_csv(clf_path, index=False)


def main():

    os.makedirs(PROCESSED_DIR, exist_ok=True)

    print("Генерация признаков для train...")
    save_task("train_cleaned.csv", "train")

    print("Генерация признаков для test...")
    save_task("test_cleaned.csv", "test")

    print("Генерация признаков успешно завершена.")


if __name__ == "__main__":
    main()