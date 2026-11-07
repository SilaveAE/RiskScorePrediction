import os
import pandas as pd
import numpy as np
from config import PROCESSED_DIR


def date_features(data: pd.DataFrame, is_classification: bool) -> pd.DataFrame:
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

    if is_classification and 'RiskScore' in df.columns:
        df['isRisk'] = np.where(df['RiskScore'] >= 50, 1, 0)
        df.drop(columns=['RiskScore'], inplace=True)

    df.drop(columns=['ApplicationDate'], inplace=True)

    return df


def main():

    os.makedirs(PROCESSED_DIR, exist_ok=True)

    print("Фич-инжиринг train_cleaned.csv для линейной модели...")
    path_train = os.path.join(PROCESSED_DIR, 'train_cleaned.csv')
    df_train = pd.read_csv(path_train)
    df_train = date_features(df_train, is_classification=False)
    df_train.to_csv(os.path.join(PROCESSED_DIR, "train_features_linear.csv"), index=False)
    print("Фич-инжиринг train завершен.")

    print("Фич-инжиринг test_cleaned.csv для линейной модели...")
    path_test = os.path.join(PROCESSED_DIR, 'test_cleaned.csv')
    df_test = pd.read_csv(path_test)
    df_test = date_features(df_test, is_classification=False)
    df_test.to_csv(os.path.join(PROCESSED_DIR, "test_features_linear.csv"), index=False)
    print("Фич-инжиринг test завершен.")


    
    print("Фич-инжиринг train_cleaned.csv для классификации...")
    path_train = os.path.join(PROCESSED_DIR, 'train_cleaned.csv')
    df_train = pd.read_csv(path_train)
    df_train = date_features(df_train, is_classification=True)
    df_train.to_csv(os.path.join(PROCESSED_DIR, "train_features_class.csv"), index=False)
    print("Фич-инжиринг train завершен.")

    print("Фич-инжиринг test_cleaned.csv для классификации...")
    path_test = os.path.join(PROCESSED_DIR, 'test_cleaned.csv')
    df_test = pd.read_csv(path_test)
    df_test = date_features(df_test, is_classification=True)
    df_test.to_csv(os.path.join(PROCESSED_DIR, "test_features_class.csv"), index=False)
    print("Фич-инжиринг test завершен.")



if __name__ == "__main__":
    main()