import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


CATEGORICAL_COLS = [
    "MaritalStatus",
    "HomeOwnershipStatus",
    "LoanPurpose",
    "EmploymentStatus",
    "EducationLevel",
]

LINEAR_NUM_COLS = [
    "Age",
    "CreditScore",
    "LoanDuration",
    "InterestRate",
    "Year",
    "Month",
    "DayOfWeek",
    "Day",
    "is_salary",
    "is_vacation",
    "is_weekend",
    "LoanToIncomeRatio",
    "RiskPlus",
    "TotalLoanPayments",
    "FreeMoney",
    "Income_per_person",
    "Savings_to_Income",
    "Checking_to_Income",
    "NetWorth_to_Income",
]

TREE_NUM_COLS = LINEAR_NUM_COLS + ['MonthlyIncome']


def get_linear_pipeline() -> ColumnTransformer:
    """
        Пайплайн препроцессинга для линейных моделей (Ridge, LogisticRegression).
    """

    standard_num_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")), 
        ("scaler", StandardScaler()),
    ])
    log_num_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy='median')),
        ("log", FunctionTransformer(np.log1p, feature_names_out='one-to-one')),
        ("scaler", StandardScaler()),
    ])
    cat_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy='constant', fill_value='Unknown')),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    return ColumnTransformer(transformers=[
        ("num", standard_num_transformer, LINEAR_NUM_COLS),
        ("log", log_num_transformer, ['MonthlyIncome']),
        ("cat", cat_transformer, CATEGORICAL_COLS),
    ],
    remainder="drop",
    )

def get_forest_pipeline() -> ColumnTransformer:
    """
        Пайплайн препроцессинга для деревянных моделей (Random Forest, CatBoost).
    """

    standard_num_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")), 
        ("scaler", StandardScaler()),
    ])
    cat_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy='constant', fill_value='Unknown')),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    return ColumnTransformer(transformers=[
        ("num", standard_num_transformer, TREE_NUM_COLS),
        ("cat", cat_transformer, CATEGORICAL_COLS),
    ],
    remainder="drop",
    )