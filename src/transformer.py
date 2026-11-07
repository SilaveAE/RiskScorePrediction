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

# Идеально отфильтрованный список числовых фич БЕЗ коллинеарного взрыва
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
    "RiskPlus",
    "TotalLoanPayments",
    "Income_per_person",
    "Savings_to_Income",
    "Checking_to_Income",
    "NetWorth_to_Income",
]

def log_transformation(x):
    return np.log1p(np.clip(x, 0, None))

def get_linear_pipeline() -> ColumnTransformer:
    """
        Пайплайн для лин. регрессии.
    """

    standard_num_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")), 
        ("scaler", StandardScaler()),
    ])
    log_num_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy='median')),
        ("log", FunctionTransformer(log_transformation)),
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
        Пайплайн для леса.
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
        ("num", standard_num_transformer, LINEAR_NUM_COLS),
        ("cat", cat_transformer, CATEGORICAL_COLS),
    ],
    remainder="drop",
    )