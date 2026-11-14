import os
import joblib

from catboost import CatBoostClassifier

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, roc_auc_score, RocCurveDisplay, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.preprocessing import get_forest_pipeline
from config import BASE_DIR, PROCESSED_DIR, MODELS_DIR


def train_cat_clf_model() -> None:
    path = os.path.join(PROCESSED_DIR, 'train_features_classification.csv')
    df = pd.read_csv(path)

    preprocessor = get_forest_pipeline()
    model = CatBoostClassifier(
        iterations=1500,
        learning_rate=0.05,
        depth=6,
        l2_leaf_reg=3.0,
        loss_function='Logloss',
        eval_metric="AUC",
        auto_class_weights="Balanced",
        random_state=42,
        verbose=False
    )

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])

    X = df.drop(columns=['isRisk', 'ID'], errors='ignore')
    y = df['isRisk']

    X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    print("Обучается CatBoostClassifier...")
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_val)
    y_prob = pipeline.predict_proba(X_val)[:, 1]

    print(classification_report(y_val, y_pred))

    auc_value = roc_auc_score(y_val, y_prob)
    print(f"ROC-AUC score: {auc_value}")

    images_dir = os.path.join(BASE_DIR, "images")
    os.makedirs(images_dir, exist_ok=True)


    fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_estimator(
        pipeline, 
        X_val, 
        y_val, 
        display_labels=["No Risk (0)", "Risk (1)"], 
        cmap="Blues", 
        ax=ax_cm
    )
    ax_cm.set_title("Confusion Matrix")
    fig_cm.savefig(
        os.path.join(images_dir, "confusion-matrix-catboost.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig_cm)


    fig_roc, ax_roc = plt.subplots(figsize=(10, 8))
    RocCurveDisplay.from_predictions(
        y_val,
        y_prob,
        name='CatBoost Classifier',
        ax=ax_roc,
        curve_kwargs={'color': 'darkred'}
    )

    ax_roc.plot([0, 1], [0, 1], 'k--', label='ROC-AUC=0.5')
    ax_roc.set_title('ROC-AUC Score')
    ax_roc.set_xlabel('FPR')
    ax_roc.set_ylabel('TPR')
    ax_roc.legend(loc='lower right')
    ax_roc.grid(True, linestyle='--', alpha=0.7)

    
    fig_roc.savefig(
        os.path.join(images_dir, "roc-auc-curve-catboost.png"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close(fig_roc)

    feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    clean_names = [name.split("__")[-1] for name in feature_names]
    importances = pipeline.named_steps["model"].get_feature_importance()

    fi_df = (
        pd.DataFrame({"feature": clean_names, "importance": importances})
        .sort_values(by="importance", ascending=False)
        .head(15)
    )

    fig_fi, ax_fi = plt.subplots(figsize=(10, 6))
    ax_fi.barh(fi_df["feature"][::-1], fi_df["importance"][::-1], color="#2b5c8f")
    ax_fi.set_title("Top-15 Feature Importances (CatBoost Classifier)")
    ax_fi.set_xlabel("Importance (%)")
    ax_fi.grid(axis="x", linestyle="--", alpha=0.6)

    fig_fi.savefig(
        os.path.join(images_dir, "feature-importance-catboost.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig_fi)


    models_dir = os.path.join(MODELS_DIR, 'classification')
    os.makedirs(models_dir, exist_ok=True)

    joblib.dump(pipeline, os.path.join(models_dir, "catboost_clf.joblib"))

if __name__ == "__main__":
    train_cat_clf_model()