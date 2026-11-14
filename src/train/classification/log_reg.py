import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, roc_auc_score, RocCurveDisplay, ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline

from src.preprocessing import get_linear_pipeline
from config import BASE_DIR, PROCESSED_DIR, MODELS_DIR

def train_log_reg_model():
    path = os.path.join(PROCESSED_DIR, "train_features_classification.csv")
    df = pd.read_csv(path)

    X = df.drop(columns=['isRisk'])
    y = df['isRisk']

    preprocessor = get_linear_pipeline()
    model = LogisticRegression(max_iter=1000)

    pipeline = Pipeline([
        ('preprocessor', preprocessor), 
        ('model', model)
        ])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipeline, X, y, cv=cv, scoring='roc_auc')

    print(f"ROC-AUC на каждом фолде: {cv_scores}")
    print(f"Средний ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})\n")

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_val)
    y_prob = pipeline.predict_proba(X_val)[:, 1]

    print("Обучается LogisticRegression...")
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
        os.path.join(images_dir, "confusion-matrix.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig_cm)


    fig_roc, ax_roc = plt.subplots(figsize=(10, 8))
    RocCurveDisplay.from_predictions(
        y_val,
        y_prob,
        name='Logistic Regression',
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
        os.path.join(images_dir, "roc-auc-curve.png"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close(fig_roc)

    feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    clean_names = [name.split("__")[-1] for name in feature_names]
    coefs = pipeline.named_steps["model"].coef_[0]

    fi_df = (
        pd.DataFrame({"feature": clean_names, "coef": coefs})
        .sort_values(by="coef", key=lambda s: s.abs(), ascending=False)
        .head(15)
    )

    fig_fi, ax_fi = plt.subplots(figsize=(10, 6))
    ax_fi.barh(fi_df["feature"][::-1], fi_df["coef"][::-1], color="#2b5c8f")
    ax_fi.axvline(0, color="black", linewidth=0.8)
    ax_fi.set_title("Top-15 Feature Coefficients (Logistic Regression)")
    ax_fi.set_xlabel("Coefficient Value (Log-Odds)")
    ax_fi.grid(axis="x", linestyle="--", alpha=0.6)

    fig_fi.savefig(
        os.path.join(images_dir, "feature-importance-logreg.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig_fi)


    models_dir = os.path.join(MODELS_DIR, 'classification')
    os.makedirs(models_dir, exist_ok=True)

    joblib.dump(pipeline, os.path.join(models_dir, "log_reg.joblib"))

if __name__ == "__main__":
    train_log_reg_model()