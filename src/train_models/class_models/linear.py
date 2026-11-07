import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, roc_auc_score, RocCurveDisplay, ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.pipeline import make_pipeline

from config import BASE_DIR, PROCESSED_DIR
from src.transformer import get_linear_pipeline

def make_class():
    path = os.path.join(PROCESSED_DIR, "train_features_class.csv")
    df = pd.read_csv(path)

    X = df.drop(columns=['isRisk'])
    y = df['isRisk']
    preprocessor = get_linear_pipeline()
    model = LogisticRegression(max_iter=1000)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_model = make_pipeline(get_linear_pipeline(), LogisticRegression(max_iter=1000))
    cv_scores = cross_val_score(cv_model, X, y, cv=cv, scoring='roc_auc')

    print(f"ROC-AUC на каждом фолде: {cv_scores}")
    print(f"Средний ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})\n")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    X_train_proc = preprocessor.fit_transform(X_train)
    X_val_proc = preprocessor.transform(X_val)


    model.fit(X_train_proc, y_train)
    y_pred = model.predict(X_val_proc)
    y_prob = model.predict_proba(X_val_proc)[:, 1]

    print(classification_report(y_val, y_pred))

    auc_value = roc_auc_score(y_val, y_prob)
    print(f"ROC-AUC score: {auc_value}")


    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_estimator(
        model, 
        X_val_proc, 
        y_val, 
        display_labels=["No Risk (0)", "Risk (1)"], 
        cmap="Blues", 
        ax=ax
    )
    ax.set_title("Confusion Matrix")
    plt.show()


    fig, ax = plt.subplots(figsize=(10, 8))
    RocCurveDisplay.from_predictions(
        y_val,
        y_prob,
        name='Logistic Regression',
        ax=ax,
        color='darkred'
    )

    ax.plot([0, 1], [0, 1], 'k--', label='ROC-AUC=0.5')
    ax.set_title('ROC-AUC Score')
    ax.set_xlabel('FPR')
    ax.set_ylabel('TPR')
    ax.legend(loc='lower right')
    ax.grid(True, linestyle='--', alpha=0.7)

    images_dir = os.path.join(BASE_DIR, "images")
    os.makedirs(images_dir, exist_ok=True)
    fig.savefig(os.path.join(images_dir, "roc-auc-curve.png"), dpi=300, bbox_inches="tight")
    plt.show()

    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)

    joblib.dump(preprocessor, os.path.join(models_dir, "preprocessor_class.joblib"))
    joblib.dump(model, os.path.join(models_dir, "logistic_model.joblib"))

if __name__ == "__main__":
    make_class()