import os
import joblib
import streamlit as st
import pandas as pd

from config import MODELS_DIR, PROCESSED_DIR
from src.predict import MODEL_REGISTRY

st.set_page_config(
    page_title='Credit Risk Scoring',
    layout='wide'
)

st.title("Credit Risk Scoring - Inference")

@st.cache_resource
def load_pipeline(task: str, model: str):
    model_file = MODEL_REGISTRY[task]['models'][model]
    model_path = os.path.join(MODELS_DIR, task, model_file)
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Модель не найдена по пути {model_path}. Сначала выполните pythom main.py train")

    return joblib.load(model_path)

@st.cache_data
def load_csv(source) -> pd.DataFrame:
    return pd.read_csv(source)


st.sidebar.header("Параметры инференса")

task_labels = {
    "classification": 'Классификация (вероятность RiskScore)',
    "regression": "Регрессия (прогноз RiskScore)"
}
selected_task = st.sidebar.selectbox(
    "Выбери задачу: ",
    options=list(MODEL_REGISTRY.keys()),
    format_func=lambda x: task_labels.get(x, x)
)

available_models = list(MODEL_REGISTRY[selected_task]["models"].keys())
selected_model = st.sidebar.selectbox(
    "Выбери модель:",
    options=available_models
)

threshold = 0.5
if selected_task == 'classification':
    threshold = st.sidebar.slider(
        "Порог отсечения риска:",
        min_value=0.10,
        max_value=0.90,
        value=0.50,
        step=0.05,
        help='Клиенты выше порога - рискованные клиенты (isRisk = 1)'
    )

st.sidebar.divider()
data_source = st.sidebar.radio(
    "Источник данных:",
    options=["Демо-датасет (test)", "Загрузить свой CSV"]
)

df = None
if data_source == "Демо-датасет (test)":
    demo_file = MODEL_REGISTRY[selected_task]['features_file']
    demo_path = os.path.join(PROCESSED_DIR, demo_file)

    if os.path.isfile(demo_path):
        df = load_csv(demo_path)
        st.info(f"Используется демо-датасет: {demo_file} c {len(df)} строк.")
    else:
        st.error(f"Файл {demo_path} не найден. Запустите python main.py data.")

else:
    uploaded_file = st.file_uploader("Загрузите CSV", type=['csv'])
    if uploaded_file is not None:
        df = load_csv(uploaded_file)

if df is not None:
    try:
        pipeline = load_pipeline(selected_task, selected_model)
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()

    with st.expander("Посмотреть входные данные (10 строк)", expanded=False):
        st.dataframe(df.head(10), use_container_width=True)

    identifiers = df['ID'].copy() if 'ID' in df.columns else None
    features = df.drop(columns=['ID', 'RiskScore', 'isRisk'], errors='ignore')

    predictions = pd.DataFrame()
    if identifiers is not None:
        predictions['ID'] = identifiers.astype(str)

    if selected_task == 'regression':
        preds = pipeline.predict(features)
        predictions['RiskScore'] = preds

        col1, col2, col3 = st.columns(3)
        col1.metric("Всего заявок", len(predictions))
        col2.metric("Средний Riskscore", f"{preds.mean():.2f}")
        col3.metric("Доля высокого риска >= 50", f"{(preds >= 50).mean() * 100:.1f}%")

        st.subheader("Распределние RiskScore")
        st.bar_chart(pd.Series(preds).round().value_counts().sort_index())
    else:
        pos_idx = list(pipeline.classes_).index(1)
        probs = pipeline.predict_proba(features)[:, pos_idx]
        labels = (probs >= threshold).astype(int)

        predictions['isRisk'] = labels
        predictions['probability'] = probs.round(4)

        col1, col2, col3 = st.columns(3)
        col1.metric("Всего заявок", len(predictions))
        col2.metric("Выявлено рисковых (isRisk = 1)", int(labels.sum()))
        col3.metric("Доля рисковых заявок", f"{labels.mean() * 100:.1f}%")

        st.subheader("Распределение вероятностей дефолта")
        hist_bins = pd.cut(probs, bins=20).value_counts().sort_index()
        hist_bins.index = [f"{interval.left:.2f}-{interval.right:.2f}" for interval in hist_bins.index]
        st.bar_chart(hist_bins)

    st.subheader("Таблица предсказаний")
    st.dataframe(predictions.head(50), use_container_width=True)

    csv_data = predictions.to_csv(
        index=False,
        sep=";",
        decimal=",",
        float_format="%.4f",
    ).encode("utf-8")

    st.download_button(
        label="Скачать полные предсказания в CSV",
        data=csv_data,
        file_name=f"predictions_{selected_task}_{selected_model}.csv",
        mime="text/csv",
    )