import streamlit as st
import pandas as pd
import joblib


st.title("RiskScore Prediction")


@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    return df

@st.cache_resource
def load_model(file):
    model = joblib.load(file)
    return model

@st.cache_resource
def load_preprocessor(file):
    preprocessor = joblib.load(file)
    return preprocessor

upload_data = st.file_uploader("Choose CSV data", type='csv')
upload_model = st.file_uploader("Choose Model", type='joblib')
upload_preprocessor = st.file_uploader("Choose Preprocessor", type='joblib')

if (upload_data is not None) and (upload_model is not None) and (upload_preprocessor is not None):
    df = load_data(upload_data)
    model = load_model(upload_model)
    preprocessor = load_preprocessor(upload_preprocessor)

    st.subheader("Данные датасета")
    st.dataframe(df)

    identifiers = df['ID'].copy() if "ID" in df.columns else None
    features = df.drop(columns=['ID'], errors="ignore")
    features_proc = preprocessor.transform(features)
    y_pred = model.predict(features_proc)

    predictions = pd.DataFrame()
    if identifiers is not None:
        predictions["ID"] = identifiers.astype(str)
    predictions['RiskScore'] = y_pred

    st.write('Предсказания модели')
    st.dataframe(predictions)

    csv = predictions.to_csv(
        index=False,
        sep=';',
        decimal=',',
        float_format='%.4f'
        ).encode('utf-8')
    st.download_button(
        
        label="Скачать предсказания в CSV",
        data=csv,
        file_name='risk_predictions.csv',
        mime='text/csv',
    )
