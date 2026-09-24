"""
App Streamlit para predicción de subsidios de vivienda.
Ejecutar con: streamlit run app/app.py
"""
import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

# Configuración de página
st.set_page_config(
    page_title="Predicción Subsidios de Vivienda",
    page_icon="🏠",
    layout="wide"
)

# Ruta a los modelos
MODELOS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
MODEL_PATH = os.path.join(MODELOS_DIR, 'xgb_mejor_modelo.pkl')
INFO_PATH = os.path.join(MODELOS_DIR, 'modelo_info.pkl')


@st.cache_resource
def cargar_modelo():
    modelo = joblib.load(MODEL_PATH)
    info = joblib.load(INFO_PATH)
    return modelo, info


modelo, info = cargar_modelo()

# ============== Header ==============
st.title('🏠 Predicción de Subsidios de Mejoramiento de Vivienda')
st.markdown('**Minería de Datos - UPB 2026** | Juan David Acevedo - Diego A. Martinez')
st.markdown('---')
st.markdown('### 📊 Rendimiento del modelo')
col1, col2, col3 = st.columns(3)
col1.metric('RMSE', f"${info['rmse_test']:,.0f} COP")
col2.metric('MAE', f"${info['mae_test']:,.0f} COP")
col3.metric('R²', f"{info['r2_test']:.4f}")
st.markdown('---')

# ============== Sidebar: inputs ==============
st.sidebar.header('🧾 Características del Beneficiario')


def user_input_form():
    form_data = {}
    for feature in info['features']:
        opciones = sorted(info['categorias_unicas'][feature])
        valor = st.sidebar.selectbox(feature, opciones, key=f'inp_{feature}')
        form_data[feature] = valor
    return pd.DataFrame([form_data])


input_df = user_input_form()

# ============== Predicción ==============
st.header('🔮 Predicción')
if st.button('Predecir valor del subsidio', type='primary'):
    prediccion = modelo.predict(input_df)[0]
    st.success(f'**Valor estimado del subsidio:** ${prediccion:,.0f} COP')
    st.markdown('---')
    st.markdown('#### 📋 Perfil del beneficiario ingresado')
    st.dataframe(input_df.T.rename(columns={0: 'Valor'}), use_container_width=True)

# ============== Importancia de features ==============
st.markdown('---')
st.header('📈 Importancia de Variables (XGBoost)')
try:
    xgb_model = modelo.named_steps['modelo']
    encoder = modelo.named_steps['pre']
    feature_names_out = encoder.get_feature_names_out()
    importancias = xgb_model.feature_importances_
    if len(feature_names_out) == len(importancias):
        df_imp = pd.DataFrame({
            'Feature': feature_names_out,
            'Importancia': importancias
        }).sort_values('Importancia', ascending=True).tail(15)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(df_imp['Feature'], df_imp['Importancia'], color='#1D3557')
        ax.set_xlabel('Importancia')
        ax.set_title('Top 15 Features más importantes')
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info(f'XGBoost - {len(importancias)} features importances')
except Exception as e:
    st.warning(f'No se pudo graficar importancia: {e}')

# ============== Estadísticas dataset ==============
st.markdown('---')
st.header('📚 Estadísticas del dataset')
try:
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'data',
                                  'dataset_procesado.csv'))
    st.write(f'**Total de registros:** {len(df):,}')
    st.write(f'**Distribución del valor del subsidio:**')
    st.dataframe(df['VALOR_SUBSIDIO'].describe().to_frame()
                 .applymap(lambda x: f'${x:,.0f}'))
except Exception as e:
    st.info('Dataset no disponible.')

st.markdown('---')
st.caption('Proyecto académico - Minería de Datos UPB 2026')
