import streamlit as st
import pandas as pd
import statsmodels.formula.api as smf
import os

# Configuración de la página
st.set_page_config(page_title="Dashboard Econométrico - Pobreza", layout="wide")

st.title("📊 Determinantes de la Pobreza por Ingresos en el Ecuador")
st.markdown("### Análisis Econométrico Mediante Modelos Logit y Probit")

# Cargar Datos con manejo de ruta
@st.cache_data
def cargar_datos():
    ruta = "data/processed/enemdu_limpia.csv"
    if os.path.exists(ruta):
        return pd.read_csv(ruta)
    else:
        st.error(f"⚠️ No se encontró el archivo de datos en: `{ruta}`")
        return None

df = cargar_datos()

if df is not None:
    # Sidebar - Simulación de Probabilidad
    st.sidebar.header("⚙️ Simulación de Probabilidad")
    st.sidebar.markdown("Ajusta las características socioeconómicas:")

    educ = st.sidebar.slider("Años de Educación", 0, 20, 8)
    edad = st.sidebar.slider("Edad", 18, 80, 35)
    horas = st.sidebar.slider("Horas Trabajadas / Semana", 1, 80, 40)
    mujer = st.sidebar.selectbox("¿Es Mujer?", [0, 1], format_func=lambda x: "Sí" if x == 1 else "No")
    rural = st.sidebar.selectbox("¿Sector Rural?", [0, 1], format_func=lambda x: "Sí" if x == 1 else "No")
    informal = st.sidebar.selectbox("¿Trabajo Informal?", [0, 1], format_func=lambda x: "Sí" if x == 1 else "No")

    # Estimación rápida para la simulación
    formula = "pobreza_ing ~ anios_educ + edad + es_mujer + es_rural + es_informal + horas_trabajo"
    modelo = smf.logit(formula, data=df).fit(disp=0)

    # Predicción
    nuevo_ind = pd.DataFrame({
        'anios_educ': [educ], 'edad': [edad], 'es_mujer': [mujer],
        'es_rural': [rural], 'es_informal': [informal], 'horas_trabajo': [horas]
    })
    prob_predicha = modelo.predict(nuevo_ind)[0] * 100

    # Interfaz Principal
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("🎯 Resultado de la Predicción")
        st.metric(label="Probabilidad Estimada de Pobreza", value=f"{prob_predicha:.2f}%")
        if prob_predicha > 50:
            st.error("⚠️ Persona en Situación de Alto Riesgo / Vulnerabilidad.")
        else:
            st.success("✅ Bajo Riesgo de Pobreza por Ingresos.")
        
        st.write("---")
        st.write(f"**Muestra Total:** {len(df)} observaciones (ENEMDU).")

    with col2:
        st.subheader("📈 Efectos Marginales Promedio (AME)")
        ruta_img = "outputs/figures/efectos_marginales.png"
        if os.path.exists(ruta_img):
            st.image(ruta_img, use_container_width=True)
        else:
            st.warning("📊 Gráfica de efectos marginales no encontrada en `outputs/figures/`.")

    # Tabla de Datos
    with st.expander("👀 Ver Datos Limpios Procesados"):
        st.dataframe(df.head(15))
        