"""Vista Técnica — Métricas del modelo ML y clustering."""

import requests
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Vista Técnica · Sephora", page_icon="🔬", layout="wide")

# ---------------------------------------------------------------------------
# Estilos — paleta lavanda + azul pálido
# ---------------------------------------------------------------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,700;1,400&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #F8F6FF;
}
h1, h2, h3 { font-family: 'Playfair Display', serif; }

[data-testid="stSidebar"] {
    background: linear-gradient(160deg, #F3E8FF 0%, #E8EAF6 100%);
    border-right: 1px solid #C9B1FF;
}

[data-testid="stMetric"] {
    background: white;
    border-radius: 16px;
    padding: 1rem;
    border: 1px solid #C9B1FF;
    box-shadow: 0 2px 12px rgba(121, 80, 200, 0.07);
}
[data-testid="stMetricLabel"] { color: #5E35B1; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.8px; }
[data-testid="stMetricValue"] { font-family: 'Playfair Display', serif; color: #7B5EA7; }

.section-title {
    font-family: 'Playfair Display', serif;
    color: #7B5EA7;
    font-size: 1.4rem;
    margin-bottom: 0.5rem;
}
.divider {
    height: 2px;
    background: linear-gradient(90deg, transparent, #C9B1FF, #90CAF9, transparent);
    margin: 1.5rem 0;
    border: none;
}
.param-box {
    background: white;
    border-radius: 12px;
    padding: 1rem 1.5rem;
    border: 1px solid #E8EAF6;
    box-shadow: 0 2px 8px rgba(121, 80, 200, 0.06);
    margin-bottom: 0.5rem;
}
.param-label { color: #9575CD; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.6px; }
.param-value { color: #311B92; font-size: 1.1rem; font-weight: 500; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

API_BASE = "http://localhost:8000"

@st.cache_data(ttl=300)
def get_metrics():
    r = requests.get(f"{API_BASE}/metrics/model", timeout=5)
    return r.json() if r.status_code == 200 else {}

@st.cache_data(ttl=300)
def get_clusters():
    r = requests.get(f"{API_BASE}/clusters", timeout=5)
    return pd.DataFrame(r.json()) if r.status_code == 200 else pd.DataFrame()

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.markdown("## 🔬 Vista Técnica")
st.markdown("<p style='color:#7B5EA7; font-size:1rem'>Métricas del modelo de Machine Learning y resultados del análisis no supervisado.</p>", unsafe_allow_html=True)
st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Cargar datos
# ---------------------------------------------------------------------------

with st.spinner("Cargando métricas..."):
    metrics = get_metrics()
    df_clusters = get_clusters()

if not metrics:
    st.error("No se pudo conectar a la API. Asegúrate de que uvicorn está corriendo.")
    st.stop()

# ---------------------------------------------------------------------------
# Modelo supervisado
# ---------------------------------------------------------------------------

st.markdown("<div class='section-title'>🤖 Modelo Supervisado</div>", unsafe_allow_html=True)

col1, col2 = st.columns([2, 3])

with col1:
    st.markdown("<div class='param-box'><div class='param-label'>Modelo</div><div class='param-value'>" + str(metrics.get("model_name", "N/A")) + "</div></div>", unsafe_allow_html=True)

    params = {
        "N° Estimadores":  metrics.get("n_estimators", "N/A"),
        "Profundidad máx": metrics.get("max_depth", "N/A"),
        "Learning Rate":   metrics.get("learning_rate", "N/A"),
        "Random State":    metrics.get("random_state", "N/A"),
    }
    for label, val in params.items():
        st.markdown(f"<div class='param-box'><div class='param-label'>{label}</div><div class='param-value'>{val}</div></div>", unsafe_allow_html=True)

with col2:
    st.markdown("**¿Qué predice este modelo?**")
    st.markdown("""
    El **Gradient Boosting Classifier** predice si un usuario recomendará o no un producto
    basándose en características como el precio, categoría, marca, y patrones de comportamiento
    de otros compradores.

    El modelo fue entrenado en el EP2 del proyecto usando técnicas de validación cruzada
    y optimización de hiperparámetros (Grid Search + Random Search).
    """)

    st.info("💡 Los parámetros mostrados corresponden al mejor modelo encontrado durante la optimización con `n_estimators=100`, `max_depth=3` y `learning_rate=0.1`.")

st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Clustering K-Means
# ---------------------------------------------------------------------------

st.markdown("<div class='section-title'>🔵 Análisis de Clustering K-Means</div>", unsafe_allow_html=True)

if not df_clusters.empty:
    m1, m2, m3 = st.columns(3)

    best = df_clusters[df_clusters["is_best"] == True].iloc[0]
    m1.metric("K óptimo", int(best["k"]))
    m2.metric("Silhouette Score", f"{best['silhouette']:.4f}")
    m3.metric("Inercia final", f"{best['inertia']:,.0f}")

    col3, col4 = st.columns(2)

    with col3:
        st.markdown("**Silhouette Score por K**")
        fig_sil = px.line(
            df_clusters,
            x="k", y="silhouette",
            markers=True,
            color_discrete_sequence=["#9575CD"],
            labels={"k": "Número de clusters (k)", "silhouette": "Silhouette Score"},
        )
        # Marcar el mejor k
        best_row = df_clusters[df_clusters["is_best"] == True]
        fig_sil.add_scatter(
            x=best_row["k"], y=best_row["silhouette"],
            mode="markers",
            marker=dict(size=14, color="#FF6B9D", symbol="star"),
            name="K óptimo",
        )
        fig_sil.update_layout(
            paper_bgcolor="#F8F6FF", plot_bgcolor="#F8F6FF",
            margin=dict(l=0, r=0, t=10, b=10),
            height=280,
            showlegend=True,
        )
        st.plotly_chart(fig_sil, use_container_width=True)

    with col4:
        st.markdown("**Inercia por K (Método del Codo)**")
        fig_ine = px.line(
            df_clusters,
            x="k", y="inertia",
            markers=True,
            color_discrete_sequence=["#7986CB"],
            labels={"k": "Número de clusters (k)", "inertia": "Inercia"},
        )
        fig_ine.add_scatter(
            x=best_row["k"], y=best_row["inertia"],
            mode="markers",
            marker=dict(size=14, color="#FF6B9D", symbol="star"),
            name="K óptimo",
        )
        fig_ine.update_layout(
            paper_bgcolor="#F8F6FF", plot_bgcolor="#F8F6FF",
            margin=dict(l=0, r=0, t=10, b=10),
            height=280,
            showlegend=True,
        )
        st.plotly_chart(fig_ine, use_container_width=True)

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # Tabla completa
    st.markdown("**Resumen completo de evaluación de K**")
    df_display = df_clusters.copy()
    df_display["is_best"] = df_display["is_best"].apply(lambda x: "⭐ Óptimo" if x else "")
    df_display["silhouette"] = df_display["silhouette"].round(4)
    df_display["inertia"] = df_display["inertia"].apply(lambda x: f"{x:,.0f}")
    df_display.columns = ["K", "Inercia", "Silhouette", "Seleccionado"]
    st.dataframe(df_display, use_container_width=True, hide_index=True)

st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Métricas de PCA
# ---------------------------------------------------------------------------

st.markdown("<div class='section-title'>📊 Reducción de Dimensionalidad (PCA)</div>", unsafe_allow_html=True)

pca_2d      = metrics.get("pca_variance_2d", 0)
best_k_val  = metrics.get("best_k", 2)
inertia_val = metrics.get("final_inertia", 0)

p1, p2, p3 = st.columns(3)
p1.metric("Varianza explicada (2D)", f"{pca_2d*100:.1f}%")
p2.metric("Componentes para 90% varianza", "33")
p3.metric("Componentes para 95% varianza", "55")

st.markdown("""
<div style='background:white; border-radius:12px; padding:1rem 1.5rem; border:1px solid #E8EAF6; margin-top:0.5rem'>
<p style='color:#555; font-size:0.9rem'>
Los primeros 2 componentes principales explican el <b>{:.1f}%</b> de la varianza total del dataset.
Aunque es una representación comprimida, es suficiente para visualizar la separación entre clusters.
Para capturar el <b>90%</b> de la información se necesitan <b>33 componentes</b>, lo que refleja la
alta dimensionalidad del espacio de características de productos cosméticos.
</p>
</div>
""".format(pca_2d * 100), unsafe_allow_html=True)
