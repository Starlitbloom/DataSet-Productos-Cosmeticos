"""Vista Técnica — Métricas del modelo ML y clustering."""

import requests
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Vista Técnica · Sephora", page_icon="🔬", layout="wide")

# ---------------------------------------------------------------------------
# Estilos editoriales — acento lavanda
# ---------------------------------------------------------------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,300;0,400;0,500;1,300;1,400&family=Jost:wght@300;400;500&display=swap');

html, body, [class*="css"], .stApp {
    font-family: 'Jost', sans-serif;
    background: linear-gradient(160deg, #FAF8FD 0%, #F8F5FB 100%) !important;
    color: #1A1A1A;
}
#MainMenu, footer { visibility: hidden; }
header { visibility: hidden; }
button[kind="headerNoPadding"],
[data-testid="stSidebarCollapseButton"],
[data-testid="collapsedControl"] { visibility: visible !important; display: flex !important; }

.block-container { padding: 2rem 3rem 4rem 3rem !important; max-width: 1150px; }

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #FFF0F5 0%, #FDF6FF 100%) !important;
    border-right: 1px solid #F0D6E0 !important;
}
[data-testid="stSidebarNav"] { display: none !important; }
[data-testid="stSidebar"] [data-testid="stPageLink"] a,
[data-testid="stSidebar"] [data-testid="stPageLink"] p {
    font-family: 'Jost', sans-serif !important;
    font-size: 0.75rem !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    color: #5A3A4A !important;
    padding: 0.5rem 1.5rem !important;
}
[data-testid="stSidebar"] [data-testid="stPageLink"]:hover a,
[data-testid="stSidebar"] [data-testid="stPageLink"]:hover p { color: #C2185B !important; }

/* Metricas — cards blancas con sombra lavanda */
[data-testid="stMetric"] {
    background: linear-gradient(145deg, #FFFFFF 0%, #FBF9FF 100%) !important;
    border: 1px solid #E4D9F5 !important;
    border-radius: 16px !important;
    padding: 1.3rem 1.4rem !important;
    box-shadow: 0 8px 24px rgba(123, 94, 167, 0.10), 0 1px 3px rgba(212, 160, 90, 0.06) !important;
}
[data-testid="stMetricLabel"] {
    font-family: 'Jost', sans-serif !important;
    color: #9B7FB5 !important;
    font-size: 0.75rem !important;
    text-transform: uppercase;
    letter-spacing: 1.5px;
}
[data-testid="stMetricValue"] {
    font-family: 'Cormorant Garamond', serif !important;
    color: #1A1A1A !important;
    font-size: 2.1rem !important;
    font-weight: 500 !important;
}

/* Contenedores reales con borde -> cards */
[data-testid="stVerticalBlockBorderWrapper"] {
    background: linear-gradient(145deg, #FFFFFF 0%, #FBF9FF 100%) !important;
    border-radius: 18px !important;
    border: 1px solid #E4D9F5 !important;
    box-shadow: 0 10px 30px rgba(123, 94, 167, 0.09), 0 1px 4px rgba(212, 160, 90, 0.06) !important;
    padding: 0.3rem !important;
}

.eyebrow {
    font-size: 0.68rem; letter-spacing: 4px; text-transform: uppercase;
    color: #7B5EA7; margin-bottom: 0.8rem;
}
.section-title {
    font-family: 'Cormorant Garamond', serif;
    color: #1A1A1A;
    font-size: 1.55rem;
    font-weight: 500;
    margin-bottom: 0.2rem;
}
.section-title::after {
    content: '';
    display: block;
    width: 32px;
    height: 2px;
    background: linear-gradient(90deg, #7B5EA7, #D4A55A);
    margin-top: 0.5rem;
    margin-bottom: 1rem;
}

.divider-floral {
    display: flex; align-items: center; gap: 12px; margin: 2.5rem 0;
}
.divider-floral::before, .divider-floral::after {
    content: ''; flex: 1; height: 1px; background: linear-gradient(90deg, transparent, #C9B1E0);
}
.divider-floral::after { background: linear-gradient(90deg, #C9B1E0, transparent); }
.divider-floral span { color: #7B5EA7; font-size: 1rem; }

.param-box {
    background: linear-gradient(145deg, #FFFFFF 0%, #FBF9FF 100%);
    border-radius: 14px;
    border: 1px solid #E4D9F5;
    padding: 1rem 1.4rem;
    margin-bottom: 0.7rem;
    box-shadow: 0 4px 16px rgba(123, 94, 167, 0.07);
}
.param-label { color: #9B7FB5; font-size: 0.7rem; text-transform: uppercase; letter-spacing: 1px; }
.param-value { font-family: 'Cormorant Garamond', serif; color: #1A1A1A; font-size: 1.4rem; font-weight: 500; margin-top: 0.2rem; }

.note-box {
    background: linear-gradient(145deg, #FFFFFF 0%, #FBF9FF 100%);
    border-radius: 14px;
    border-left: 4px solid #7B5EA7;
    border-top: 1px solid #E4D9F5; border-right: 1px solid #E4D9F5; border-bottom: 1px solid #E4D9F5;
    padding: 1.1rem 1.5rem;
    color: #555;
    font-size: 0.88rem;
    line-height: 1.7;
    font-weight: 300;
    box-shadow: 0 6px 20px rgba(123, 94, 167, 0.08);
}
.note-box b { color: #5E3F8A; font-weight: 600; }
.note-box code { background: #F0E8FA; color: #6B4E94; padding: 2px 6px; border-radius: 4px; font-size: 0.82rem; }

[data-testid="stDataFrame"] {
    border-radius: 16px;
    overflow: hidden;
    box-shadow: 0 8px 24px rgba(123, 94, 167, 0.08);
    border: 1px solid #E4D9F5;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("""
    <div style='padding: 2.5rem 1.5rem 0.5rem 1.5rem;'>
        <p style='font-family: Cormorant Garamond, serif; font-size: 0.75rem; letter-spacing: 4px; text-transform: uppercase; color: #C2185B; margin: 0 0 0.3rem 0;'>✦ Sephora</p>
        <p style='font-family: Cormorant Garamond, serif; font-size: 1.6rem; font-style: italic; font-weight: 300; color: #1A1A1A; margin: 0 0 1.5rem 0;'>Intelligence</p>
        <div style='height:1px; background: linear-gradient(90deg, #F4A0B8, transparent); margin-bottom: 1.5rem;'></div>
    </div>
    """, unsafe_allow_html=True)
    st.page_link("app.py", label="Inicio")
    st.page_link("pages/1_vista_ejecutiva.py", label="Vista Ejecutiva")
    st.page_link("pages/2_vista_tecnica.py", label="Vista Técnica")
    st.page_link("pages/3_vista_operativa.py", label="Vista Operativa")

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

st.markdown("<p class='eyebrow'>✦ Machine Learning</p>", unsafe_allow_html=True)
st.markdown("<h1 style='font-family: Cormorant Garamond, serif; font-weight: 400; font-size: 3rem; margin: 0 0 0.5rem 0;'>Vista Técnica</h1>", unsafe_allow_html=True)
st.markdown("<p style='color:#888; font-size:0.92rem; font-weight: 300;'>Métricas del modelo de Machine Learning y resultados del análisis no supervisado.</p>", unsafe_allow_html=True)
st.markdown("<div class='divider-floral'><span>✦</span></div>", unsafe_allow_html=True)

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

st.markdown("<p class='section-title'>Modelo Supervisado</p>", unsafe_allow_html=True)

col1, col2 = st.columns([2, 3], gap="large")

with col1:
    st.markdown(f"<div class='param-box'><div class='param-label'>Modelo</div><div class='param-value' style='font-size:1.15rem;'>{metrics.get('model_name', 'N/A')}</div></div>", unsafe_allow_html=True)

    params = {
        "N° Estimadores":  metrics.get("n_estimators", "N/A"),
        "Profundidad máx": metrics.get("max_depth", "N/A"),
        "Learning Rate":   metrics.get("learning_rate", "N/A"),
        "Random State":    metrics.get("random_state", "N/A"),
    }
    for label, val in params.items():
        st.markdown(f"<div class='param-box'><div class='param-label'>{label}</div><div class='param-value'>{val}</div></div>", unsafe_allow_html=True)

with col2:
    st.markdown("<p style='font-family: Cormorant Garamond, serif; font-size: 1.2rem; color:#1A1A1A; margin-bottom: 0.8rem;'>¿Qué predice este modelo?</p>", unsafe_allow_html=True)
    st.markdown("""
    <p style='color:#555; font-size:0.9rem; font-weight:300; line-height:1.8;'>
    El <b style='color:#1A1A1A; font-weight:500;'>Gradient Boosting Classifier</b> predice si un usuario recomendará o no un producto,
    basándose en características como el precio, categoría, marca, y patrones de comportamiento
    de otros compradores.
    </p>
    <p style='color:#555; font-size:0.9rem; font-weight:300; line-height:1.8;'>
    El modelo fue entrenado en el EP2 del proyecto usando técnicas de validación cruzada
    y optimización de hiperparámetros (Grid Search + Random Search).
    </p>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class='note-box'>
        Los parámetros mostrados corresponden al mejor modelo encontrado durante la optimización,
        con <code>n_estimators={metrics.get('n_estimators','—')}</code>,
        <code>max_depth={metrics.get('max_depth','—')}</code> y
        <code>learning_rate={metrics.get('learning_rate','—')}</code>.
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div class='divider-floral'><span>✦</span></div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Clustering K-Means
# ---------------------------------------------------------------------------

st.markdown("<p class='section-title'>Análisis de Clustering K-Means</p>", unsafe_allow_html=True)

if not df_clusters.empty:
    m1, m2, m3 = st.columns(3)

    best = df_clusters[df_clusters["is_best"] == True].iloc[0]
    m1.metric("K óptimo", int(best["k"]))
    m2.metric("Silhouette Score", f"{best['silhouette']:.4f}")
    m3.metric("Inercia final", f"{best['inertia']:,.0f}")

    st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)

    col3, col4 = st.columns(2, gap="large")

    with col3:
        with st.container(border=True):
            st.markdown("<p class='section-title' style='font-size:1.25rem;'>Silhouette Score por K</p>", unsafe_allow_html=True)

            best_row = df_clusters[df_clusters["is_best"] == True]

            fig_sil = go.Figure()
            fig_sil.add_trace(go.Scatter(
                x=df_clusters["k"], y=df_clusters["silhouette"],
                mode="lines+markers",
                line=dict(color="#9B7FB5", width=2),
                marker=dict(color="#7B5EA7", size=7),
                name="Silhouette",
            ))
            fig_sil.add_trace(go.Scatter(
                x=best_row["k"], y=best_row["silhouette"],
                mode="markers",
                marker=dict(size=15, color="#D4A55A", symbol="star"),
                name="K óptimo",
            ))
            fig_sil.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font_family="Jost", font_color="#555",
                margin=dict(l=0, r=0, t=10, b=10), height=280,
                xaxis=dict(title="Número de clusters (k)", gridcolor="#F0EAF7"),
                yaxis=dict(title="Silhouette Score", gridcolor="#F0EAF7"),
                showlegend=True,
                legend=dict(font=dict(size=10)),
            )
            st.plotly_chart(fig_sil, use_container_width=True)

    with col4:
        with st.container(border=True):
            st.markdown("<p class='section-title' style='font-size:1.25rem;'>Inercia por K (Método del Codo)</p>", unsafe_allow_html=True)

            fig_ine = go.Figure()
            fig_ine.add_trace(go.Scatter(
                x=df_clusters["k"], y=df_clusters["inertia"],
                mode="lines+markers",
                line=dict(color="#B0A0D0", width=2),
                marker=dict(color="#7B5EA7", size=7),
                name="Inercia",
            ))
            fig_ine.add_trace(go.Scatter(
                x=best_row["k"], y=best_row["inertia"],
                mode="markers",
                marker=dict(size=15, color="#D4A55A", symbol="star"),
                name="K óptimo",
            ))
            fig_ine.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font_family="Jost", font_color="#555",
                margin=dict(l=0, r=0, t=10, b=10), height=280,
                xaxis=dict(title="Número de clusters (k)", gridcolor="#F0EAF7"),
                yaxis=dict(title="Inercia", gridcolor="#F0EAF7"),
                showlegend=True,
                legend=dict(font=dict(size=10)),
            )
            st.plotly_chart(fig_ine, use_container_width=True)

    st.markdown("<div class='divider-floral'><span>✦</span></div>", unsafe_allow_html=True)

    st.markdown("<p class='section-title'>Resumen completo de evaluación de K</p>", unsafe_allow_html=True)
    df_display = df_clusters.copy()
    df_display["is_best"] = df_display["is_best"].apply(lambda x: "★ Óptimo" if x else "")
    df_display["silhouette"] = df_display["silhouette"].round(4)
    df_display["inertia"] = df_display["inertia"].apply(lambda x: f"{x:,.0f}")
    df_display.columns = ["K", "Inercia", "Silhouette", "Seleccionado"]
    st.dataframe(df_display, use_container_width=True, hide_index=True)

st.markdown("<div class='divider-floral'><span>✦</span></div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Métricas de PCA
# ---------------------------------------------------------------------------

st.markdown("<p class='section-title'>Reducción de Dimensionalidad (PCA)</p>", unsafe_allow_html=True)

pca_2d = metrics.get("pca_variance_2d", 0) or 0

p1, p2, p3 = st.columns(3)
p1.metric("Varianza explicada (2D)", f"{pca_2d*100:.1f}%")
p2.metric("Componentes p/ 90% var.", "33")
p3.metric("Componentes p/ 95% var.", "55")

st.markdown(f"""
<div class='note-box' style='margin-top:1rem;'>
Los primeros 2 componentes principales explican el <b>{pca_2d*100:.1f}%</b> de la varianza total del dataset.
Aunque es una representación comprimida, es suficiente para visualizar la separación entre clusters.
Para capturar el <b>90%</b> de la información se necesitan <b>33 componentes</b>, lo que refleja la
alta dimensionalidad del espacio de características de productos cosméticos.
</div>
""", unsafe_allow_html=True)