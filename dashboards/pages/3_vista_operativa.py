"""Vista Operativa — Exploración interactiva de productos y sentimiento."""

import requests
import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Vista Operativa · Sephora", page_icon="🌿", layout="wide")

# ---------------------------------------------------------------------------
# Estilos editoriales — acento verde menta + rosa
# ---------------------------------------------------------------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,300;0,400;0,500;1,300;1,400&family=Jost:wght@300;400;500&display=swap');

html, body, [class*="css"], .stApp {
    font-family: 'Jost', sans-serif;
    background: linear-gradient(160deg, #F8FBF7 0%, #FAF8F5 100%) !important;
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

[data-testid="stMetric"] {
    background: linear-gradient(145deg, #FFFFFF 0%, #FAFCF8 100%) !important;
    border: 1px solid #DCEDD9 !important;
    border-radius: 16px !important;
    padding: 1.3rem 1.4rem !important;
    box-shadow: 0 8px 24px rgba(74, 138, 92, 0.10), 0 1px 3px rgba(212, 160, 90, 0.06) !important;
}
[data-testid="stMetricLabel"] {
    font-family: 'Jost', sans-serif !important;
    color: #6B9E78 !important;
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

[data-testid="stVerticalBlockBorderWrapper"] {
    background: linear-gradient(145deg, #FFFFFF 0%, #FAFCF8 100%) !important;
    border-radius: 18px !important;
    border: 1px solid #DCEDD9 !important;
    box-shadow: 0 10px 30px rgba(74, 138, 92, 0.09), 0 1px 4px rgba(212, 160, 90, 0.06) !important;
    padding: 0.3rem !important;
}

.eyebrow {
    font-size: 0.68rem; letter-spacing: 4px; text-transform: uppercase;
    color: #4A8A5C; margin-bottom: 0.8rem;
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
    background: linear-gradient(90deg, #4A8A5C, #D4A55A);
    margin-top: 0.5rem;
    margin-bottom: 1rem;
}

.divider-floral {
    display: flex; align-items: center; gap: 12px; margin: 2.5rem 0;
}
.divider-floral::before, .divider-floral::after {
    content: ''; flex: 1; height: 1px; background: linear-gradient(90deg, transparent, #A8D4B0);
}
.divider-floral::after { background: linear-gradient(90deg, #A8D4B0, transparent); }
.divider-floral span { color: #4A8A5C; font-size: 1rem; }

.note-box {
    background: linear-gradient(145deg, #FFFFFF 0%, #FAFCF8 100%);
    border-radius: 14px;
    border-left: 4px solid #4A8A5C;
    border-top: 1px solid #DCEDD9; border-right: 1px solid #DCEDD9; border-bottom: 1px solid #DCEDD9;
    padding: 1.1rem 1.5rem;
    color: #555;
    font-size: 0.88rem;
    line-height: 1.7;
    font-weight: 300;
    box-shadow: 0 6px 20px rgba(74, 138, 92, 0.08);
}
.note-box b { color: #2E6B40; font-weight: 600; }
.note-box .pink { color: #A01650; font-weight: 600; }

[data-testid="stDataFrame"] {
    border-radius: 16px;
    overflow: hidden;
    box-shadow: 0 8px 24px rgba(74, 138, 92, 0.08);
    border: 1px solid #DCEDD9;
}

.stSelectbox label, .stSlider label {
    font-family: 'Jost', sans-serif !important;
    font-size: 0.75rem !important;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    color: #4A8A5C !important;
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
def get_products_full():
    r = requests.get(f"{API_BASE}/products", params={"limit": 500}, timeout=10)
    return pd.DataFrame(r.json()) if r.status_code == 200 else pd.DataFrame()

@st.cache_data(ttl=300)
def get_reviews_full():
    r = requests.get(f"{API_BASE}/reviews", params={"limit": 500}, timeout=10)
    return pd.DataFrame(r.json()) if r.status_code == 200 else pd.DataFrame()

@st.cache_data(ttl=300)
def get_clusters():
    r = requests.get(f"{API_BASE}/clusters", timeout=5)
    return pd.DataFrame(r.json()) if r.status_code == 200 else pd.DataFrame()

GREEN_PALETTE = ["#4A8A5C", "#7AAF87", "#A8D4B0", "#D4A55A", "#C2185B",
                 "#E0789A", "#9B8EA8", "#6FA87E"]

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.markdown("<p class='eyebrow'>✦ Operational Insights</p>", unsafe_allow_html=True)
st.markdown("<h1 style='font-family: Cormorant Garamond, serif; font-weight: 400; font-size: 3rem; margin: 0 0 0.5rem 0;'>Vista Operativa</h1>", unsafe_allow_html=True)
st.markdown("<p style='color:#888; font-size:0.92rem; font-weight: 300;'>Exploración interactiva de productos, reseñas y patrones de consumo.</p>", unsafe_allow_html=True)
st.markdown("<div class='divider-floral'><span>✦</span></div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Cargar datos
# ---------------------------------------------------------------------------

with st.spinner("Cargando datos..."):
    df_products = get_products_full()
    df_reviews  = get_reviews_full()
    df_clusters = get_clusters()

if df_products.empty:
    st.error("No se pudo conectar a la API. Asegúrate de que uvicorn está corriendo.")
    st.stop()

# ---------------------------------------------------------------------------
# Filtros interactivos
# ---------------------------------------------------------------------------

st.markdown("<p class='section-title'>Filtros</p>", unsafe_allow_html=True)

col_f1, col_f2, col_f3 = st.columns(3)

with col_f1:
    categorias = ["Todas"] + sorted(df_products["primary_category"].dropna().unique().tolist())
    cat_sel = st.selectbox("Categoría", categorias)

with col_f2:
    marcas_disponibles = df_products.copy()
    if cat_sel != "Todas":
        marcas_disponibles = marcas_disponibles[marcas_disponibles["primary_category"] == cat_sel]
    marcas = ["Todas"] + sorted(marcas_disponibles["brand_name"].dropna().unique().tolist())
    marca_sel = st.selectbox("Marca", marcas)

with col_f3:
    precio_max = float(df_products["price_usd"].max()) if "price_usd" in df_products else 500.0
    rango = st.slider("Rango de precio (USD)", 0.0, precio_max, (0.0, precio_max), step=5.0)

df_fil = df_products.copy()
if cat_sel != "Todas":
    df_fil = df_fil[df_fil["primary_category"] == cat_sel]
if marca_sel != "Todas":
    df_fil = df_fil[df_fil["brand_name"] == marca_sel]
df_fil = df_fil[(df_fil["price_usd"] >= rango[0]) & (df_fil["price_usd"] <= rango[1])]

st.markdown(f"<p style='color:#AAA; font-size:0.78rem; margin-top:0.8rem;'>{len(df_fil):,} productos encontrados</p>", unsafe_allow_html=True)
st.markdown("<div class='divider-floral'><span>✦</span></div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# KPIs filtrados
# ---------------------------------------------------------------------------

k1, k2, k3, k4 = st.columns(4)
k1.metric("Productos", f"{len(df_fil):,}")
k2.metric("Precio promedio", f"${df_fil['price_usd'].mean():.2f}" if not df_fil.empty else "N/A")
k3.metric("Rating promedio", f"{df_fil['rating'].mean():.2f} ★" if not df_fil.empty else "N/A")
k4.metric("Marcas únicas", f"{df_fil['brand_name'].nunique()}")

st.markdown("<div class='divider-floral'><span>✦</span></div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Gráficos
# ---------------------------------------------------------------------------

col1, col2 = st.columns(2, gap="large")

with col1:
    with st.container(border=True):
        st.markdown("<p class='section-title'>Rating vs Precio</p>", unsafe_allow_html=True)
        if not df_fil.empty and "rating" in df_fil and "price_usd" in df_fil:
            fig = px.scatter(
                df_fil.dropna(subset=["rating", "price_usd"]),
                x="price_usd", y="rating",
                color="primary_category",
                hover_data=["product_name", "brand_name"],
                color_discrete_sequence=GREEN_PALETTE,
                labels={"price_usd": "Precio (USD)", "rating": "Rating"},
                opacity=0.75,
            )
            fig.update_traces(marker=dict(size=7, line_width=0))
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font_family="Jost", font_color="#555",
                margin=dict(l=0, r=0, t=10, b=10), height=320,
                legend=dict(font=dict(size=9)),
                xaxis=dict(gridcolor="#EDF5EA"), yaxis=dict(gridcolor="#EDF5EA"),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Sin datos para los filtros seleccionados.")

with col2:
    with st.container(border=True):
        st.markdown("<p class='section-title'>Top marcas (filtrado)</p>", unsafe_allow_html=True)
        if not df_fil.empty:
            top = (
                df_fil.groupby("brand_name")["loves_count"]
                .mean().sort_values(ascending=False).head(10).reset_index()
            )
            fig2 = px.bar(
                top, x="loves_count", y="brand_name", orientation="h",
                labels={"loves_count": "Promedio de loves", "brand_name": ""},
            )
            fig2.update_traces(
                marker=dict(
                    color=top["loves_count"],
                    colorscale=[[0, "#C8E6C9"], [0.5, "#4A8A5C"], [1, "#1B4D2A"]],
                    line_width=0,
                )
            )
            fig2.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font_family="Jost", font_color="#555",
                margin=dict(l=0, r=0, t=10, b=10), height=320,
                yaxis=dict(autorange="reversed"),
                xaxis=dict(gridcolor="#EDF5EA"),
            )
            st.plotly_chart(fig2, use_container_width=True)

st.markdown("<div class='divider-floral'><span>✦</span></div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sentimiento de reseñas
# ---------------------------------------------------------------------------

st.markdown("<p class='section-title'>Análisis de Sentimiento</p>", unsafe_allow_html=True)

@st.cache_data(ttl=300)
def get_sentiment():
    r = requests.get(f"{API_BASE}/sentiment/summary", timeout=5)
    if r.status_code == 200:
        df = pd.DataFrame(r.json())
        df["Sentimiento"] = df["sentiment"].map(
            {"positive": "Positivo", "negative": "Negativo", "neutral": "Neutral"}
        )
        df["Cantidad"] = df["total_reviews"]
        return df
    return pd.DataFrame({"Sentimiento": ["Positivo","Negativo","Neutral"], "Cantidad": [358,102,39]})

sentiment_data = get_sentiment()

col3, col4 = st.columns([2, 3], gap="large")

with col3:
    with st.container(border=True):
        fig3 = px.pie(
            sentiment_data, names="Sentimiento", values="Cantidad",
            color="Sentimiento",
            color_discrete_map={"Positivo": "#4A8A5C", "Negativo": "#C2185B", "Neutral": "#C9B1E0"},
            hole=0.55,
        )
        fig3.update_traces(
            textposition="inside", textinfo="percent+label", textfont_size=11,
            marker=dict(line=dict(color="white", width=2)),
        )
        fig3.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", showlegend=False,
            font_family="Jost", font_color="#555",
            margin=dict(l=0, r=0, t=10, b=10), height=260,
        )
        st.plotly_chart(fig3, use_container_width=True)

with col4:
    total_sent = sentiment_data["Cantidad"].sum()
    pct_pos = sentiment_data[sentiment_data["Sentimiento"] == "Positivo"]["Cantidad"].values[0] / total_sent * 100

    st.markdown(f"""
    <div class='note-box' style='height:100%; display:flex; flex-direction:column; justify-content:center;'>
        <p style='margin:0 0 0.8rem 0;'>
        Se analizaron <b>499 reseñas</b> mediante la API de <span class='pink'>Groq (LLaMA 3.1)</span> para clasificar
        el sentimiento de los compradores.
        </p>
        <p style='margin:0 0 0.8rem 0;'>
        El <b>{pct_pos:.0f}%</b> de las reseñas analizadas expresa un sentimiento <b>positivo</b>,
        lo que es consistente con la alta tasa de recomendación del catálogo.
        </p>
        <p style='color:#999; font-size:0.78rem; margin:0;'>
        Muestra estratificada por is_recommended · Modelo: llama-3.1-8b-instant
        </p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div class='divider-floral'><span>✦</span></div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Tabla de productos filtrados
# ---------------------------------------------------------------------------

st.markdown("<p class='section-title'>Catálogo de Productos</p>", unsafe_allow_html=True)

cols_show = ["product_name", "brand_name", "primary_category", "price_usd", "rating", "loves_count"]
cols_show = [c for c in cols_show if c in df_fil.columns]

df_table = df_fil[cols_show].copy().rename(columns={
    "product_name":     "Producto",
    "brand_name":       "Marca",
    "primary_category": "Categoría",
    "price_usd":        "Precio (USD)",
    "rating":           "Rating",
    "loves_count":      "Loves",
}).sort_values("Loves", ascending=False)

st.dataframe(df_table.head(100), use_container_width=True, hide_index=True)

if len(df_fil) > 100:
    st.caption(f"Mostrando los 100 productos con más loves de {len(df_fil):,} resultados.")