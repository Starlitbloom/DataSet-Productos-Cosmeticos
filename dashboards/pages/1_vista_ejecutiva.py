"""Vista Ejecutiva — KPIs de negocio para toma de decisiones."""

import requests
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Vista Ejecutiva · Sephora", page_icon="💼", layout="wide")

# ---------------------------------------------------------------------------
# Estilos — elegante, con vida, glamoroso
# ---------------------------------------------------------------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,300;0,400;0,600;1,300;1,400&family=Jost:wght@300;400;500&display=swap');

html, body, [class*="css"], .stApp {
    font-family: 'Jost', sans-serif;
    background: linear-gradient(160deg, #FDF6F3 0%, #FBF1F5 100%) !important;
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

/* Métricas — cards con borde dorado sutil */
[data-testid="stMetric"] {
    background: linear-gradient(145deg, #FFFFFF 0%, #FFF9FB 100%) !important;
    border: 1px solid #F5DCE6 !important;
    border-radius: 16px !important;
    padding: 1.3rem 1.4rem !important;
    box-shadow: 0 8px 24px rgba(194, 24, 91, 0.10), 0 1px 3px rgba(212, 160, 90, 0.08) !important;
}
[data-testid="stMetricLabel"] {
    font-family: 'Jost', sans-serif !important;
    color: #B5708A !important;
    font-size: 0.65rem !important;
    text-transform: uppercase;
    letter-spacing: 1.5px;
}
[data-testid="stMetricValue"] {
    font-family: 'Cormorant Garamond', serif !important;
    color: #1A1A1A !important;
    font-size: 2.15rem !important;
    font-weight: 500 !important;
}

/* Contenedores reales de Streamlit con borde -> usados como "cards" */
[data-testid="stVerticalBlockBorderWrapper"] {
    background: linear-gradient(145deg, #FFFFFF 0%, #FFFAFB 100%) !important;
    border-radius: 18px !important;
    border: 1px solid #F5DCE6 !important;
    box-shadow: 0 10px 30px rgba(194, 24, 91, 0.09), 0 1px 4px rgba(212, 160, 90, 0.08) !important;
    padding: 0.3rem !important;
}

.eyebrow {
    font-size: 0.68rem; letter-spacing: 4px; text-transform: uppercase;
    color: #C2185B; margin-bottom: 0.8rem;
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
    background: linear-gradient(90deg, #C2185B, #D4A55A);
    margin-top: 0.5rem;
    margin-bottom: 1rem;
}
.thin-line { height: 1px; background: linear-gradient(90deg, #EDE0DC, transparent); margin: 2.5rem 0; border: none; }
.divider-floral {
    display: flex; align-items: center; gap: 12px; margin: 2.5rem 0;
}
.divider-floral::before, .divider-floral::after {
    content: ''; flex: 1; height: 1px; background: linear-gradient(90deg, transparent, #E0A0B8);
}
.divider-floral::after { background: linear-gradient(90deg, #E0A0B8, transparent); }
.divider-floral span { color: #C2185B; font-size: 1rem; }

.insight-box {
    background: linear-gradient(145deg, #FFFFFF 0%, #FFF9FB 100%);
    border-radius: 14px;
    border: 1px solid #F5DCE6;
    border-left: 4px solid #C2185B;
    padding: 1.1rem 1.5rem;
    margin: 0.7rem 0;
    color: #555;
    font-size: 0.89rem;
    font-weight: 300;
    line-height: 1.65;
    box-shadow: 0 6px 20px rgba(194, 24, 91, 0.08);
}
.insight-box b { color: #A01650; font-weight: 600; }

[data-testid="stDataFrame"] {
    border-radius: 16px;
    overflow: hidden;
    box-shadow: 0 8px 24px rgba(194, 24, 91, 0.08);
    border: 1px solid #F5DCE6;
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

import os

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")

@st.cache_data(ttl=300)
def get_exchange_rate():
    r = requests.get(f"{API_BASE}/exchange-rate", timeout=5)
    return r.json() if r.status_code == 200 else {"rate": 950.0, "date": "N/A", "source": "fallback"}

@st.cache_data(ttl=300)
def get_products(limit=500):
    r = requests.get(f"{API_BASE}/products", params={"limit": limit}, timeout=10)
    return pd.DataFrame(r.json()) if r.status_code == 200 else pd.DataFrame()

@st.cache_data(ttl=300)
def get_reviews(limit=5000):
    r = requests.get(f"{API_BASE}/reviews", params={"limit": limit}, timeout=10)
    return pd.DataFrame(r.json()) if r.status_code == 200 else pd.DataFrame()

# Paleta con más variación tonal — rosa, burgundy, dorado, malva
PALETTE = ["#C2185B", "#D4A55A", "#9B6B8E", "#E0789A", "#B08850",
           "#8E5A7A", "#D49070", "#A04868", "#C9A876", "#9B5570"]

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.markdown("<p class='eyebrow'>✦ Business Intelligence</p>", unsafe_allow_html=True)
st.markdown("<h1 style='font-family: Cormorant Garamond, serif; font-weight: 400; font-size: 3rem; margin: 0 0 0.5rem 0;'>Vista Ejecutiva</h1>", unsafe_allow_html=True)
st.markdown("<p style='color:#888; font-size:0.92rem; font-weight: 300;'>Indicadores clave de negocio para la toma de decisiones estratégicas.</p>", unsafe_allow_html=True)
st.markdown("<div class='divider-floral'><span>✦</span></div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Cargar datos
# ---------------------------------------------------------------------------

with st.spinner("Cargando datos..."):
    fx = get_exchange_rate()
    df_products = get_products(500)
    df_reviews = get_reviews(5000)

if df_products.empty:
    st.error("No se pudo conectar a la API. Asegúrate de que uvicorn está corriendo.")
    st.stop()

rate = fx.get("rate", 950.0)

# ---------------------------------------------------------------------------
# KPIs principales
# ---------------------------------------------------------------------------

total_productos = len(df_products)
avg_precio_usd  = df_products["price_usd"].mean() if "price_usd" in df_products else 0
avg_precio_clp  = avg_precio_usd * rate
avg_rating      = df_products["rating"].mean() if "rating" in df_products else 0
tasa_rec        = df_reviews["is_recommended"].mean() * 100 if not df_reviews.empty else 0

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Catálogo", f"{total_productos:,}")
k2.metric("Precio prom. USD", f"${avg_precio_usd:.0f}")
k3.metric("Precio prom. CLP", f"${avg_precio_clp:,.0f}")
k4.metric("Rating", f"{avg_rating:.2f} ★")
k5.metric("Recomendación", f"{tasa_rec:.1f}%")

st.markdown(f"<p style='color:#AAA; font-size:0.75rem; margin-top:1.2rem;'>Tipo de cambio: 1 USD = {rate:,.2f} CLP · Fuente: {fx.get('source','N/A')} · {fx.get('date','')}</p>", unsafe_allow_html=True)
st.markdown("<div class='divider-floral'><span>✦</span></div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Top marcas + Categorías — usando st.container(border=True) real
# ---------------------------------------------------------------------------

col1, col2 = st.columns([3, 2], gap="large")

with col1:
    with st.container(border=True):
        st.markdown("<p class='section-title'>Top 10 Marcas más Amadas</p>", unsafe_allow_html=True)

        top_brands = (
            df_products.groupby("brand_name")
            .agg(avg_loves=("loves_count", "mean"), productos=("product_id", "count"))
            .sort_values("avg_loves", ascending=False)
            .head(10)
            .reset_index()
        )

        # Gradiente de color por posición — más intenso arriba, más suave abajo
        n = len(top_brands)
        bar_colors = [f"rgba(194,24,91,{1 - i*0.07:.2f})" for i in range(n)]

        fig = go.Figure(go.Bar(
            x=top_brands["avg_loves"],
            y=top_brands["brand_name"],
            orientation="h",
            marker=dict(
                color=top_brands["avg_loves"],
                colorscale=[[0, "#E8A0BC"], [0.5, "#C2185B"], [1, "#7A0E38"]],
                line=dict(width=0),
            ),
            text=top_brands["avg_loves"].apply(lambda x: f"{x:,.0f}"),
            textposition="outside",
            textfont=dict(size=11, family="Jost"),
        ))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_family="Jost", font_color="#555",
            margin=dict(l=0, r=30, t=10, b=10), height=360,
            yaxis=dict(autorange="reversed"),
            xaxis=dict(gridcolor="#F5EBEE", showgrid=True),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

with col2:
    with st.container(border=True):
        st.markdown("<p class='section-title'>Por Categoría</p>", unsafe_allow_html=True)

        cat_counts = df_products["primary_category"].value_counts().reset_index()
        cat_counts.columns = ["Categoría", "Productos"]

        fig2 = px.pie(
            cat_counts, names="Categoría", values="Productos",
            color_discrete_sequence=PALETTE, hole=0.55,
        )
        fig2.update_traces(
            textposition="inside", textinfo="percent", textfont_size=10,
            marker=dict(line=dict(color="white", width=2)),
        )
        fig2.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", showlegend=True,
            font_family="Jost", font_color="#555",
            legend=dict(font=dict(size=10), orientation="h", yanchor="bottom", y=-0.3),
            margin=dict(l=0, r=0, t=10, b=10), height=360,
        )
        st.plotly_chart(fig2, use_container_width=True)

st.markdown("<div class='divider-floral'><span>✦</span></div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Distribución de precios
# ---------------------------------------------------------------------------

with st.container(border=True):
    st.markdown("<p class='section-title'>Distribución de Precios en CLP</p>", unsafe_allow_html=True)

    df_products["price_clp_calc"] = df_products["price_usd"] * rate

    fig3 = px.histogram(
        df_products.dropna(subset=["price_clp_calc"]), x="price_clp_calc", nbins=40,
        labels={"price_clp_calc": "Precio (CLP)", "count": "Cantidad de productos"},
    )
    fig3.update_traces(
        marker=dict(
            color=df_products.dropna(subset=["price_clp_calc"])["price_clp_calc"],
            colorscale=[[0, "#D4A55A"], [0.5, "#C2185B"], [1, "#7A0E38"]],
            line_width=0,
        )
    )
    fig3.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_family="Jost", font_color="#555",
        bargap=0.08, margin=dict(l=0, r=0, t=10, b=10), height=260,
        xaxis=dict(tickformat="$,.0f", gridcolor="#F5EBEE"),
        yaxis=dict(gridcolor="#F5EBEE"),
    )
    st.plotly_chart(fig3, use_container_width=True)

precio_cat = (
    df_products.groupby("primary_category")["price_clp_calc"]
    .agg(["min", "max", "mean"]).round(0)
    .sort_values("mean", ascending=False).reset_index()
)
precio_cat.columns = ["Categoría", "Precio mín (CLP)", "Precio máx (CLP)", "Precio prom (CLP)"]
precio_cat[["Precio mín (CLP)", "Precio máx (CLP)", "Precio prom (CLP)"]] = (
    precio_cat[["Precio mín (CLP)", "Precio máx (CLP)", "Precio prom (CLP)"]].map(lambda x: f"${x:,.0f}")
)
st.dataframe(precio_cat, use_container_width=True, hide_index=True)

st.markdown("<div class='divider-floral'><span>✦</span></div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Insights
# ---------------------------------------------------------------------------

st.markdown("<p class='section-title'>Insights</p>", unsafe_allow_html=True)

marca_top  = top_brands.iloc[0]["brand_name"]
loves_top  = top_brands.iloc[0]["avg_loves"]
cat_top    = cat_counts.iloc[0]["Categoría"]
n_cat_top  = cat_counts.iloc[0]["Productos"]
precio_med = df_products["price_clp_calc"].median()

st.markdown(f"<div class='insight-box'><b>{marca_top}</b> lidera en popularidad con un promedio de <b>{loves_top:,.0f} loves</b> por producto.</div>", unsafe_allow_html=True)
st.markdown(f"<div class='insight-box'><b>{cat_top}</b> es la categoría con más productos en el catálogo ({n_cat_top:,} productos).</div>", unsafe_allow_html=True)
st.markdown(f"<div class='insight-box'>El precio mediano del catálogo es <b>${precio_med:,.0f} CLP</b> (~${precio_med/rate:.0f} USD).</div>", unsafe_allow_html=True)
st.markdown(f"<div class='insight-box'>El <b>{tasa_rec:.1f}%</b> de los compradores recomienda los productos que adquirió.</div>", unsafe_allow_html=True)