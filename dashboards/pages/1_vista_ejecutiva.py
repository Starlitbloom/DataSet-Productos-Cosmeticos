"""Vista Ejecutiva — KPIs de negocio para toma de decisiones."""

import requests
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Vista Ejecutiva · Sephora", page_icon="💼", layout="wide")

# ---------------------------------------------------------------------------
# Estilos
# ---------------------------------------------------------------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,700;1,400&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #FFF8FB;
}
h1, h2, h3 { font-family: 'Playfair Display', serif; }

[data-testid="stSidebar"] {
    background: linear-gradient(160deg, #FFE4F0 0%, #F3E8FF 100%);
    border-right: 1px solid #F4C2D8;
}

[data-testid="stMetric"] {
    background: white;
    border-radius: 16px;
    padding: 1rem;
    border: 1px solid #FFD6E7;
    box-shadow: 0 2px 12px rgba(255, 107, 157, 0.08);
}
[data-testid="stMetricLabel"] { color: #9E3A6B; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.8px; }
[data-testid="stMetricValue"] { font-family: 'Playfair Display', serif; color: #C2185B; }

.section-title {
    font-family: 'Playfair Display', serif;
    color: #C2185B;
    font-size: 1.4rem;
    margin-bottom: 0.5rem;
}
.divider {
    height: 2px;
    background: linear-gradient(90deg, transparent, #FF6B9D, #F4C2C2, transparent);
    margin: 1.5rem 0;
    border: none;
}
.insight-box {
    background: linear-gradient(135deg, #FFF0F5, #FFF8E1);
    border-left: 4px solid #FF6B9D;
    border-radius: 0 12px 12px 0;
    padding: 0.8rem 1.2rem;
    margin: 0.5rem 0;
    color: #555;
    font-size: 0.9rem;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

API_BASE = "http://localhost:8000"

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

# ---------------------------------------------------------------------------
# Colores
# ---------------------------------------------------------------------------

PALETTE = ["#FF6B9D", "#F4A7C3", "#C2185B", "#F4C2C2", "#FFB3C6",
           "#FF85B3", "#E91E8C", "#FF4081", "#F48FB1", "#FCE4EC"]

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.markdown("## 💼 Vista Ejecutiva")
st.markdown("<p style='color:#9E3A6B; font-size:1rem'>Indicadores clave de negocio para la toma de decisiones estratégicas.</p>", unsafe_allow_html=True)
st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Cargar datos
# ---------------------------------------------------------------------------

with st.spinner("Cargando datos..."):
    fx = get_exchange_rate()
    df_products = get_products(500)
    df_reviews = get_reviews(1000)

if df_products.empty:
    st.error("No se pudo conectar a la API. Asegúrate de que uvicorn está corriendo.")
    st.stop()

rate = fx.get("rate", 950.0)

# ---------------------------------------------------------------------------
# KPIs principales
# ---------------------------------------------------------------------------

st.markdown("<div class='section-title'>✨ Indicadores Clave</div>", unsafe_allow_html=True)

total_productos = len(df_products)
avg_precio_usd  = df_products["price_usd"].mean() if "price_usd" in df_products else 0
avg_precio_clp  = avg_precio_usd * rate
avg_rating      = df_products["rating"].mean() if "rating" in df_products else 0
tasa_rec        = df_reviews["is_recommended"].mean() * 100 if not df_reviews.empty else 0

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Productos en catálogo", f"{total_productos:,}")
k2.metric("Precio promedio (USD)", f"${avg_precio_usd:.0f}")
k3.metric("Precio promedio (CLP)", f"${avg_precio_clp:,.0f}")
k4.metric("Rating promedio", f"⭐ {avg_rating:.2f}")
k5.metric("Tasa de recomendación", f"{tasa_rec:.1f}%")

st.markdown(f"<small style='color:#9E3A6B'>Tipo de cambio: 1 USD = {rate:,.2f} CLP · Fuente: {fx.get('source','N/A')} · {fx.get('date','')}</small>", unsafe_allow_html=True)
st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Top marcas por popularidad
# ---------------------------------------------------------------------------

col1, col2 = st.columns([3, 2])

with col1:
    st.markdown("<div class='section-title'>💖 Top 10 Marcas más Amadas</div>", unsafe_allow_html=True)

    top_brands = (
        df_products.groupby("brand_name")
        .agg(avg_loves=("loves_count", "mean"), productos=("product_id", "count"))
        .sort_values("avg_loves", ascending=False)
        .head(10)
        .reset_index()
    )

    fig = px.bar(
        top_brands,
        x="avg_loves",
        y="brand_name",
        orientation="h",
        color="avg_loves",
        color_continuous_scale=["#FFB3C6", "#FF6B9D", "#C2185B"],
        labels={"avg_loves": "Promedio de 'Loves'", "brand_name": ""},
        text=top_brands["avg_loves"].apply(lambda x: f"{x:,.0f}"),
    )
    fig.update_traces(textposition="outside", textfont_size=11)
    fig.update_layout(
        paper_bgcolor="#FFF8FB", plot_bgcolor="#FFF8FB",
        coloraxis_showscale=False,
        margin=dict(l=0, r=20, t=10, b=10),
        height=380,
        yaxis=dict(autorange="reversed"),
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown("<div class='section-title'>🗂️ Productos por Categoría</div>", unsafe_allow_html=True)

    cat_counts = df_products["primary_category"].value_counts().reset_index()
    cat_counts.columns = ["Categoría", "Productos"]

    fig2 = px.pie(
        cat_counts,
        names="Categoría",
        values="Productos",
        color_discrete_sequence=PALETTE,
        hole=0.45,
    )
    fig2.update_traces(textposition="inside", textinfo="percent+label", textfont_size=11)
    fig2.update_layout(
        paper_bgcolor="#FFF8FB",
        showlegend=False,
        margin=dict(l=0, r=0, t=10, b=10),
        height=380,
    )
    st.plotly_chart(fig2, use_container_width=True)

st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Distribución de precios en CLP
# ---------------------------------------------------------------------------

st.markdown("<div class='section-title'>💸 Distribución de Precios en CLP</div>", unsafe_allow_html=True)

df_products["price_clp_calc"] = df_products["price_usd"] * rate

fig3 = px.histogram(
    df_products.dropna(subset=["price_clp_calc"]),
    x="price_clp_calc",
    nbins=40,
    color_discrete_sequence=["#FF6B9D"],
    labels={"price_clp_calc": "Precio (CLP)", "count": "Cantidad de productos"},
)
fig3.update_layout(
    paper_bgcolor="#FFF8FB", plot_bgcolor="#FFF8FB",
    bargap=0.05,
    margin=dict(l=0, r=0, t=10, b=10),
    height=280,
    xaxis=dict(tickformat="$,.0f"),
)
st.plotly_chart(fig3, use_container_width=True)

# Rango de precios por categoría
precio_cat = (
    df_products.groupby("primary_category")["price_clp_calc"]
    .agg(["min", "max", "mean"])
    .round(0)
    .sort_values("mean", ascending=False)
    .reset_index()
)
precio_cat.columns = ["Categoría", "Precio mín (CLP)", "Precio máx (CLP)", "Precio prom (CLP)"]
precio_cat[["Precio mín (CLP)", "Precio máx (CLP)", "Precio prom (CLP)"]] = (
    precio_cat[["Precio mín (CLP)", "Precio máx (CLP)", "Precio prom (CLP)"]].map(lambda x: f"${x:,.0f}")
)
st.dataframe(precio_cat, use_container_width=True, hide_index=True)

st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Insights automáticos
# ---------------------------------------------------------------------------

st.markdown("<div class='section-title'>💡 Insights</div>", unsafe_allow_html=True)

marca_top  = top_brands.iloc[0]["brand_name"]
loves_top  = top_brands.iloc[0]["avg_loves"]
cat_top    = cat_counts.iloc[0]["Categoría"]
n_cat_top  = cat_counts.iloc[0]["Productos"]
precio_med = df_products["price_clp_calc"].median()

st.markdown(f"<div class='insight-box'>🏆 <b>{marca_top}</b> lidera en popularidad con un promedio de <b>{loves_top:,.0f} loves</b> por producto.</div>", unsafe_allow_html=True)
st.markdown(f"<div class='insight-box'>📦 <b>{cat_top}</b> es la categoría con más productos en el catálogo ({n_cat_top:,} productos).</div>", unsafe_allow_html=True)
st.markdown(f"<div class='insight-box'>💰 El precio mediano del catálogo es <b>${precio_med:,.0f} CLP</b> (~ ${precio_med/rate:.0f} USD).</div>", unsafe_allow_html=True)
st.markdown(f"<div class='insight-box'>⭐ El <b>{tasa_rec:.1f}%</b> de los compradores recomienda los productos que adquirió.</div>", unsafe_allow_html=True)