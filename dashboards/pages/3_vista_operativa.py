"""Vista Operativa — Exploración interactiva de productos y sentimiento."""

import requests
import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Vista Operativa · Sephora", page_icon="🌿", layout="wide")

# ---------------------------------------------------------------------------
# Estilos — rosa + verde menta
# ---------------------------------------------------------------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,700;1,400&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #F6FBF8;
}
h1, h2, h3 { font-family: 'Playfair Display', serif; }

[data-testid="stSidebar"] {
    background: linear-gradient(160deg, #E8F5E9 0%, #FFE4F0 100%);
    border-right: 1px solid #A5D6A7;
}

[data-testid="stMetric"] {
    background: white;
    border-radius: 16px;
    padding: 1rem;
    border: 1px solid #A5D6A7;
    box-shadow: 0 2px 12px rgba(46, 139, 110, 0.07);
}
[data-testid="stMetricLabel"] { color: #2E7D32; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.8px; }
[data-testid="stMetricValue"] { font-family: 'Playfair Display', serif; color: #1B5E20; }

.section-title {
    font-family: 'Playfair Display', serif;
    color: #2E8B6E;
    font-size: 1.4rem;
    margin-bottom: 0.5rem;
}
.divider {
    height: 2px;
    background: linear-gradient(90deg, transparent, #A5D6A7, #FF6B9D, transparent);
    margin: 1.5rem 0;
    border: none;
}
.filter-label {
    color: #2E7D32;
    font-size: 0.8rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.6px;
}
</style>
""", unsafe_allow_html=True)

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

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.markdown("## 🌿 Vista Operativa")
st.markdown("<p style='color:#2E8B6E; font-size:1rem'>Exploración interactiva de productos, reseñas y patrones de consumo.</p>", unsafe_allow_html=True)
st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

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

st.markdown("<div class='section-title'>🔍 Filtros</div>", unsafe_allow_html=True)

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

# Aplicar filtros
df_fil = df_products.copy()
if cat_sel != "Todas":
    df_fil = df_fil[df_fil["primary_category"] == cat_sel]
if marca_sel != "Todas":
    df_fil = df_fil[df_fil["brand_name"] == marca_sel]
df_fil = df_fil[(df_fil["price_usd"] >= rango[0]) & (df_fil["price_usd"] <= rango[1])]

st.markdown(f"<small style='color:#2E8B6E'>{len(df_fil):,} productos encontrados</small>", unsafe_allow_html=True)
st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# KPIs filtrados
# ---------------------------------------------------------------------------

k1, k2, k3, k4 = st.columns(4)
k1.metric("Productos", f"{len(df_fil):,}")
k2.metric("Precio promedio", f"${df_fil['price_usd'].mean():.2f}" if not df_fil.empty else "N/A")
k3.metric("Rating promedio", f"⭐ {df_fil['rating'].mean():.2f}" if not df_fil.empty else "N/A")
k4.metric("Marcas únicas", f"{df_fil['brand_name'].nunique()}")

st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Gráficos
# ---------------------------------------------------------------------------

col1, col2 = st.columns(2)

with col1:
    st.markdown("<div class='section-title'>💚 Rating vs Precio</div>", unsafe_allow_html=True)
    if not df_fil.empty and "rating" in df_fil and "price_usd" in df_fil:
        fig = px.scatter(
            df_fil.dropna(subset=["rating", "price_usd"]),
            x="price_usd",
            y="rating",
            color="primary_category",
            hover_data=["product_name", "brand_name"],
            color_discrete_sequence=["#66BB6A", "#FF6B9D", "#C9B1FF", "#F4A7C3",
                                      "#80CBC4", "#FFB74D", "#90CAF9", "#CE93D8"],
            labels={"price_usd": "Precio (USD)", "rating": "Rating"},
            opacity=0.7,
        )
        fig.update_layout(
            paper_bgcolor="#F6FBF8", plot_bgcolor="#F6FBF8",
            margin=dict(l=0, r=0, t=10, b=10),
            height=320,
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Sin datos para los filtros seleccionados.")

with col2:
    st.markdown("<div class='section-title'>🌸 Top Marcas (filtrado)</div>", unsafe_allow_html=True)
    if not df_fil.empty:
        top = (
            df_fil.groupby("brand_name")["loves_count"]
            .mean()
            .sort_values(ascending=False)
            .head(10)
            .reset_index()
        )
        fig2 = px.bar(
            top,
            x="loves_count",
            y="brand_name",
            orientation="h",
            color="loves_count",
            color_continuous_scale=["#C8E6C9", "#66BB6A", "#2E7D32"],
            labels={"loves_count": "Promedio de Loves", "brand_name": ""},
        )
        fig2.update_layout(
            paper_bgcolor="#F6FBF8", plot_bgcolor="#F6FBF8",
            coloraxis_showscale=False,
            margin=dict(l=0, r=0, t=10, b=10),
            height=320,
            yaxis=dict(autorange="reversed"),
        )
        st.plotly_chart(fig2, use_container_width=True)

st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sentimiento de reseñas
# ---------------------------------------------------------------------------

st.markdown("<div class='section-title'>💬 Análisis de Sentimiento</div>", unsafe_allow_html=True)

# Datos estáticos de sentimiento (desde la base vía API de métricas)
sentiment_data = pd.DataFrame({
    "Sentimiento": ["Positivo", "Negativo", "Neutral"],
    "Cantidad": [358, 102, 39],
    "Color": ["#66BB6A", "#FF6B9D", "#C9B1FF"],
})

col3, col4 = st.columns([2, 3])

with col3:
    fig3 = px.pie(
        sentiment_data,
        names="Sentimiento",
        values="Cantidad",
        color="Sentimiento",
        color_discrete_map={"Positivo": "#66BB6A", "Negativo": "#FF6B9D", "Neutral": "#C9B1FF"},
        hole=0.5,
    )
    fig3.update_traces(textposition="inside", textinfo="percent+label", textfont_size=12)
    fig3.update_layout(
        paper_bgcolor="#F6FBF8",
        showlegend=False,
        margin=dict(l=0, r=0, t=10, b=10),
        height=260,
    )
    st.plotly_chart(fig3, use_container_width=True)

with col4:
    total_sent = sentiment_data["Cantidad"].sum()
    pct_pos = sentiment_data[sentiment_data["Sentimiento"] == "Positivo"]["Cantidad"].values[0] / total_sent * 100

    st.markdown(f"""
    <div style='background:white; border-radius:16px; padding:1.5rem; border:1px solid #C8E6C9;'>
        <p style='color:#555; font-size:0.95rem'>
        Se analizaron <b>499 reseñas</b> mediante la API de <b>Groq (LLaMA 3.1)</b> para clasificar
        el sentimiento de los compradores.
        </p>
        <p style='color:#555; font-size:0.95rem'>
        El <b>{pct_pos:.0f}%</b> de las reseñas analizadas expresa un sentimiento <b style='color:#2E7D32'>positivo</b>,
        lo que es consistente con la alta tasa de recomendación del catálogo.
        </p>
        <p style='color:#888; font-size:0.8rem'>
        Muestra estratificada por is_recommended · Modelo: llama-3.1-8b-instant
        </p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Tabla de productos filtrados
# ---------------------------------------------------------------------------

st.markdown("<div class='section-title'>📋 Catálogo de Productos</div>", unsafe_allow_html=True)

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
