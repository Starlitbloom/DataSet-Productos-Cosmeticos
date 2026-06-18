"""dashboards/app.py — Sephora Intelligence"""

import streamlit as st

st.set_page_config(
    page_title="Sephora Intelligence",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,300;0,400;0,600;1,300;1,400&family=Jost:wght@300;400;500&display=swap');

html, body, [class*="css"], .stApp {
    font-family: 'Jost', sans-serif;
    background-color: #FDFAF7 !important;
    color: #1A1A1A;
}

#MainMenu, footer { visibility: hidden; }
header { visibility: hidden; }
button[kind="headerNoPadding"],
[data-testid="stSidebarCollapseButton"],
[data-testid="collapsedControl"] { visibility: visible !important; display: flex !important; }
.block-container { padding: 0 3rem 4rem 3rem !important; max-width: 1100px; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #FFF0F5 0%, #FDF6FF 100%) !important;
    border-right: 1px solid #F0D6E0 !important;
}
[data-testid="stSidebarNav"] { display: none !important; }

/* Todos los page_link del sidebar */
[data-testid="stSidebar"] [data-testid="stPageLink"] {
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
    margin: 0 !important;
}
[data-testid="stSidebar"] [data-testid="stPageLink"] a,
[data-testid="stSidebar"] [data-testid="stPageLink"] p {
    font-family: 'Jost', sans-serif !important;
    font-size: 0.75rem !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    color: #5A3A4A !important;
    font-weight: 400 !important;
    text-decoration: none !important;
    padding: 0.5rem 1.5rem !important;
    display: block !important;
}
[data-testid="stSidebar"] [data-testid="stPageLink"]:hover a,
[data-testid="stSidebar"] [data-testid="stPageLink"]:hover p {
    color: #C2185B !important;
}

/* ── Botones ── */
/* Force cards equal height via column containers */
[data-testid="stVerticalBlock"] > [data-testid="stVerticalBlockBorderWrapper"],
[data-testid="stColumn"] > div > div > div {
    height: 100%;
}

/* Fix button position - add margin top auto equivalent */
[data-testid="stColumn"] .stButton {
    margin-top: 0.75rem !important;
}

.stButton > button {
    background: #C2185B !important;
    color: white !important;
    border: none !important;
    border-radius: 2px !important;
    padding: 0.65rem 2rem !important;
    font-family: 'Jost', sans-serif !important;
    font-size: 0.72rem !important;
    letter-spacing: 2.5px !important;
    text-transform: uppercase !important;
    font-weight: 400 !important;
    transition: all 0.25s ease !important;
    width: auto !important;
}
.stButton > button:hover {
    background: #8C0F3E !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 16px rgba(194,24,91,0.25) !important;
}

/* Monograma */
.monogram {
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.6rem;
    font-style: italic;
    color: #C2185B;
    border: 1.5px solid #D4A55A;
    border-radius: 50%;
    width: 56px; height: 56px;
    display: flex; align-items: center; justify-content: center;
    margin-bottom: 1rem;
}

/* Divider floral */
.divider-floral {
    display: flex; align-items: center; gap: 12px; margin: 1.2rem 0;
}
.divider-floral::before, .divider-floral::after {
    content: ''; flex: 1; height: 1px; background: linear-gradient(90deg, transparent, #E0A0B8);
}
.divider-floral::after { background: linear-gradient(90deg, #E0A0B8, transparent); }
.divider-floral span { color: #C2185B; font-size: 1rem; }

/* Hover card con elevacion */
.hover-card {
    background: linear-gradient(145deg, #FFFFFF 0%, #FFFAFB 100%);
    background-image: repeating-linear-gradient(45deg, transparent, transparent 18px, rgba(212,165,90,0.03) 18px, rgba(212,165,90,0.03) 19px), linear-gradient(145deg, #FFFFFF 0%, #FFFAFB 100%);
    border-radius: 4px;
    padding: 2rem 2rem 1.5rem 2rem;
    min-height: 260px;
    margin-bottom: 0.5rem;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    transition: all 0.3s ease;
    position: relative;
    overflow: hidden;
}
.hover-card::after {
    content: '';
    position: absolute; top: 0; right: 0;
    width: 50px; height: 50px;
    background: linear-gradient(135deg, transparent 50%, rgba(212,165,90,0.12) 50%);
}
.hover-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 16px 32px rgba(194,24,91,0.14);
}
</style>
""", unsafe_allow_html=True)

# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style='padding: 2.5rem 1.5rem 0.5rem 1.5rem;'>
        <p style='font-family: Cormorant Garamond, serif; font-size: 0.75rem; letter-spacing: 4px; text-transform: uppercase; color: #C2185B; margin: 0 0 0.3rem 0;'>✦ Sephora</p>
        <p style='font-family: Cormorant Garamond, serif; font-size: 1.6rem; font-style: italic; font-weight: 300; color: #1A1A1A; margin: 0 0 1.5rem 0;'>Intelligence</p>
        <div style='height:1px; background: linear-gradient(90deg, #F4A0B8, transparent); margin-bottom: 1.5rem;'></div>
    </div>
    """, unsafe_allow_html=True)

    st.page_link("app.py",                     label="Inicio")
    st.page_link("pages/1_vista_ejecutiva.py", label="Vista Ejecutiva")
    st.page_link("pages/2_vista_tecnica.py",   label="Vista Técnica")
    st.page_link("pages/3_vista_operativa.py", label="Vista Operativa")

    st.markdown("""
    <div style='margin-top: 3rem; padding: 1rem 1.5rem; border-top: 1px solid #F0D6E0;'>
        <p style='font-size: 0.68rem; color: #B08090; letter-spacing: 0.5px; line-height: 2;'>
            Sephora US Dataset<br>
            Gradient Boosting · K-Means<br>
            FastAPI · Groq LLaMA 3.1
        </p>
    </div>
    """, unsafe_allow_html=True)

# ── Hero ─────────────────────────────────────────────────────────────────────

st.markdown("""
<div style='
    background: linear-gradient(135deg, #FFF0F5 0%, #FDE8F0 40%, #F8F0FF 100%);
    padding: 4.5rem 3.5rem;
    margin: 0 -3rem 3rem -3rem;
    border-bottom: 1px solid #F0D0E0;
'>
    <p style='font-family: Jost, sans-serif; font-size: 0.68rem; letter-spacing: 4px; text-transform: uppercase; color: #C2185B; margin-bottom: 1.2rem;'>✦ Data Intelligence Platform</p>
    <h1 style='font-family: Cormorant Garamond, serif; font-size: 5rem; font-weight: 300; line-height: 1.05; color: #1A1A1A; margin: 0 0 1.5rem 0;'>
        Descubre los datos<br>
        <em style='color: #C2185B; font-style: italic;'>detrás de la belleza.</em>
    </h1>
    <p style='font-family: Jost, sans-serif; font-size: 0.95rem; color: #7A4A5A; max-width: 520px; line-height: 1.9; font-weight: 300; margin: 0;'>
        Análisis profundo de 8,494 productos y 924K reseñas de Sephora US.<br>
        Insights de negocio, modelo ML y patrones de consumo cosmético.
    </p>
</div>
""", unsafe_allow_html=True)

# ── Stats ────────────────────────────────────────────────────────────────────

col_s1, col_s2, col_s3, col_s4 = st.columns(4, gap="large")

for col, num, label, color in [
    (col_s1, "8,494",  "Productos",    "#C2185B"),
    (col_s2, "924K",   "Reseñas",      "#9B8EA8"),
    (col_s3, "499",    "Sentimientos", "#7AAB8E"),
    (col_s4, "2",      "Clusters",     "#D4956A"),
]:
    with col:
        st.markdown(f"""
        <div style='padding: 1.5rem 0; border-top: 3px solid {color};'>
            <p style='font-family: Cormorant Garamond, serif; font-size: 3rem; font-weight: 300; color: #1A1A1A; margin: 0 0 0.3rem 0; line-height: 1;'>{num}</p>
            <p style='font-size: 0.65rem; letter-spacing: 2.5px; text-transform: uppercase; color: #AAA; margin: 0;'>{label}</p>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<div style='height:1px; background:#EEE4E8; margin: 2rem 0 3rem 0;'></div>", unsafe_allow_html=True)

# ── Cards ────────────────────────────────────────────────────────────────────

st.markdown("""
<p style='font-size: 0.68rem; letter-spacing: 4px; text-transform: uppercase; color: #C2185B; margin-bottom: 2rem;'>✦ Explorar</p>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3, gap="large")

cards = [
    ("btn_exec", "pages/1_vista_ejecutiva.py", "01", "#C2185B", "#FFF5F8",
     "Vista Ejecutiva",
     "KPIs de negocio, top marcas por popularidad, distribución de precios en CLP y tasa de recomendación global."),
    ("btn_tech", "pages/2_vista_tecnica.py", "02", "#9B8EA8", "#F8F5FF",
     "Vista Técnica",
     "Parámetros del modelo Gradient Boosting, curvas de evaluación K-Means y métricas de reducción dimensional PCA."),
    ("btn_ops", "pages/3_vista_operativa.py", "03", "#7AAB8E", "#F5FBF7",
     "Vista Operativa",
     "Exploración interactiva del catálogo de productos. Filtra por categoría, marca y precio. Sentimiento con Groq."),
]

for col, (key, page, num, color, bg, title, desc) in zip([col1, col2, col3], cards):
    with col:
        st.markdown(f"""
        <div class='hover-card' style='border-top: 3px solid {color}; background: {bg};'>
            <div>
                <p style='font-size: 0.65rem; letter-spacing: 2px; text-transform: uppercase; color: {color}; margin-bottom: 0.8rem;'>{num}</p>
                <h3 style='font-family: Cormorant Garamond, serif; font-size: 1.9rem; font-weight: 300; color: #1A1A1A; margin-bottom: 0.8rem;'>{title}</h3>
                <p style='font-size: 0.83rem; color: #666; line-height: 1.75; font-weight: 300; margin: 0;'>{desc}</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Explorar →", key=key):
            st.switch_page(page)

# ── Footer ───────────────────────────────────────────────────────────────────

st.markdown("""
<div style='margin-top: 4rem; padding-top: 1.5rem; border-top: 1px solid #EEE4E8; display: flex; justify-content: space-between; align-items: center;'>
    <p style='font-size: 0.65rem; color: #CCC; letter-spacing: 2px; text-transform: uppercase; margin: 0;'>Sephora Intelligence · EP3 · 2026</p>
    <p style='font-size: 0.65rem; color: #CCC; letter-spacing: 1px; margin: 0;'>FastAPI · Streamlit · PostgreSQL</p>
</div>
""", unsafe_allow_html=True)