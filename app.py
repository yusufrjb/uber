import streamlit as st
from segmentasi_page import show_segmentasi
from global_demand_page import show_global_demand
from vehicle_demand_page import show_vehicle_demand
from utils import get_theme_tokens, inject_css

# ========================================
# Page Configuration
# ========================================
st.set_page_config(
    page_title="Dashboard Analisis Permintaan",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========================================
# Initialize Session State
# ========================================
def init_state():
    if "theme" not in st.session_state:
        st.session_state.theme = "Light"
    if "page" not in st.session_state:
        st.session_state.page = "Segmentasi Pelanggan"

init_state()

# Get theme tokens and inject CSS
t = get_theme_tokens(st.session_state.theme)
inject_css(t)

# ========================================
# Sidebar Navigation
# ========================================
with st.sidebar:
    st.markdown('<div class="sidebar-title">📊 Dashboard Analisis</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Theme Selector
    st.markdown('<div class="sidebar-section">🎨 Tema</div>', unsafe_allow_html=True)
    theme_choice = st.radio(
        "Pilih Tema",
        ["🌤️ Terang", "🌙 Gelap"],
        index=0 if st.session_state.theme == "Light" else 1,
        label_visibility="collapsed"
    )
    
    if "🌤️ Terang" in theme_choice:
        st.session_state.theme = "Light"
    else:
        st.session_state.theme = "Dark"
    
    st.markdown("---")
    
    # Page Navigation
    st.markdown('<div class="sidebar-section">🧭 Navigasi</div>', unsafe_allow_html=True)
    
    pages = {
        "📈 Segmentasi Pelanggan": "Segmentasi Pelanggan",
        "🌍 Permintaan Global": "Permintaan Global",
        "🚗 Permintaan Kendaraan": "Permintaan Kendaraan"
    }
    
    page_choice = st.radio(
        "Pilih Halaman",
        list(pages.keys()),
        label_visibility="collapsed"
    )
    
    st.session_state.page = pages[page_choice]
    
    st.markdown("---")
    
    # Footer Info
    st.markdown("""
    <div style="text-align: center; padding: 20px 0; color: #B3B3B3; font-size: 12px;">
        <p>Dashboard Analisis Permintaan</p>
        <p>v1.0.0 | 2025</p>
    </div>
    """, unsafe_allow_html=True)

# Update theme after navigation
t = get_theme_tokens(st.session_state.theme)
inject_css(t)

# ========================================
# Main Content Routing
# ========================================
if st.session_state.page == "Segmentasi Pelanggan":
    show_segmentasi()
elif st.session_state.page == "Permintaan Global":
    show_global_demand()
elif st.session_state.page == "Permintaan Kendaraan":
    show_vehicle_demand()