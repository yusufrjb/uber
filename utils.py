import streamlit as st

def get_theme_tokens(theme_name):
    """Return theme configuration tokens"""
    if theme_name == "Dark":
        return {
            "bg_primary": "#0E1117",
            "bg_secondary": "#1E1E1E",
            "bg_tertiary": "#262730",
            "text_primary": "#FAFAFA",
            "text_secondary": "#B3B3B3",
            "accent": "#FF4B4B",
            "accent_hover": "#FF6B6B",
            "border": "#2E2E2E",
            "shadow": "rgba(0,0,0,0.3)",
            "gradient_start": "#FF4B4B",
            "gradient_end": "#FF8E53",
        }
    else:
        return {
            "bg_primary": "#FFFFFF",
            "bg_secondary": "#F8F9FA",
            "bg_tertiary": "#F0F2F6",
            "text_primary": "#0E1117",
            "text_secondary": "#31333F",
            "accent": "#FF4B4B",
            "accent_hover": "#FF6B6B",
            "border": "#E0E0E0",
            "shadow": "rgba(0,0,0,0.1)",
            "gradient_start": "#FF4B4B",
            "gradient_end": "#FF8E53",
        }

def inject_css(t):
    """Inject custom CSS for enhanced styling"""
    css = f"""
    <style>
        /* Global Styles */
        .stApp {{
            background: {t['bg_primary']};
        }}
        
        /* Sidebar Styling */
        [data-testid="stSidebar"] {{
            background: {t['bg_secondary']};
            border-right: 1px solid {t['border']};
        }}
        
        .sidebar-title {{
            font-size: 24px;
            font-weight: 700;
            color: {t['text_primary']};
            padding: 20px 0;
            text-align: center;
            background: linear-gradient(135deg, {t['gradient_start']}, {t['gradient_end']});
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }}
        
        .sidebar-section {{
            font-size: 16px;
            font-weight: 600;
            color: {t['text_secondary']};
            padding: 10px 0;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        
        /* Main Content Area */
        .main-header {{
            font-size: 42px;
            font-weight: 800;
            color: {t['text_primary']};
            margin-bottom: 10px;
            background: linear-gradient(135deg, {t['gradient_start']}, {t['gradient_end']});
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }}
        
        .subtitle {{
            font-size: 18px;
            color: {t['text_secondary']};
            margin-bottom: 30px;
            line-height: 1.6;
        }}
        
        /* Card Component */
        .metric-card {{
            background: {t['bg_secondary']};
            border: 1px solid {t['border']};
            border-radius: 12px;
            padding: 24px;
            margin: 10px 0;
            box-shadow: 0 2px 8px {t['shadow']};
            transition: transform 0.2s, box-shadow 0.2s;
        }}
        
        .metric-card:hover {{
            transform: translateY(-4px);
            box-shadow: 0 4px 16px {t['shadow']};
        }}
        
        .metric-label {{
            font-size: 14px;
            color: {t['text_secondary']};
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 8px;
        }}
        
        .metric-value {{
            font-size: 32px;
            font-weight: 800;
            color: {t['text_primary']};
            margin-bottom: 8px;
        }}
        
        .metric-delta {{
            font-size: 14px;
            font-weight: 600;
        }}
        
        .metric-delta.positive {{
            color: #00D084;
        }}
        
        .metric-delta.negative {{
            color: #FF4B4B;
        }}
        
        /* Info Box */
        .info-box {{
            background: linear-gradient(135deg, {t['gradient_start']}15, {t['gradient_end']}15);
            border-left: 4px solid {t['accent']};
            border-radius: 8px;
            padding: 20px;
            margin: 20px 0;
        }}
        
        .info-box-title {{
            font-size: 18px;
            font-weight: 700;
            color: {t['text_primary']};
            margin-bottom: 10px;
        }}
        
        .info-box-content {{
            font-size: 15px;
            color: {t['text_secondary']};
            line-height: 1.6;
        }}
        
        /* Button Styling */
        .stButton>button {{
            background: linear-gradient(135deg, {t['gradient_start']}, {t['gradient_end']});
            color: white;
            border: none;
            border-radius: 8px;
            padding: 12px 24px;
            font-weight: 600;
            font-size: 16px;
            transition: transform 0.2s, box-shadow 0.2s;
        }}
        
        .stButton>button:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 12px {t['shadow']};
        }}
        
        /* Radio Button Styling */
        .stRadio > label {{
            font-weight: 600;
            color: {t['text_primary']};
        }}
        
        /* Image Container */
        .image-container {{
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 4px 16px {t['shadow']};
            margin: 20px 0;
        }}
        
        /* Section Divider */
        .section-divider {{
            height: 2px;
            background: linear-gradient(90deg, transparent, {t['accent']}, transparent);
            margin: 40px 0;
        }}
        
        /* Stats Row */
        .stats-row {{
            display: flex;
            gap: 20px;
            flex-wrap: wrap;
            margin: 20px 0;
        }}
        
        /* Hide Streamlit Branding */
        #MainMenu {{visibility: hidden;}}
        footer {{visibility: hidden;}}
        
        /* Dataframe Styling */
        .dataframe {{
            border-radius: 8px;
            overflow: hidden;
        }}
        
        /* Expander Styling */
        .streamlit-expanderHeader {{
            background: {t['bg_secondary']};
            border-radius: 8px;
            font-weight: 600;
        }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

def create_metric_card(label, value, delta=None, delta_color="positive"):
    """Create a styled metric card"""
    delta_html = ""
    if delta:
        delta_html = f'<div class="metric-delta {delta_color}">{"▲" if delta_color == "positive" else "▼"} {delta}</div>'
    
    html = f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        {delta_html}
    </div>
    """
    return html

def create_info_box(title, content):
    """Create a styled info box"""
    html = f"""
    <div class="info-box">
        <div class="info-box-title">{title}</div>
        <div class="info-box-content">{content}</div>
    </div>
    """
    return html

def create_section_divider():
    """Create a styled section divider"""
    return '<div class="section-divider"></div>'