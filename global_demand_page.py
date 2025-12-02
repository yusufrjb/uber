import streamlit as st
import pandas as pd
import os
from utils import create_metric_card, create_info_box, create_section_divider

def show_global_demand():
    # Header
    st.markdown('<h1 class="main-header">🌍 Permintaan Global</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="subtitle">Analisis dan prediksi permintaan secara keseluruhan menggunakan model forecasting</p>',
        unsafe_allow_html=True
    )
    
    # Info Box
    st.markdown(
        create_info_box(
            "📊 Tentang Analisis Ini",
            "Dashboard ini menampilkan forecast permintaan global menggunakan dua pendekatan: "
            "Model Baseline (naive last value) dan Prophet (time series forecasting). "
            "Bandingkan performa kedua model untuk mendapatkan insight terbaik."
        ),
        unsafe_allow_html=True
    )
    
    # Load Metrics
    metrics_path = "output_global_demand/global_metrics.csv"
    if os.path.exists(metrics_path):
        df_metrics = pd.read_csv(metrics_path)
        
        # Create two columns for metrics
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📉 Model Baseline")
            baseline = df_metrics[df_metrics['model'] == 'baseline_naive'].iloc[0]
            
            metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
            with metrics_col1:
                st.markdown(
                    create_metric_card("MAE", f"{baseline['mae']:.2f}"),
                    unsafe_allow_html=True
                )
            with metrics_col2:
                st.markdown(
                    create_metric_card("RMSE", f"{baseline['rmse']:.2f}"),
                    unsafe_allow_html=True
                )
            with metrics_col3:
                st.markdown(
                    create_metric_card("MAPE", f"{baseline['mape']:.2f}%"),
                    unsafe_allow_html=True
                )
        
        with col2:
            if 'prophet' in df_metrics['model'].values:
                st.markdown("### 🔮 Model Prophet")
                prophet = df_metrics[df_metrics['model'] == 'prophet'].iloc[0]
                
                # Calculate improvement
                mae_improvement = ((baseline['mae'] - prophet['mae']) / baseline['mae'] * 100)
                
                metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
                with metrics_col1:
                    st.markdown(
                        create_metric_card(
                            "MAE", 
                            f"{prophet['mae']:.2f}",
                            f"{abs(mae_improvement):.1f}% {'better' if mae_improvement > 0 else 'worse'}",
                            "positive" if mae_improvement > 0 else "negative"
                        ),
                        unsafe_allow_html=True
                    )
                with metrics_col2:
                    st.markdown(
                        create_metric_card("RMSE", f"{prophet['rmse']:.2f}"),
                        unsafe_allow_html=True
                    )
                with metrics_col3:
                    st.markdown(
                        create_metric_card("MAPE", f"{prophet['mape']:.2f}%"),
                        unsafe_allow_html=True
                    )
    
    st.markdown(create_section_divider(), unsafe_allow_html=True)
    
    # Forecast Visualizations
    st.markdown("### 📈 Visualisasi Forecast")
    
    tab1, = st.tabs(["🟣 Model Prophet"])
    
    with tab1:
        baseline_plot = "output_global_demand_v2/global_forecast_plot_prophet_log1p.png"
        if os.path.exists(baseline_plot):
            st.markdown('<div class="image-container">', unsafe_allow_html=True)
            st.image(baseline_plot, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            st.caption("Model Baseline menggunakan pendekatan naive - memprediksi nilai terakhir dari data training")
        else:
            st.warning("Plot baseline tidak ditemukan")
    
    
    st.markdown(create_section_divider(), unsafe_allow_html=True)
    
    # Data Preview
    with st.expander("📋 Lihat Data Forecast"):
        forecast_data = "output_global_demand/global_forecast_prophet_test.csv"
        if os.path.exists(forecast_data):
            df_forecast = pd.read_csv(forecast_data)
            st.dataframe(df_forecast.tail(20), use_container_width=True)
            
            # Download button
            csv = df_forecast.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Data Forecast",
                data=csv,
                file_name="global_forecast.csv",
                mime="text/csv"
            )
        else:
            st.info("Data forecast belum tersedia")
    
    # Insights
    st.markdown(create_section_divider(), unsafe_allow_html=True)
    st.markdown("### 💡 Insights & Rekomendasi")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **Keuntungan Model Prophet:**
        - Menangkap seasonality (harian & mingguan)
        - Lebih akurat untuk pola yang kompleks
        - Confidence interval untuk uncertainty
        """)
    
    with col2:
        st.markdown("""
        **Aplikasi Bisnis:**
        - Optimasi alokasi driver
        - Perencanaan resource harian/mingguan
        - Antisipasi peak demand
        """)