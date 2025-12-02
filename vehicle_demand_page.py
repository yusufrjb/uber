import streamlit as st
import pandas as pd
import os
from utils import create_metric_card, create_info_box, create_section_divider

def show_vehicle_demand():
    # Header
    st.markdown('<h1 class="main-header">🚗 Permintaan per Kendaraan</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="subtitle">Analisis dan prediksi permintaan berdasarkan tipe kendaraan untuk optimasi armada</p>',
        unsafe_allow_html=True
    )
    
    # Info Box
    st.markdown(
        create_info_box(
            "🎯 Vehicle Demand Forecasting",
            "Analisis ini membantu memahami pola permintaan untuk setiap tipe kendaraan, "
            "sehingga dapat mengoptimalkan alokasi armada dan meningkatkan service level. "
            "Model Prophet digunakan untuk menangkap seasonality dan trend."
        ),
        unsafe_allow_html=True
    )
    
    # Load vehicle metrics
    metrics_path = "output_vehicle_demand_v2/vehicle_demand_metrics.csv"
    
    if os.path.exists(metrics_path):
        df_metrics = pd.read_csv(metrics_path)
        
        # Get unique vehicle types
        vehicle_types = sorted(df_metrics['vehicle_type'].unique())
        
        # Overall metrics
        st.markdown("### 📊 Overview Semua Kendaraan")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(
                create_metric_card("Tipe Kendaraan", f"{len(vehicle_types)}"),
                unsafe_allow_html=True
            )
        
        with col2:
            # Average MAE across all vehicles (Prophet model)
            prophet_metrics = df_metrics[df_metrics['model'] == 'prophet']
            if not prophet_metrics.empty:
                avg_mae = prophet_metrics['mae'].mean()
                st.markdown(
                    create_metric_card("Avg MAE (Prophet)", f"{avg_mae:.2f}"),
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    create_metric_card("Avg MAE", "N/A"),
                    unsafe_allow_html=True
                )
        
        with col3:
            if not prophet_metrics.empty:
                avg_mape = prophet_metrics['mape'].mean()
                st.markdown(
                    create_metric_card("Avg MAPE (Prophet)", f"{avg_mape:.2f}%"),
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    create_metric_card("Avg MAPE", "N/A"),
                    unsafe_allow_html=True
                )
        
        st.markdown(create_section_divider(), unsafe_allow_html=True)
        
        # Vehicle type selector
        st.markdown("### 🚙 Pilih Tipe Kendaraan")
        
        # Create mapping for better display names
        vehicle_display_names = {
            vtype: f"🚗 {vtype.upper()}" for vtype in vehicle_types
        }
        
        selected_vehicle = st.selectbox(
            "Tipe Kendaraan",
            vehicle_types,
            format_func=lambda x: vehicle_display_names.get(x, x),
            label_visibility="collapsed"
        )
        
        st.markdown(create_section_divider(), unsafe_allow_html=True)
        
        # Display metrics for selected vehicle
        st.markdown(f"### 📈 Performa Model - {selected_vehicle.upper()}")
        
        vehicle_metrics = df_metrics[df_metrics['vehicle_type'] == selected_vehicle]
        
        if not vehicle_metrics.empty:
            col2, = st.columns(1)
            
            # Baseline metrics
            with col1:
                
                baseline = vehicle_metrics[vehicle_metrics['model'] == 'baseline']
                if not baseline.empty:
                    baseline = baseline.iloc[0]
                    
                    m_col1, m_col2, m_col3 = st.columns(3)
                    with m_col1:
                        st.markdown(
                            create_metric_card("MAE", f"{baseline['mae']:.2f}"),
                            unsafe_allow_html=True
                        )
                    with m_col2:
                        st.markdown(
                            create_metric_card("RMSE", f"{baseline['rmse']:.2f}"),
                            unsafe_allow_html=True
                        )
                    with m_col3:
                        st.markdown(
                            create_metric_card("MAPE", f"{baseline['mape']:.2f}%"),
                            unsafe_allow_html=True
                        )
                else:
                    pass
            
            # Prophet metrics
            with col2:
                st.markdown("#### 🔮 Prophet Model")
                prophet = vehicle_metrics[vehicle_metrics['model'] == 'prophet']
                if not prophet.empty:
                    prophet = prophet.iloc[0]
                    
                    # Calculate improvement
                    if not baseline.empty:
                        mae_improvement = ((baseline['mae'] - prophet['mae']) / baseline['mae'] * 100)
                    else:
                        mae_improvement = 0
                    
                    m_col1, m_col2, m_col3 = st.columns(3)
                    with m_col1:
                        st.markdown(
                            create_metric_card(
                                "MAE", 
                                f"{prophet['mae']:.2f}",
                                f"{abs(mae_improvement):.1f}% better" if mae_improvement > 0 else f"{abs(mae_improvement):.1f}% worse",
                                "positive" if mae_improvement > 0 else "negative"
                            ),
                            unsafe_allow_html=True
                        )
                    with m_col2:
                        st.markdown(
                            create_metric_card("RMSE", f"{prophet['rmse']:.2f}"),
                            unsafe_allow_html=True
                        )
                    with m_col3:
                        st.markdown(
                            create_metric_card("MAPE", f"{prophet['mape']:.2f}%"),
                            unsafe_allow_html=True
                        )
                else:
                    st.info("Data Prophet tidak tersedia")
        
        st.markdown(create_section_divider(), unsafe_allow_html=True)
        
        # Forecast visualizations
        st.markdown("### 📊 Visualisasi Forecast")
        
        tab1,  = st.tabs(["🟣 Prophet"])
        
        with tab1:
            baseline_plot = f"output_vehicle_demand_v2/vehicle_{selected_vehicle}_forecast_plot_prophet.png"
            if os.path.exists(baseline_plot):
                st.markdown('<div class="image-container">', unsafe_allow_html=True)
                st.image(baseline_plot, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
                st.caption(f"Forecast baseline untuk {selected_vehicle.upper()}")
            else:
                st.warning(f"Plot baseline untuk {selected_vehicle} tidak ditemukan")
        
        
        
        st.markdown(create_section_divider(), unsafe_allow_html=True)
        
        # Comparison table
        st.markdown("### 🔄 Perbandingan Antar Kendaraan")
        
        # Create comparison for Prophet models
        prophet_comparison = df_metrics[df_metrics['model'] == 'prophet'].copy()
        
        if not prophet_comparison.empty:
            comparison_display = prophet_comparison[['vehicle_type', 'mae', 'rmse', 'mape']].copy()
            comparison_display.columns = ['Tipe Kendaraan', 'MAE', 'RMSE', 'MAPE (%)']
            comparison_display['Tipe Kendaraan'] = comparison_display['Tipe Kendaraan'].str.upper()
            comparison_display = comparison_display.sort_values('MAE')
            
            # Style the dataframe
            st.dataframe(
                comparison_display.style.format({
                    'MAE': '{:.2f}',
                    'RMSE': '{:.2f}',
                    'MAPE (%)': '{:.2f}'
                }).background_gradient(subset=['MAE'], cmap='RdYlGn_r'),
                use_container_width=True,
                hide_index=True
            )
        
        # Data preview
        with st.expander("📋 Lihat Data Hourly Demand"):
            hourly_path = f"output_vehicle_demand_v2/vehicle_{selected_vehicle}_hourly.csv"
            if os.path.exists(hourly_path):
                df_hourly = pd.read_csv(hourly_path)
                st.dataframe(df_hourly.tail(20), use_container_width=True)
                
                csv = df_hourly.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label=f"📥 Download Data {selected_vehicle.upper()}",
                    data=csv,
                    file_name=f"vehicle_{selected_vehicle}_demand.csv",
                    mime="text/csv"
                )
            else:
                st.info(f"Data hourly untuk {selected_vehicle} tidak tersedia")
        
        st.markdown(create_section_divider(), unsafe_allow_html=True)
        
        # Insights and recommendations
        st.markdown("### 💡 Insights & Strategi Optimasi")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            **📊 Analisis Demand**
            - Monitor pola peak hours
            - Identifikasi seasonal patterns
            - Track demand fluctuations
            - Predict high/low periods
            """)
        
        with col2:
            st.markdown("""
            **🚗 Fleet Optimization**
            - Balance supply & demand
            - Optimize vehicle allocation
            - Reduce idle time
            - Improve utilization rate
            """)
        
        with col3:
            st.markdown("""
            **💰 Revenue Strategy**
            - Dynamic pricing model
            - Surge pricing optimization
            - Incentive management
            - Commission adjustment
            """)
        
        # Vehicle-specific recommendations
        st.markdown(create_section_divider(), unsafe_allow_html=True)
        st.markdown("### 🎯 Rekomendasi Spesifik per Tipe Kendaraan")
        
        vehicle_recommendations = {
            'car': {
                'icon': '🚗',
                'name': 'Regular Car',
                'tips': [
                    'Maintain high availability during peak hours (07:00-09:00, 17:00-19:00)',
                    'Focus on urban short-distance trips',
                    'Implement surge pricing during rush hours',
                    'Balance between volume and margin'
                ]
            },
            'bike': {
                'icon': '🏍️',
                'name': 'Motorcycle',
                'tips': [
                    'Highest demand for quick short trips',
                    'Optimize for heavy traffic conditions',
                    'Focus on instant availability',
                    'Lower price point, higher volume strategy'
                ]
            },
            'premium': {
                'icon': '🚙',
                'name': 'Premium Vehicle',
                'tips': [
                    'Target business hours and special events',
                    'Ensure high service quality',
                    'Premium pricing with added value',
                    'Focus on long-distance and airport trips'
                ]
            }
        }
        
        # Display recommendation for selected vehicle
        if selected_vehicle.lower() in vehicle_recommendations:
            rec = vehicle_recommendations[selected_vehicle.lower()]
            st.markdown(f"#### {rec['icon']} {rec['name']}")
            for tip in rec['tips']:
                st.markdown(f"✓ {tip}")
        else:
            st.info("Rekomendasi spesifik untuk tipe kendaraan ini sedang dikembangkan")
    
    else:
        st.warning("⚠️ Data metrics kendaraan belum tersedia. Silakan jalankan vehicle_demand.py terlebih dahulu.")
        
        st.markdown(
            create_info_box(
                "🚀 Cara Menjalankan Vehicle Demand",
                "1. Pastikan preprocessing.py sudah dijalankan\n"
                "2. Jalankan vehicle_demand.py untuk membuat forecast per vehicle\n"
                "3. Refresh halaman ini untuk melihat hasil"
            ),
            unsafe_allow_html=True
        )