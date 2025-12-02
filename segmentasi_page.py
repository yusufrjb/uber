import streamlit as st
import pandas as pd
import os
from utils import create_metric_card, create_info_box, create_section_divider

def show_segmentasi():
    # Header
    st.markdown('<h1 class="main-header">📈 Segmentasi Pelanggan</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="subtitle">Analisis clustering untuk memahami pola dan karakteristik perjalanan pelanggan</p>',
        unsafe_allow_html=True
    )
    
    # Info Box
    st.markdown(
        create_info_box(
            "🎯 Tujuan Segmentasi",
            "Segmentasi pelanggan menggunakan algoritma K-Means clustering untuk mengelompokkan "
            "perjalanan berdasarkan karakteristik seperti jarak, durasi, dan nilai booking. "
            "Hal ini memungkinkan strategi bisnis yang lebih targeted dan efektif."
        ),
        unsafe_allow_html=True
    )
    
    # Load segmented data
    seg_path = "outputs_customer_segmentation_k_2/segmented_customers.csv"
    
    if os.path.exists(seg_path):
        df_seg = pd.read_csv(seg_path)
        
        # Overview Metrics
        st.markdown("### 📊 Overview Segmentasi")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_trips = len(df_seg)
            st.markdown(
                create_metric_card("Total Trips", f"{total_trips:,}"),
                unsafe_allow_html=True
            )
        
        with col2:
            n_clusters = df_seg['customer_segment'].nunique()
            st.markdown(
                create_metric_card("Jumlah Cluster", f"{n_clusters}"),
                unsafe_allow_html=True
            )
        
        with col3:
            if 'avg_ride_distance' in df_seg.columns:
                avg_distance = df_seg['avg_ride_distance'].mean()
                st.markdown(
                    create_metric_card("Avg Distance", f"{avg_distance:.2f} km"),
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    create_metric_card("Avg Distance", "N/A"),
                    unsafe_allow_html=True
                )
        
        with col4:
            if 'avg_booking_value' in df_seg.columns:
                avg_value = df_seg['avg_booking_value'].mean()
                st.markdown(
                    create_metric_card("Avg Booking", f"{avg_value:,.0f}"),
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    create_metric_card("Avg Booking", "N/A"),
                    unsafe_allow_html=True
                )
        
        st.markdown(create_section_divider(), unsafe_allow_html=True)
        
        # Cluster Distribution
        st.markdown("### 🎯 Distribusi Cluster")
        
        cluster_counts = df_seg['customer_segment'].value_counts().sort_index()
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Create distribution chart data
            chart_data = pd.DataFrame({
                'Cluster': [f"Cluster {i}" for i in cluster_counts.index],
                'Jumlah Trips': cluster_counts.values,
                'Persentase': (cluster_counts.values / total_trips * 100).round(2)
            })
            
            st.bar_chart(chart_data.set_index('Cluster')['Jumlah Trips'])
        
        with col2:
            st.markdown("#### 📈 Detail Distribusi")
            for idx, count in cluster_counts.items():
                percentage = (count / total_trips) * 100
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Cluster {idx}</div>
                    <div class="metric-value">{count:,}</div>
                    <div class="metric-delta positive">{percentage:.1f}% dari total</div>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown(create_section_divider(), unsafe_allow_html=True)
        
        
        # Customer Segment Summary
        st.markdown("### 📊 Ringkasan Segmen Pelanggan")
        
        summary_path = "outputs_customer_segmentation_k_2/customer_cluster_summary_numeric.csv"
        if os.path.exists(summary_path):
            df_summary = pd.read_csv(summary_path)
            
            # Display summary table
            st.markdown("#### 📋 Detail Statistik per Segmen")
            
            # Format column names for better readability
            display_summary = df_summary.copy()
            column_mapping = {
                'customer_segment': 'Segmen',
                'total_trips': 'Total Trips',
                'completed_trips': 'Completed Trips',
                'incomplete_trips': 'Incomplete Trips',
                'cancelled_trips': 'Cancelled Trips',
                'avg_booking_value': 'Avg Booking ',
                'avg_ride_distance': 'Avg Distance (km)',
                'avg_driver_rating': 'Driver Rating',
                'avg_customer_rating': 'Customer Rating',
                'avg_cancelled_by_customer': 'Cancelled by Customer',
                'avg_cancelled_by_driver': 'Cancelled by Driver',
                'avg_incomplete_rides': 'Avg Incomplete Rides'
            }
            
            # Rename columns that exist
            for old_col, new_col in column_mapping.items():
                if old_col in display_summary.columns:
                    display_summary.rename(columns={old_col: new_col}, inplace=True)
            
            # Display styled dataframe with all columns
            st.dataframe(
                display_summary.style.format({
                    'Avg Booking ': '{:,.2f}',
                    'Avg Distance (km)': '{:.2f}',
                    'Driver Rating': '{:.2f}',
                    'Customer Rating': '{:.2f}',
                    'Cancelled by Customer': '{:.2f}',
                    'Cancelled by Driver': '{:.2f}',
                    'Avg Incomplete Rides': '{:.2f}'
                }).background_gradient(
                    subset=['Total Trips'] if 'Total Trips' in display_summary.columns else [],
                    cmap='Blues'
                ),
                use_container_width=True,
                hide_index=True
            )
            
            # Key insights from summary
            st.markdown("#### 💡 Insights Utama")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if 'total_trips' in df_summary.columns:
                    total_all = df_summary['total_trips'].sum()
                    st.markdown(
                        create_metric_card("Total Semua Trips", f"{total_all:,.0f}"),
                        unsafe_allow_html=True
                    )
            
            with col2:
                if 'completed_trips' in df_summary.columns and 'total_trips' in df_summary.columns:
                    total_completed = df_summary['completed_trips'].sum()
                    completion_rate = (total_completed / df_summary['total_trips'].sum() * 100)
                    st.markdown(
                        create_metric_card("Overall Completion Rate", f"{completion_rate:.1f}%"),
                        unsafe_allow_html=True
                    )
            
            with col3:
                if 'avg_booking_value' in df_summary.columns:
                    avg_all_booking = df_summary['avg_booking_value'].mean()
                    st.markdown(
                        create_metric_card("Avg Booking (All)", f"{avg_all_booking:,.0f}"),
                        unsafe_allow_html=True
                    )
            
            # Visualizations for segment summary
            st.markdown("#### 📈 Visualisasi Perbandingan")
            
            viz_tab1, viz_tab2, viz_tab3, viz_tab4 = st.tabs(["🎯 Trip Status", "💰 Booking & Distance", "⭐ Ratings", "❌ Cancellations"])
            
            with viz_tab1:
                if all(col in df_summary.columns for col in ['completed_trips', 'incomplete_trips', 'cancelled_trips']):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Trip status breakdown
                        st.markdown("**Status Trips per Segmen**")
                        status_data = df_summary[['customer_segment', 'completed_trips', 'incomplete_trips', 'cancelled_trips']].set_index('customer_segment')
                        status_data.columns = ['Completed', 'Incomplete', 'Cancelled']
                        st.bar_chart(status_data)
                    
                    with col2:
                        # Completion rate per segment
                        st.markdown("**Tingkat Penyelesaian per Segmen (%)**")
                        df_summary['completion_rate'] = (df_summary['completed_trips'] / df_summary['total_trips'] * 100).round(2)
                        
                        for idx, row in df_summary.iterrows():
                            segment = int(row['customer_segment'])
                            rate = row['completion_rate']
                            total = int(row['total_trips'])
                            st.markdown(
                                create_metric_card(
                                    f"Segmen {segment}",
                                    f"{rate:.1f}%",
                                    f"{total:,} trips"
                                ),
                                unsafe_allow_html=True
                            )
                else:
                    st.info("Data status trips tidak lengkap")
            
            with viz_tab2:
                col1, col2 = st.columns(2)
                
                with col1:
                    if 'avg_booking_value' in df_summary.columns:
                        st.markdown("**Rata-rata Booking Value**")
                        booking_data = df_summary[['customer_segment', 'avg_booking_value']].copy()
                        booking_data['customer_segment'] = booking_data['customer_segment'].apply(lambda x: f'Segmen {int(x)}')
                        booking_data = booking_data.set_index('customer_segment')
                        booking_data.columns = ['Avg Booking ']
                        st.bar_chart(booking_data)
                    else:
                        st.info("Data booking value tidak tersedia")
                
                with col2:
                    if 'avg_ride_distance' in df_summary.columns:
                        st.markdown("**Rata-rata Distance**")
                        distance_data = df_summary[['customer_segment', 'avg_ride_distance']].copy()
                        distance_data['customer_segment'] = distance_data['customer_segment'].apply(lambda x: f'Segmen {int(x)}')
                        distance_data = distance_data.set_index('customer_segment')
                        distance_data.columns = ['Avg Distance (km)']
                        st.bar_chart(distance_data)
                    else:
                        st.info("Data distance tidak tersedia")
            
            with viz_tab3:
                if all(col in df_summary.columns for col in ['avg_driver_rating', 'avg_customer_rating']):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Driver Ratings per Segmen**")
                        driver_rating_data = df_summary[['customer_segment', 'avg_driver_rating']].copy()
                        driver_rating_data['customer_segment'] = driver_rating_data['customer_segment'].apply(lambda x: f'Segmen {int(x)}')
                        driver_rating_data = driver_rating_data.set_index('customer_segment')
                        driver_rating_data.columns = ['Driver Rating']
                        st.bar_chart(driver_rating_data)
                    
                    with col2:
                        st.markdown("**Customer Ratings per Segmen**")
                        customer_rating_data = df_summary[['customer_segment', 'avg_customer_rating']].copy()
                        customer_rating_data['customer_segment'] = customer_rating_data['customer_segment'].apply(lambda x: f'Segmen {int(x)}')
                        customer_rating_data = customer_rating_data.set_index('customer_segment')
                        customer_rating_data.columns = ['Customer Rating']
                        st.bar_chart(customer_rating_data)
                else:
                    st.info("Data rating tidak tersedia")
            
            with viz_tab4:
                if all(col in df_summary.columns for col in ['avg_cancelled_by_customer', 'avg_cancelled_by_driver']):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Cancellations by Customer**")
                        cancel_cust_data = df_summary[['customer_segment', 'avg_cancelled_by_customer']].copy()
                        cancel_cust_data['customer_segment'] = cancel_cust_data['customer_segment'].apply(lambda x: f'Segmen {int(x)}')
                        cancel_cust_data = cancel_cust_data.set_index('customer_segment')
                        cancel_cust_data.columns = ['Avg Cancelled by Customer']
                        st.bar_chart(cancel_cust_data)
                    
                    with col2:
                        st.markdown("**Cancellations by Driver**")
                        cancel_driver_data = df_summary[['customer_segment', 'avg_cancelled_by_driver']].copy()
                        cancel_driver_data['customer_segment'] = cancel_driver_data['customer_segment'].apply(lambda x: f'Segmen {int(x)}')
                        cancel_driver_data = cancel_driver_data.set_index('customer_segment')
                        cancel_driver_data.columns = ['Avg Cancelled by Driver']
                        st.bar_chart(cancel_driver_data)
                else:
                    st.info("Data cancellation tidak tersedia")
            
            st.markdown(create_section_divider(), unsafe_allow_html=True)
        else:
            st.info("📊 Customer segment summary belum tersedia")
        
        # Feature Importance (if available)
        st.markdown("### 📊 Analisis Feature")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🔑 Key Features")
            st.markdown("""
            Feature yang digunakan dalam clustering:
            - **Distance**: Jarak perjalanan (km)
            - **Duration**: Durasi perjalanan (menit)
            - **Booking Value**: Nilai Booking
            - **Vehicle Type**: Tipe kendaraan
            - **Time Features**: Hour, day, month
            """)
        
        with col2:
            st.markdown("#### 📈 Cluster Quality")
            
            # Load k evaluation if exists
            k_eval_path = "outputs/k_evaluation.csv"
            if os.path.exists(k_eval_path):
                df_k_eval = pd.read_csv(k_eval_path)
                optimal_k = df_k_eval.loc[df_k_eval['silhouette'].idxmax()]
                
                st.markdown(f"""
                **Optimal K**: {int(optimal_k['k'])}  
                **Silhouette Score**: {optimal_k['silhouette']:.4f}  
                **Inertia**: {optimal_k['inertia']:.2f}
                """)
            else:
                st.info("Data evaluasi K belum tersedia")
        
        # Download section
        st.markdown(create_section_divider(), unsafe_allow_html=True)
        
        st.markdown("### 📥 Export Data")
        
        col1,  = st.columns(1)
        
        with col1:
            csv = df_seg.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Segmented Data (Full)",
                data=csv,
                file_name="segmented_trips_full.csv",
                mime="text/csv"
            )

    
    else:
        st.warning("⚠️ Data segmentasi belum tersedia. Silakan jalankan segmentasi.py terlebih dahulu.")
        
        st.markdown(
            create_info_box(
                "🚀 Cara Menjalankan Segmentasi",
                "1. Pastikan preprocessing.py sudah dijalankan\n"
                "2. Jalankan evaluasi_segment.py untuk menentukan optimal K\n"
                "3. Jalankan segmentasi.py untuk membuat cluster\n"
                "4. Refresh halaman ini untuk melihat hasil"
            ),
            unsafe_allow_html=True
        )