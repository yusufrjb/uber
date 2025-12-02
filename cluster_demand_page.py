import streamlit as st

# Fungsi untuk menampilkan halaman permintaan berdasarkan cluster
def show_cluster_demand():
    st.title('Permintaan Berdasarkan Cluster')

    st.markdown("""
    Di halaman ini, kita akan melihat analisis permintaan berdasarkan cluster yang telah dibagi.
    Masing-masing cluster akan menampilkan hasil forecast untuk melihat pola permintaan yang lebih mendalam.
    """)

    # Menampilkan grafik dan data terkait permintaan cluster
    st.subheader("Permintaan Cluster Forecast")
    st.image("output_cluster_demand/cluster_0_forecast_plot_baseline.png", caption="Cluster 0 - Baseline")
    st.image("output_cluster_demand/cluster_0_forecast_plot_prophet.png", caption="Cluster 0 - Prophet")
