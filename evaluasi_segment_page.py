import streamlit as st

# Fungsi untuk menampilkan halaman evaluasi segmentasi
def show_evaluasi_segment():
    st.title('Evaluasi Segmentasi')

    st.markdown("""
    Di halaman ini, kita akan melihat hasil evaluasi dari segmentasi yang telah dilakukan.
    Evaluasi ini akan menunjukkan seberapa efektif segmentasi dalam memisahkan pelanggan berdasarkan kebutuhan atau pola tertentu.
    """)

    # Menampilkan data evaluasi segmentasi
    st.subheader("Evaluasi Hasil Segmentasi")
    st.write("Hasil evaluasi segmentasi akan ditampilkan di sini.")
