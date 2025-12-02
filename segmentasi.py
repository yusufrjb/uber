# ============================================================
# segmentasi.py
# - Pakai preprocessor.joblib + feature_names.json
# - Lakukan KMeans clustering (trip-based segmentation)
# - Simpan berbagai output analisis siap untuk Streamlit
# ============================================================

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from preprocessing_segmentation import prepare_segmentation_data

# Direktori input (dari preprocessing.py)
OUT_DIR = "outputs"
CLEAN_CSV = os.path.join(OUT_DIR, "cleaned_data.csv")
PREP_PATH = os.path.join(OUT_DIR, "preprocessor.joblib")
FEAT_PATH = os.path.join(OUT_DIR, "feature_names.json")

# Direktori khusus untuk hasil segmentasi
OUT_SEG = "outputs_segmentation"
os.makedirs(OUT_SEG, exist_ok=True)

def _save_cluster_plots_pca(X, labels, filename="cluster_pca_scatter.png"):
    """
    Buat scatter plot PCA 2D dari X dan warna berdasarkan cluster label.
    """
    print("🔎 Membuat PCA 2D untuk visualisasi cluster...")
    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, alpha=0.4, s=10)
    plt.title("Cluster Visualization (PCA 2D)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.colorbar(scatter, label="Cluster")
    plt.tight_layout()

    save_path = os.path.join(OUT_SEG, filename)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"💾 PCA scatter plot disimpan → {save_path}")

def _compute_feature_importance(model: KMeans, feature_names: list, top_n: int = 30):
    """
    Hitung 'importance' berdasarkan rentang centroid per fitur.
    Importance = max(centroid) - min(centroid).
    """
    centers = model.cluster_centers_
    ranges = centers.max(axis=0) - centers.min(axis=0)

    feat_imp = pd.DataFrame({
        "feature": feature_names,
        "range_across_clusters": ranges
    })

    feat_imp = feat_imp.sort_values("range_across_clusters", ascending=False)
    top_imp = feat_imp.head(top_n)

    out_path = os.path.join(OUT_SEG, "cluster_feature_importance.csv")
    top_imp.to_csv(out_path, index=False)
    print(f"💾 Feature importance (top {top_n}) disimpan → {out_path}")

    return top_imp

def _save_cluster_profiles_text(df_seg: pd.DataFrame, numeric_cols: list):
    """
    Membuat file teks berisi profil ringkas tiap cluster:
    - jumlah trip
    - rata-rata beberapa metrik numerik
    """
    txt_path = os.path.join(OUT_SEG, "cluster_profile.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("Cluster Profiles (trip-based segmentation)\n")
        f.write("=========================================\n\n")

        clusters = sorted(df_seg["trip_segment"].unique())
        for c in clusters:
            sub = df_seg[df_seg["trip_segment"] == c]
            f.write(f"Cluster {c}\n")
            f.write(f"Jumlah trip : {len(sub)}\n")

            if numeric_cols:
                desc = sub[numeric_cols].mean().round(2)
                for col, val in desc.items():
                    f.write(f"  avg_{col} : {val}\n")

            f.write("\n")
    print(f"💾 Cluster profile text disimpan → {txt_path}")

def run_segmentation(
    k: int = 3,
    filter_completed_only: bool = False,
    min_booking_value: float | None = None,
    random_state: int = 42,
    save_csv: bool = True,
):
    """
    Jalankan trip-based segmentation (KMeans) menggunakan preprocessor dan feature names
    yang sudah dibuat di preprocessing.py.
    Params
    ------
    k : int
        Jumlah cluster KMeans.
    filter_completed_only : bool
        Jika True, hanya cluster rides yang selesai (is_completed == 1).
    min_booking_value : float atau None
        Jika di-set, hanya rides dengan booking_value >= nilai ini yang di-cluster.
    random_state : int
        Seed untuk KMeans.
    save_csv : bool
        Jika True, simpan hasil segmentasi ke outputs_segmentation/segmented_trips.csv
    Return
    ------
    df_seg : DataFrame
        Data dengan kolom tambahan 'trip_segment'
    model : KMeans
        Model KMeans yang sudah di-fit
    feature_names : list[str]
        Nama fitur yang digunakan dalam X (hasil preprocessor)
    """
    # --------------------------------------------------------
    # 1) Cek keberadaan artefak preprocessing
    # --------------------------------------------------------
    if not os.path.exists(PREP_PATH):
        raise FileNotFoundError(
            f"{PREP_PATH} tidak ditemukan. Jalankan preprocessing.py terlebih dahulu."
        )
    if not os.path.exists(FEAT_PATH):
        raise FileNotFoundError(
            f"{FEAT_PATH} tidak ditemukan. Jalankan preprocessing.py terlebih dahulu."
        )
    if not os.path.exists(CLEAN_CSV):
        raise FileNotFoundError(
            f"{CLEAN_CSV} tidak ditemukan. Jalankan preprocessing.py terlebih dahulu."
        )

    # Load preprocessor & feature_names
    preprocessor = joblib.load(PREP_PATH)
    with open(FEAT_PATH, "r", encoding="utf-8") as f:
        feature_names = json.load(f)

    # --------------------------------------------------------
    # 2) Load dan siapkan data untuk segmentasi
    # --------------------------------------------------------
    df_seg = prepare_segmentation_data(
        path=CLEAN_CSV,
        filter_completed_only=filter_completed_only,
        min_booking_value=min_booking_value,
    )

    print("Shape data untuk segmentasi:", df_seg.shape)

    # --------------------------------------------------------
    # 3) Transform data menggunakan preprocessor
    # --------------------------------------------------------
    X = preprocessor.transform(df_seg)
    print("Shape X (setelah preprocessor):", X.shape)

    # --------------------------------------------------------
    # 4) KMeans Clustering
    # --------------------------------------------------------
    print(f"\n🚀 Menjalankan KMeans dengan k={k}...")
    model = KMeans(
        n_clusters=k,
        random_state=random_state,
        n_init=10  # pakai angka biar aman untuk berbagai versi sklearn
    )
    labels = model.fit_predict(X)

    # Tambahkan label cluster ke df_seg
    df_seg["trip_segment"] = labels

    # --------------------------------------------------------
    # 5) Simpan hasil utama
    # --------------------------------------------------------
    if save_csv:
        seg_path = os.path.join(OUT_SEG, "segmented_trips.csv")
        df_seg.to_csv(seg_path, index=False)
        print(f"💾 Hasil segmentasi disimpan ke → {seg_path}")

    print("\nDistribusi cluster (jumlah trip per segment):")
    print(df_seg["trip_segment"].value_counts().sort_index())

    # --------------------------------------------------------
    # 6) Ringkasan metrik numerik per cluster
    # --------------------------------------------------------
    base_num_cols = [
        "booking_value",
        "ride_distance",
        "driver_ratings",
        "customer_rating",
        "avg_vtat",
        "avg_ctat",
        "fare_per_km",
    ]
    numeric_cols = [c for c in base_num_cols if c in df_seg.columns]

    if numeric_cols:
        summary = (
            df_seg.groupby("trip_segment")[numeric_cols]
            .mean()
            .rename(columns=lambda x: f"avg_{x}")
            .round(2)
        )
        print("\nRingkasan rata-rata numerik per cluster:")
        print(summary)

        summary_path = os.path.join(OUT_SEG, "cluster_summary_numeric.csv")
        summary.to_csv(summary_path)
        print(f"💾 Ringkasan numerik per cluster disimpan → {summary_path}")

    # --------------------------------------------------------
    # 7) Ringkasan kategori (vehicle_type, payment_method, time_of_day)
    # --------------------------------------------------------
    def _save_crosstab(col_name: str, filename: str):
        if col_name in df_seg.columns:
            tab = pd.crosstab(
                df_seg["trip_segment"],
                df_seg[col_name],
                normalize="index"
            ) * 100.0
            tab = tab.round(2)
            out_path = os.path.join(OUT_SEG, filename)
            tab.to_csv(out_path)
            print(f"💾 Distribusi {col_name} per cluster disimpan → {out_path}")
            return tab
        return None

    vt_tab = _save_crosstab("vehicle_type", "cluster_vehicle_type.csv")
    pay_tab = _save_crosstab("payment_method", "cluster_payment_method.csv")
    tod_tab = _save_crosstab("time_of_day", "cluster_time_of_day.csv")

    # --------------------------------------------------------
    # 8) Feature importance (berdasarkan range centroid)
    # --------------------------------------------------------
    top_imp = _compute_feature_importance(model, feature_names, top_n=30)

    # --------------------------------------------------------
    # 9) PCA plot untuk visualisasi cluster
    # --------------------------------------------------------
    # X bisa berupa sparse matrix; convert dulu
    if not isinstance(X, np.ndarray):
        X_dense = X.toarray()
    else:
        X_dense = X
    _save_cluster_plots_pca(X_dense, labels)

    # --------------------------------------------------------
    # 10) Simpan profil cluster dalam bentuk teks
    # --------------------------------------------------------
    _save_cluster_profiles_text(df_seg, numeric_cols)

    return df_seg, model, feature_names


if __name__ == "__main__":
    # Contoh eksekusi default: k=3 (berdasarkan diskusi evaluasi K)
    df_seg, model, feats = run_segmentation(
        k=3,
        filter_completed_only=False,   # set True jika mau cluster hanya completed rides
        min_booking_value=None,        # misal 50 kalau mau exclude trip kecil sekali
    )

    # Contoh print beberapa kolom penting
    cols_preview = [
        c for c in ["booking_id", "customer_id", "booking_value", "ride_distance", "trip_segment"]
        if c in df_seg.columns
    ]
    print("\nPreview hasil:")
    print(df_seg[cols_preview].head())
