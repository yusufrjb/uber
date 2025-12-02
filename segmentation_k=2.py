# ============================================================
# segmentasi_customer.py
# - Segmentasi CUSTOMER (bukan trip)
# - Sumber: outputs/cleaned_data.csv
# - Output: outputs_customer_segmentation/*
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# ------------------------------------------------------------
# 0) Path & output dir
# ------------------------------------------------------------
OUT_DIR = "outputs"
CLEAN_CSV = os.path.join(OUT_DIR, "cleaned_data.csv")

OUT_CUST = "outputs_customer_segmentation_k_2"
os.makedirs(OUT_CUST, exist_ok=True)


# ------------------------------------------------------------
# (A) Load Cleaned Data
# ------------------------------------------------------------
def load_clean_data(path=CLEAN_CSV) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} tidak ditemukan. Pastikan preprocessing sudah dijalankan."
        )
    df = pd.read_csv(path)
    print(f"Loaded cleaned_data: {df.shape[0]} rows, {df.shape[1]} kolom")
    return df


# ------------------------------------------------------------
# (B) Build Customer Features
# ------------------------------------------------------------
def build_customer_features(df: pd.DataFrame) -> pd.DataFrame:
    if "customer_id" not in df.columns:
        raise ValueError("Kolom 'customer_id' tidak ditemukan di cleaned_data.csv")

    g = df.groupby("customer_id")

    cust = pd.DataFrame({
        "total_trips": g["booking_id"].count(),
        "completed_trips": g["is_completed"].sum(),
        "incomplete_trips": g["is_incomplete"].sum(),
        "cancelled_trips": g["is_cancelled"].sum(),
        "avg_booking_value": g["booking_value"].mean(),
        "avg_ride_distance": g["ride_distance"].mean(),
        "avg_driver_rating": g["driver_ratings"].mean(),
        "avg_customer_rating": g["customer_rating"].mean(),
        "avg_vtat": g["avg_vtat"].mean(),
        "avg_ctat": g["avg_ctat"].mean(),
        "weekend_ratio": g["is_weekend"].mean(),
        "peak_ratio": g["is_peak"].mean(),
    })

    cust["completion_rate"] = cust["completed_trips"] / cust["total_trips"]
    cust["cancel_rate"] = cust["cancelled_trips"] / cust["total_trips"]

    print("Shape customer-level features:", cust.shape)
    return cust


# ------------------------------------------------------------
# (C) Preprocessing (imputer + scaler)
# ------------------------------------------------------------
def preprocess_features(cust_feat: pd.DataFrame):
    feature_names = list(cust_feat.columns)

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    X_imputed = imputer.fit_transform(cust_feat)
    X_scaled = scaler.fit_transform(X_imputed)

    import joblib
    joblib.dump(imputer, os.path.join(OUT_CUST, "cust_imputer.joblib"))
    joblib.dump(scaler, os.path.join(OUT_CUST, "cust_scaler.joblib"))

    print("X_scaled shape:", X_scaled.shape)
    return X_scaled, feature_names


# ------------------------------------------------------------
# (D) PCA 2D Visualization
# ------------------------------------------------------------
def _save_pca_plot(X_scaled, labels):
    print("🔎 Membuat PCA 2D scatter...")

    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X_scaled)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1],
                          c=labels, cmap="viridis", s=8, alpha=0.5)

    plt.title("Customer Segmentation (PCA 2D)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.colorbar(scatter, label="Cluster")
    plt.tight_layout()

    out_path = os.path.join(OUT_CUST, "customer_pca_scatter.png")
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"💾 PCA scatter disimpan → {out_path}")


# ------------------------------------------------------------
# (E) Customer-Level KMeans Segmentation
# ------------------------------------------------------------
def run_customer_segmentation(
    k: int = 3,
    random_state: int = 42,
    save_csv: bool = True,
):

    df = load_clean_data()
    cust_feat = build_customer_features(df)

    X_scaled, feature_names = preprocess_features(cust_feat)

    print(f"\n🚀 Menjalankan KMeans customer-level dengan k={k}...")
    model = KMeans(
        n_clusters=k,
        random_state=random_state,
        n_init=10,
    )
    labels = model.fit_predict(X_scaled)

    # Tambah ke tabel customer
    cust_seg = cust_feat.copy()
    cust_seg["customer_segment"] = labels
    cust_seg = cust_seg.reset_index()  # penting untuk merge

    # Merge kembali ke trip-level
    df_with_seg = df.merge(
        cust_seg[["customer_id", "customer_segment"]],
        on="customer_id",
        how="left",
    )

    # Simpan CSV
    if save_csv:
        cust_seg_path = os.path.join(OUT_CUST, "segmented_customers.csv")
        trips_seg_path = os.path.join(OUT_CUST, "trips_with_customer_segment.csv")

        cust_seg.to_csv(cust_seg_path, index=False)
        df_with_seg.to_csv(trips_seg_path, index=False)

        print(f"💾 segmented_customers disimpan → {cust_seg_path}")
        print(f"💾 trips_with_customer_segment disimpan → {trips_seg_path}")

    # Distribusi cluster
    print("\nDistribusi customer per segment:")
    print(cust_seg["customer_segment"].value_counts().sort_index())

    # Summary numerik
    numeric_cols = [
        "total_trips", "completed_trips", "incomplete_trips", "cancelled_trips",
        "avg_booking_value", "avg_ride_distance",
        "avg_driver_rating", "avg_customer_rating",
        "avg_vtat", "avg_ctat",
        "weekend_ratio", "peak_ratio",
        "completion_rate", "cancel_rate",
    ]

    summary = cust_seg.groupby("customer_segment")[numeric_cols].mean().round(2)
    summary_path = os.path.join(OUT_CUST, "customer_cluster_summary_numeric.csv")
    summary.to_csv(summary_path)
    print(f"💾 Summary numerik disimpan → {summary_path}")

    # Simpan profil cluster (text)
    profile_path = os.path.join(OUT_CUST, "customer_cluster_profile.txt")
    with open(profile_path, "w", encoding="utf-8") as f:
        f.write("Customer Cluster Profiles\n")
        f.write("=========================\n\n")
        for c in sorted(cust_seg["customer_segment"].unique()):
            sub = cust_seg[cust_seg["customer_segment"] == c]
            f.write(f"Segment {c}\n")
            f.write(f"Jumlah customer : {len(sub)}\n")
            desc = sub[numeric_cols].mean().round(2)
            for col, val in desc.items():
                f.write(f"  avg_{col} : {val}\n")
            f.write("\n")

    print(f"💾 Profil cluster disimpan → {profile_path}")

    # TAMBAHAN: PCA
    _save_pca_plot(X_scaled, labels)

    return cust_seg, model, feature_names


# ------------------------------------------------------------
# (F) MAIN
# ------------------------------------------------------------
if __name__ == "__main__":
    cust_seg, model, feats = run_customer_segmentation(
        k=3,
        random_state=42,
        save_csv=True,
    )

    print("\nPreview segmented_customers:")
    print(cust_seg[[
        "customer_id", "customer_segment",
        "total_trips", "avg_booking_value", "avg_ride_distance"
    ]].head())
