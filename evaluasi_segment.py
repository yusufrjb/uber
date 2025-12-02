# ============================================================
# evaluasi_segment.py — FAST K EVALUATION (SAMPLING + MINIBATCH)
# ============================================================

import os
import json
import joblib
import pandas as pd
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import silhouette_score

from preprocessing_segmentation import prepare_segmentation_data

# Lokasi aset dari preprocessing.py
OUT_DIR   = "outputs"
CLEAN_CSV = os.path.join(OUT_DIR, "cleaned_data.csv")
PREP_PATH = os.path.join(OUT_DIR, "preprocessor.joblib")
FEAT_PATH = os.path.join(OUT_DIR, "feature_names.json")


def evaluate_k(
    max_k: int = 8,
    sample_frac: float = 0.1,
    use_minibatch: bool = True,
    filter_completed_only: bool = False,
    min_booking_value: float | None = None,
):
    """
    Evaluasi jumlah cluster K (2..max_k) dengan:
      - optional sampling (sample_frac)
      - optional MiniBatchKMeans (lebih cepat)
      - hitung inertia + silhouette score

    Hasil disimpan ke: outputs/k_evaluation.csv
    """

    # --------------------------------------------------------
    # 1) Cek aset preprocessing
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

    # Load preprocessor & feature names
    preprocessor = joblib.load(PREP_PATH)
    with open(FEAT_PATH, "r", encoding="utf-8") as f:
        feature_names = json.load(f)

    # --------------------------------------------------------
    # 2) Load data untuk segmentasi
    # --------------------------------------------------------
    df_seg = prepare_segmentation_data(
        path=CLEAN_CSV,
        filter_completed_only=filter_completed_only,
        min_booking_value=min_booking_value,
    )

    print(f"Total data untuk evaluasi k: {df_seg.shape[0]} rows, {df_seg.shape[1]} kolom")

    # --------------------------------------------------------
    # 3) Sampling agar lebih cepat (kalau dataset besar)
    # --------------------------------------------------------
    if (sample_frac is not None) and (0 < sample_frac < 1.0) and (df_seg.shape[0] > 5000):
        print(f"⏳ Sampling {sample_frac*100:.1f}% data untuk evaluasi...")
        df_sample = df_seg.sample(frac=sample_frac, random_state=42)
    else:
        print("⚠️ Sampling dimatikan atau data kecil, pakai full data.")
        df_sample = df_seg

    print("Shape sample:", df_sample.shape)

    # Transform sample dengan preprocessor
    X = preprocessor.transform(df_sample)
    print("Shape X (setelah preprocessor):", X.shape)

    # --------------------------------------------------------
    # 4) Evaluasi k
    # --------------------------------------------------------
    inertias = []
    silhouettes = []
    ks = list(range(2, max_k + 1))

    ModelCls = MiniBatchKMeans if use_minibatch else KMeans
    model_type_name = "MiniBatchKMeans" if use_minibatch else "KMeans"
    print(f"\n🚀 Menggunakan {model_type_name} untuk evaluasi k...")

    for k in ks:
        print(f"\n=== Evaluasi k={k} ===")
        model = ModelCls(
            n_clusters=k,
            random_state=42,
            n_init=10,
            batch_size=2048 if use_minibatch else None,
        )

        labels = model.fit_predict(X)

        inertia = model.inertia_
        inertias.append(inertia)

        try:
            sil = silhouette_score(X, labels)
        except Exception as e:
            print(f"  ⚠️ Gagal hitung silhouette untuk k={k}: {e}")
            sil = float("nan")

        silhouettes.append(sil)

        print(f"  inertia    = {inertia:.2f}")
        print(f"  silhouette = {sil:.4f}" if sil == sil else f"  silhouette = NaN")

    # --------------------------------------------------------
    # 5) Simpan hasil evaluasi
    # --------------------------------------------------------
    result = pd.DataFrame({
        "k": ks,
        "inertia": inertias,
        "silhouette": silhouettes,
    })

    out_path = os.path.join(OUT_DIR, "k_evaluation.csv")
    result.to_csv(out_path, index=False)

    print(f"\n💾 Hasil evaluasi k disimpan → {out_path}")
    print(result)

    return result


if __name__ == "__main__":
    # Silakan sesuaikan parameter di sini
    evaluate_k(
        max_k=8,                # misal cek k=2..8
        sample_frac=0.10,       # 10% data untuk percepat
        use_minibatch=True,     # True = pakai MiniBatchKMeans
        filter_completed_only=False,
        min_booking_value=None,
    )
