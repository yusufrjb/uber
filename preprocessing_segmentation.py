# ============================================================
# preprocessing_segmentation.py
# - Baca cleaned_data.csv
# - Siapkan df untuk segmentasi (filter baris, tapi kolom tetap)
# ============================================================

import os
import pandas as pd

# Lokasi default file hasil preprocessing umum
OUT_DIR = "outputs"
CLEAN_CSV = os.path.join(OUT_DIR, "cleaned_data.csv")


def load_clean_data(path: str = CLEAN_CSV) -> pd.DataFrame:
    """
    Load data bersih (cleaned_data.csv) yang dihasilkan oleh preprocessing.py.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} tidak ditemukan. Jalankan preprocessing.py terlebih dahulu."
        )

    df_clean = pd.read_csv(path)
    return df_clean


def prepare_segmentation_data(
    path: str = CLEAN_CSV,
    filter_completed_only: bool = False,
    min_booking_value: float | None = None,
) -> pd.DataFrame:
    """
    Siapkan data untuk segmentasi.
    Hanya mem-filter baris (tidak mengubah struktur kolom) agar tetap kompatibel
    dengan preprocessor.joblib yang sudah di-fit pada preprocessing.py.

    Params
    ------
    path : str
        Path ke cleaned_data.csv.
    filter_completed_only : bool
        Jika True, hanya ambil rides dengan is_completed == 1.
    min_booking_value : float atau None
        Jika di-set, hanya ambil rides dengan booking_value >= nilai ini.

    Return
    ------
    df_seg : DataFrame
        Data yang sudah difilter, siap di-transform oleh preprocessor.
    """
    df = load_clean_data(path)

    # Filter ride completed saja (opsional)
    if filter_completed_only and "is_completed" in df.columns:
        df = df[df["is_completed"] == 1].copy()

    # Filter minimal booking value (opsional)
    if (min_booking_value is not None) and ("booking_value" in df.columns):
        df = df[df["booking_value"] >= min_booking_value].copy()

    # Reset index setelah filter
    df.reset_index(drop=True, inplace=True)
    return df


if __name__ == "__main__":
    # Contoh pemakaian ketika file ini dijalankan langsung
    df_seg = prepare_segmentation_data(
        filter_completed_only=False,
        min_booking_value=None,
    )
    print("Shape df_seg:", df_seg.shape)
    print(df_seg.head())
