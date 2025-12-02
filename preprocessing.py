# ============================================================
# preprocessing.py — GENERAL CLEAN + PREPROCESSOR BUILDER
# ============================================================

import os
import re
import json
import joblib
import inspect
from typing import List

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# ============================================================
# CONFIG
# ============================================================

INPUT_CSV = "ncr_ride_bookings.csv"  # ganti kalau nama file berbeda
OUT_DIR   = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)


# ============================================================
# HELPERS
# ============================================================

def to_snake(s: str) -> str:
    """Ubah nama kolom ke snake_case yang rapi."""
    s = s.strip().lower()
    s = re.sub(r"[^\w\s]", "", s)
    return re.sub(r"\s+", "_", s)


def rare_category_collapse(s: pd.Series, min_frac: float, other_label="Other"):
    """Gabungkan kategori yang jarang jadi 'Other'."""
    if s.dropna().empty:
        return s
    vc = s.value_counts(normalize=True, dropna=False)
    rare = set(vc[vc < min_frac].index)
    if not rare:
        return s
    return s.apply(lambda x: other_label if x in rare else x)


def iqr_cap(series: pd.Series, low_q=0.01, high_q=0.99):
    """Capping outlier berdasarkan quantile rendah & tinggi."""
    series = pd.to_numeric(series, errors="coerce")
    if series.dropna().empty:
        return series
    lo = series.quantile(low_q)
    hi = series.quantile(high_q)
    return series.clip(lower=lo, upper=hi)


def _norm_str(x):
    return str(x).lower().strip() if pd.notna(x) else ""


# ============================================================
# MAIN PREPROCESSING LOGIC
# ============================================================

print(f"📥 Loading: {INPUT_CSV}")
df = pd.read_csv(INPUT_CSV, dtype=str)
print("Shape awal:", df.shape)

# ------------------------------------------------------------
# 1) Normalisasi nama kolom
# ------------------------------------------------------------
df.columns = [to_snake(c) for c in df.columns]

# ------------------------------------------------------------
# 2) Normalisasi nilai null-like
# ------------------------------------------------------------
# kita mau treat 'null', 'NaN', 'N/A', dll sebagai NaN
NULL_LIKE_LOWER = {"", "null", "none", "na", "n/a", "nan"}

for c in df.columns:
    if df[c].dtype == "object":
        raw = df[c].astype(str)
        stripped = raw.str.strip()
        lowered  = stripped.str.lower()
        mask_null = lowered.isin(NULL_LIKE_LOWER)
        # assign nilai yang bukan null-like tetap pakai versi stripped (preserve case)
        df[c] = stripped.mask(mask_null, np.nan)

# hilangkan kutip berlebih di ID
for c in ("booking_id", "customer_id"):
    if c in df.columns:
        df[c] = df[c].str.replace(r'^"|"$', "", regex=True)


# ------------------------------------------------------------
# 3) DATETIME PARSING
# ------------------------------------------------------------
if {"date", "time"}.issubset(df.columns):
    df["datetime"] = pd.to_datetime(
        df["date"].astype(str) + " " + df["time"].astype(str),
        errors="coerce"
    )
    df = df.drop(columns=["date", "time"], errors="ignore")

elif "date" in df.columns:
    df["datetime"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.drop(columns=["date"], errors="ignore")

else:
    df["datetime"] = pd.NaT

# fitur waktu
if df["datetime"].notna().any():
    df["year"]        = df["datetime"].dt.year
    df["month"]       = df["datetime"].dt.month
    df["day"]         = df["datetime"].dt.day
    df["hour"]        = df["datetime"].dt.hour
    df["dow"]         = df["datetime"].dt.dayofweek  # 0=Mon
    df["day_name"]    = df["datetime"].dt.day_name()
    df["is_weekend"]  = df["dow"].isin([5, 6]).astype(int)
    df["is_peak"]     = df["hour"].isin([7, 8, 9, 17, 18, 19, 20]).astype(int)

    # time of day kategori
    def _time_of_day(h):
        if pd.isna(h):
            return np.nan
        h = int(h)
        if 5 <= h < 12:
            return "morning"
        elif 12 <= h < 17:
            return "afternoon"
        elif 17 <= h < 21:
            return "evening"
        else:
            return "night"

    df["time_of_day"] = df["hour"].apply(_time_of_day)
else:
    df["hour"]        = np.nan
    df["dow"]         = np.nan
    df["day_name"]    = np.nan
    df["is_weekend"]  = 0
    df["is_peak"]     = 0
    df["time_of_day"] = np.nan


# ------------------------------------------------------------
# 4) DEDUP + STATUS FLAGS
# ------------------------------------------------------------
if "booking_id" in df.columns:
    before = len(df)
    df = df.drop_duplicates(subset="booking_id")
    print(f"Dedup booking_id: {before} → {len(df)}")

# flag status booking
if "booking_status" in df.columns:
    df["is_completed"]  = df["booking_status"].apply(lambda x: int(_norm_str(x) == "completed"))
    df["is_incomplete"] = df["booking_status"].apply(lambda x: int("incomplete" in _norm_str(x)))
    df["is_cancelled"]  = df["booking_status"].apply(
        lambda x: int(("cancel" in _norm_str(x)) or ("no driver" in _norm_str(x)))
    )
else:
    df["is_completed"]  = 0
    df["is_incomplete"] = 0
    df["is_cancelled"]  = 0


# ------------------------------------------------------------
# 5) SMART NUMERIC CAST
# ------------------------------------------------------------
protect_cols = {"datetime", "booking_id", "customer_id"}

for col in df.columns:
    if col in protect_cols:
        continue
    if df[col].dtype == "object":
        # ganti koma desimal → titik untuk safety
        df[col] = df[col].str.replace(",", ".", regex=False)
        num_try = pd.to_numeric(df[col], errors="coerce")
        # convert hanya jika minimal 40% baris bisa jadi angka
        if num_try.notna().mean() >= 0.40:
            df[col] = num_try

# ------------------------------------------------------------
# 6) LOGICAL IMPUTATION (berdasarkan domain)
# ------------------------------------------------------------
print("\n🔄 Logical Imputation...")

count_cols = [
    "cancelled_rides_by_customer",
    "cancelled_rides_by_driver",
    "incomplete_rides"
]
for c in count_cols:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

# booking_value & ride_distance di-cancel seharusnya 0
if "booking_value" in df.columns:
    df["booking_value"] = pd.to_numeric(df["booking_value"], errors="coerce")
    df.loc[df["is_cancelled"] == 1, "booking_value"] = df.loc[df["is_cancelled"] == 1, "booking_value"].fillna(0)

if "ride_distance" in df.columns:
    df["ride_distance"] = pd.to_numeric(df["ride_distance"], errors="coerce")
    df.loc[df["is_cancelled"] == 1, "ride_distance"] = df.loc[df["is_cancelled"] == 1, "ride_distance"].fillna(0)

reason_cols = [
    "reason_for_cancelling_by_customer",
    "driver_cancellation_reason",
    "incomplete_rides_reason"
]
for c in reason_cols:
    if c in df.columns and df[c].dtype == "object":
        df[c] = df[c].fillna("Not Applicable")

if "payment_method" in df.columns:
    df["payment_method"] = df["payment_method"].fillna("Unknown")


# ------------------------------------------------------------
# 7) FEATURE GROUPING (untuk kebutuhan ML/segmentasi nanti)
# ------------------------------------------------------------
dtypes = df.dtypes
all_num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
all_cat_cols = [c for c in df.columns if df[c].dtype == "object" and c not in protect_cols]

# numeric yang akan diskalakan
scaled_num_cols = [
    "avg_vtat",
    "avg_ctat",
    "booking_value",
    "ride_distance",
    "driver_ratings",
    "customer_rating"
]
scaled_num_cols = [c for c in scaled_num_cols if c in all_num_cols]

# numeric lain tetap dipass-through (tapi tetap di-impute)
passthrough_num_cols = [c for c in all_num_cols if c not in scaled_num_cols]

cat_cols = all_cat_cols.copy()

print(f"📈 Scaled numeric: {scaled_num_cols}")
print(f"➡️ Passthrough nums: {passthrough_num_cols}")
print(f"🔤 Categorical: {cat_cols}")


# ------------------------------------------------------------
# 8) OUTLIER CAPPING
# ------------------------------------------------------------
for c in scaled_num_cols:
    df[c] = iqr_cap(df[c])


# ------------------------------------------------------------
# 9) RARE CATEGORY COLLAPSE
# ------------------------------------------------------------
for c in cat_cols:
    if c in ["pickup_location", "drop_location"]:
        df[c] = rare_category_collapse(df[c], min_frac=0.0005)
    else:
        df[c] = rare_category_collapse(df[c], min_frac=0.01)


# ------------------------------------------------------------
# 10) ENCODING + SCALING PIPELINE (untuk ML)
#     → bagian ini untuk nanti dipakai segmentasi / model lain
# ------------------------------------------------------------
ohe_kwargs = {"handle_unknown": "ignore"}
if "sparse_output" in inspect.signature(OneHotEncoder).parameters:
    ohe_kwargs["sparse_output"] = True
else:
    ohe_kwargs["sparse"] = True

numeric_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", RobustScaler())
    ]
)

categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(**ohe_kwargs))
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("scaled_nums", numeric_pipeline, scaled_num_cols),
        ("categoricals", categorical_pipeline, cat_cols),
        ("pass_nums", SimpleImputer(strategy="most_frequent"), passthrough_num_cols),
    ],
    remainder="drop",
)

print("\n⚙️ Fitting transformer untuk ML...")
X_prep = preprocessor.fit_transform(df)
print("Prepared shape (X_prep):", X_prep.shape)

# ------------------------------------------------------------
# 11) FEATURE NAME EXTRACTION
# ------------------------------------------------------------
feature_names: List[str] = []

# scaled numeric asli namanya masih sama
feature_names += scaled_num_cols

# nama fitur dummy dari OHE
ohe = preprocessor.named_transformers_["categoricals"].named_steps["onehot"]
cat_features = ohe.get_feature_names_out(cat_cols).tolist()
feature_names += cat_features

# numeric pass-through
feature_names += passthrough_num_cols

# simpan nama fitur supaya konsisten di modeling
feat_path = os.path.join(OUT_DIR, "feature_names.json")
with open(feat_path, "w", encoding="utf-8") as f:
    json.dump(feature_names, f, indent=2)
print(f"💾 Feature names saved → {feat_path}")


# ------------------------------------------------------------
# 12) SAVE CLEAN CSV (untuk analisis umum, KPI, dsb)
# ------------------------------------------------------------
df_clean = df.copy()
clean_path = os.path.join(OUT_DIR, "cleaned_data.csv")
df_clean.to_csv(clean_path, index=False)
print(f"\n💾 Clean CSV saved → {clean_path}")


# ------------------------------------------------------------
# 13) SAVE PREPROCESSOR MODEL (untuk ML/segmentasi)
# ------------------------------------------------------------
prep_path = os.path.join(OUT_DIR, "preprocessor.joblib")
joblib.dump(preprocessor, prep_path)
print(f"💾 Preprocessor saved → {prep_path}")

print("\n✅ PREPROCESSING DONE.")
