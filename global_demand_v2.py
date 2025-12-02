# ============================================================
# global_demand_v2.py
# - Forecast GLOBAL HOURLY DEMAND (jumlah order per jam)
# - Bandingkan beberapa model:
#     1) Baseline Naive Last Value
#     2) Baseline Weekly Naive (copy nilai 1 minggu sebelumnya)
#     3) Prophet (dengan transformasi log1p)
# - Simpan output untuk analisis & Streamlit
# ============================================================

import os
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error

# ------------------------------------------------------------
# Coba import Prophet
# ------------------------------------------------------------
try:
    from prophet import Prophet
    HAS_PROPHET = True
except Exception:
    print("⚠ Prophet tidak tersedia, hanya akan pakai baseline.")
    HAS_PROPHET = False

# ------------------------------------------------------------
# CONFIG PATH
# ------------------------------------------------------------

OUT_PREP   = "outputs"  # dari preprocessing.py
CLEAN_CSV  = os.path.join(OUT_PREP, "cleaned_data.csv")

OUT_GLOBAL = "output_global_demand_v2"
os.makedirs(OUT_GLOBAL, exist_ok=True)


# ------------------------------------------------------------
# 1) LOAD & BUILD GLOBAL HOURLY DEMAND
# ------------------------------------------------------------

def load_clean_data(path=CLEAN_CSV) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} tidak ditemukan. Jalankan preprocessing.py terlebih dahulu.")
    df = pd.read_csv(path, parse_dates=["datetime"])
    return df


def build_hourly_demand(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agregasi jumlah order per jam.
    Output kolom: ['ds', 'y']
      - ds = timestamp (datetime)
      - y  = jumlah order (demand)
    """
    df_ts = (
        df
        .groupby(pd.Grouper(key="datetime", freq="H"))
        .size()
        .reset_index(name="y")
        .rename(columns={"datetime": "ds"})
    )

    df_ts = df_ts.sort_values("ds").reset_index(drop=True)

    # simpan raw global demand
    base_path = os.path.join(OUT_GLOBAL, "global_hourly_demand.csv")
    df_ts.to_csv(base_path, index=False)
    print(f"💾 global_hourly_demand.csv disimpan → {base_path}")

    return df_ts


# ------------------------------------------------------------
# 2) TRAIN-TEST SPLIT TIME SERIES
# ------------------------------------------------------------

def train_test_split_time(df_ts: pd.DataFrame, test_size_ratio: float = 0.2):
    """
    Split time-series berdasarkan proporsi (berdasarkan urutan waktu).
    """
    n = len(df_ts)
    n_test = int(n * test_size_ratio)
    n_train = n - n_test

    df_train = df_ts.iloc[:n_train].copy()
    df_test  = df_ts.iloc[n_train:].copy()

    print(f"Total points: {n} | Train: {len(df_train)} | Test: {len(df_test)}")

    return df_train, df_test


# ------------------------------------------------------------
# 3) BASELINE MODELS
# ------------------------------------------------------------

def baseline_naive_last(train: pd.DataFrame, test: pd.DataFrame):
    """
    Baseline 1: semua titik test diprediksi = nilai terakhir di train.
    """
    last_value = train["y"].iloc[-1]
    y_pred = np.full(shape=len(test), fill_value=last_value)
    return y_pred


def baseline_weekly_naive(train: pd.DataFrame, test: pd.DataFrame):
    """
    Baseline 2: untuk setiap timestamp di test,
    prediksi = nilai pada waktu yang sama 1 minggu (7 hari) sebelumnya.
    Jika tidak ada data 1 minggu sebelumnya, fallback ke last value train.
    """
    train_idx = train.set_index("ds")["y"]

    y_pred = []
    last_value = train["y"].iloc[-1]

    for t in test["ds"]:
        t_prev_week = t - pd.Timedelta(days=7)
        if t_prev_week in train_idx.index:
            y_pred.append(train_idx.loc[t_prev_week])
        else:
            y_pred.append(last_value)

    return np.array(y_pred)


# ------------------------------------------------------------
# 4) PROPHET MODEL (DENGAN TRANSFORMASI LOG1P)
# ------------------------------------------------------------

def model_prophet_log(train: pd.DataFrame, test: pd.DataFrame):
    """
    Prophet dengan transformasi log1p untuk stabilkan variance.
    Kita fit Prophet di y_log = log1p(y), lalu saat prediksi di-eksponen kembali.
    """
    # siapkan data train untuk Prophet
    df_train_p = train[["ds"]].copy()
    df_train_p["y"] = np.log1p(train["y"].values)

    m = Prophet(
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=False,
        seasonality_mode="additive",
        changepoint_prior_scale=0.5,
    )
    m.fit(df_train_p)

    # buat future = timestamp di test
    future = pd.DataFrame({"ds": test["ds"].values})
    forecast = m.predict(future)

    # yhat_log → balikkan dengan expm1
    yhat_log = forecast["yhat"].values
    y_pred = np.expm1(yhat_log)  # balik dari log1p

    # pastikan tidak negatif (kadang Prophet bisa menghasilkan nilai kecil negatif)
    y_pred = np.clip(y_pred, a_min=0, a_max=None)

    return y_pred, forecast, m


# ------------------------------------------------------------
# 5) METRIC EVALUATION
# ------------------------------------------------------------

def eval_metrics(y_true, y_pred):
    """
    Hitung:
      - MAE
      - RMSE (manual, tanpa squared=False)
      - MAPE (hanya y_true > 0)
      - sMAPE (lebih stabil untuk nilai kecil)
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    # MAE
    mae = mean_absolute_error(y_true, y_pred)

    # RMSE manual (karena mean_squared_error di versi sklearn kamu
    # tidak menerima parameter squared=False)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    # MAPE aman: hanya untuk y_true > 0
    mask = y_true > 0
    if mask.sum() > 0:
        mape = (np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])).mean() * 100
    else:
        mape = np.nan

    # sMAPE: 2 * |F - A| / (|F| + |A|)
    denom = np.abs(y_true) + np.abs(y_pred)
    denom = np.where(denom == 0, 1e-6, denom)
    smape = (2.0 * np.abs(y_pred - y_true) / denom).mean() * 100

    return mae, rmse, mape, smape


# ------------------------------------------------------------
# 6) PLOTTING
# ------------------------------------------------------------

def plot_forecast(df_train, df_test, y_pred, model_name: str):
    plt.figure(figsize=(12, 5))

    plt.plot(df_train["ds"], df_train["y"], label="Train", alpha=0.7)
    plt.plot(df_test["ds"], df_test["y"], label="Test (actual)", alpha=0.7)
    plt.plot(df_test["ds"], y_pred, label=f"Predicted ({model_name})", linewidth=2)

    plt.title(f"Global Hourly Demand Forecast - {model_name}")
    plt.xlabel("Time")
    plt.ylabel("Demand (#orders)")
    plt.legend()
    plt.tight_layout()

    png_path = os.path.join(OUT_GLOBAL, f"global_forecast_plot_{model_name}.png")
    plt.savefig(png_path, dpi=300)
    plt.close()
    print(f"💾 Plot forecast ({model_name}) disimpan → {png_path}")


# ------------------------------------------------------------
# 7) MAIN LOGIC
# ------------------------------------------------------------

def main():
    print("📥 Load cleaned data...")
    df = load_clean_data()
    print("Shape df_clean:", df.shape)

    print("🔁 Build global hourly demand...")
    df_ts = build_hourly_demand(df)

    df_ts = df_ts.dropna(subset=["y"])

    print("✂ Train-test split...")
    df_train, df_test = train_test_split_time(df_ts, test_size_ratio=0.2)

    results = []

    # ========================================================
    # MODEL 1: Baseline Naive Last Value
    # ========================================================
    print("\n🚀 Model 1: Baseline Naive Last Value")
    y_pred_last = baseline_naive_last(df_train, df_test)
    mae, rmse, mape, smape = eval_metrics(df_test["y"].values, y_pred_last)
    print(f"Baseline Last → MAE={mae:.2f} | RMSE={rmse:.2f} | MAPE={mape:.2f}% | sMAPE={smape:.2f}%")

    df_fore_last = df_test[["ds", "y"]].copy()
    df_fore_last["y_pred"] = y_pred_last
    df_fore_last.to_csv(
        os.path.join(OUT_GLOBAL, "global_forecast_baseline_last.csv"),
        index=False
    )
    plot_forecast(df_train, df_test, y_pred_last, model_name="baseline_last")

    results.append({
        "model": "baseline_last",
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "smape": smape,
    })

    # ========================================================
    # MODEL 2: Baseline Weekly Naive
    # ========================================================
    print("\n🚀 Model 2: Baseline Weekly Naive (copy nilai 1 minggu sebelumnya)")
    y_pred_weekly = baseline_weekly_naive(df_train, df_test)
    mae_w, rmse_w, mape_w, smape_w = eval_metrics(df_test["y"].values, y_pred_weekly)
    print(f"Baseline Weekly → MAE={mae_w:.2f} | RMSE={rmse_w:.2f} | MAPE={mape_w:.2f}% | sMAPE={smape_w:.2f}%")

    df_fore_weekly = df_test[["ds", "y"]].copy()
    df_fore_weekly["y_pred"] = y_pred_weekly
    df_fore_weekly.to_csv(
        os.path.join(OUT_GLOBAL, "global_forecast_baseline_weekly.csv"),
        index=False
    )
    plot_forecast(df_train, df_test, y_pred_weekly, model_name="baseline_weekly")

    results.append({
        "model": "baseline_weekly",
        "mae": mae_w,
        "rmse": rmse_w,
        "mape": mape_w,
        "smape": smape_w,
    })

    # ========================================================
    # MODEL 3: Prophet (log1p)
    # ========================================================
    if HAS_PROPHET:
        print("\n🚀 Model 3: Prophet (log1p)")
        y_pred_prophet, forecast_obj, m = model_prophet_log(df_train, df_test)
        mae_p, rmse_p, mape_p, smape_p = eval_metrics(df_test["y"].values, y_pred_prophet)
        print(f"Prophet (log1p) → MAE={mae_p:.2f} | RMSE={rmse_p:.2f} | "
              f"MAPE={mape_p:.2f}% | sMAPE={smape_p:.2f}%")

        df_fore_prophet = df_test[["ds", "y"]].copy()
        df_fore_prophet["y_pred"] = y_pred_prophet
        df_fore_prophet.to_csv(
            os.path.join(OUT_GLOBAL, "global_forecast_prophet_test.csv"),
            index=False
        )
        plot_forecast(df_train, df_test, y_pred_prophet, model_name="prophet_log1p")

        results.append({
            "model": "prophet_log1p",
            "mae": mae_p,
            "rmse": rmse_p,
            "mape": mape_p,
            "smape": smape_p,
        })

        # Optional: forecast ke depan (misal 7 hari = 168 jam)
        future_h = 24 * 7
        future_full = m.make_future_dataframe(periods=future_h, freq="H")
        forecast_full = m.predict(future_full)

        future_path = os.path.join(OUT_GLOBAL, "global_forecast_prophet_future.csv")
        forecast_full[["ds", "yhat", "yhat_lower", "yhat_upper"]].to_csv(
            future_path, index=False
        )
        print(f"💾 Future forecast Prophet disimpan → {future_path}")
    else:
        print("\n⚠ Prophet tidak tersedia, lewati model Prophet.")

    # ========================================================
    # SIMPAN METRIK
    # ========================================================
    df_metrics = pd.DataFrame(results)
    metrics_path = os.path.join(OUT_GLOBAL, "global_metrics_v2.csv")
    df_metrics.to_csv(metrics_path, index=False)
    print(f"\n💾 Metrics semua model disimpan → {metrics_path}")
    print(df_metrics)

    metrics_json_path = os.path.join(OUT_GLOBAL, "global_metrics_v2.json")
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"💾 Metrics JSON disimpan → {metrics_json_path}")

    print("\n✅ Selesai: Global demand forecasting v2 (baseline + Prophet log1p).")


# ------------------------------------------------------------
# 8) RUN
# ------------------------------------------------------------
if __name__ == "__main__":
    main()
