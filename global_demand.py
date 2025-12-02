# ============================================================
# global_demand.py
# - Forecast GLOBAL HOURLY DEMAND (jumlah order per jam)
# - Uji beberapa model (baseline, Prophet) dan bandingkan
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

# Coba import Prophet (bisa pip install prophet / pip install cmdstanpy prophet)
try:
    from prophet import Prophet
    HAS_PROPHET = True
except Exception as e:
    print("⚠ Prophet tidak tersedia, hanya akan pakai baseline.")
    HAS_PROPHET = False

# ------------------------------------------------------------
# CONFIG PATH
# ------------------------------------------------------------

OUT_PREP  = "outputs"                  # dari preprocessing.py
CLEAN_CSV = os.path.join(OUT_PREP, "cleaned_data.csv")

OUT_GLOBAL = "output_global_demand"
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
    ds = datetime, y = demand
    """
    df_ts = (
        df
        .groupby(pd.Grouper(key="datetime", freq="H"))
        .size()
        .reset_index(name="y")
        .rename(columns={"datetime": "ds"})
    )

    # sort by time
    df_ts = df_ts.sort_values("ds").reset_index(drop=True)

    # simpan raw global demand
    base_path = os.path.join(OUT_GLOBAL, "global_hourly_demand.csv")
    df_ts.to_csv(base_path, index=False)
    print(f"💾 global_hourly_demand.csv disimpan → {base_path}")

    return df_ts


# ------------------------------------------------------------
# 2) TRAIN-TEST SPLIT
# ------------------------------------------------------------

def train_test_split_time(df_ts: pd.DataFrame, test_size_ratio: float = 0.2):
    """
    Split time-series berdasarkan proporsi.
    """
    n = len(df_ts)
    n_test = int(n * test_size_ratio)
    n_train = n - n_test

    df_train = df_ts.iloc[:n_train].copy()
    df_test  = df_ts.iloc[n_train:].copy()

    print(f"Total points: {n} | Train: {len(df_train)} | Test: {len(df_test)}")

    return df_train, df_test


# ------------------------------------------------------------
# 3) BASELINE MODEL (NAIVE LAST VALUE)
# ------------------------------------------------------------

def baseline_naive_last(train: pd.DataFrame, test: pd.DataFrame):
    """
    Baseline: prediksi semua titik test = nilai terakhir train.
    """
    last_value = train["y"].iloc[-1]
    y_pred = np.full(shape=len(test), fill_value=last_value)
    return y_pred


# ------------------------------------------------------------
# 4) PROPHET MODEL
# ------------------------------------------------------------

def model_prophet(train: pd.DataFrame, test: pd.DataFrame):
    """
    Train Prophet di data train, prediksi di periode test (berdasarkan ds).
    """
    m = Prophet(
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=False,
        seasonality_mode="additive",
    )
    m.fit(train[["ds", "y"]])

    future = pd.DataFrame({"ds": test["ds"].values})
    forecast = m.predict(future)

    # Prophet output punya kolom 'yhat'
    y_pred = forecast["yhat"].values
    return y_pred, forecast, m


# ------------------------------------------------------------
# 5) EVALUASI MODEL
# ------------------------------------------------------------

def eval_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)   # versi lama hanya ini
    rmse = np.sqrt(mse)                        # akar manual buat RMSE
    mape = (np.abs((y_true - y_pred) / np.maximum(y_true, 1e-6))).mean() * 100
    return mae, rmse, mape

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
    plt.ylabel("Demand")
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

    # buang jam yang mungkin tanpa data di awal/akhir kalau mau (opsional)
    df_ts = df_ts.dropna(subset=["y"])

    print("✂ Train-test split...")
    df_train, df_test = train_test_split_time(df_ts, test_size_ratio=0.2)

    results = []  # untuk simpan metrik

    # ------------------ Baseline Model -----------------------
    print("\n🚀 Model 1: Baseline Naive Last Value")
    y_pred_base = baseline_naive_last(df_train, df_test)
    mae, rmse, mape = eval_metrics(df_test["y"].values, y_pred_base)
    print(f"Baseline MAE={mae:.2f} | RMSE={rmse:.2f} | MAPE={mape:.2f}%")

    # simpan forecast baseline
    df_fore_base = df_test[["ds", "y"]].copy()
    df_fore_base["y_pred"] = y_pred_base
    df_fore_base.to_csv(
        os.path.join(OUT_GLOBAL, "global_forecast_baseline.csv"),
        index=False
    )

    plot_forecast(df_train, df_test, y_pred_base, model_name="baseline")

    results.append({
        "model": "baseline_naive",
        "mae": mae,
        "rmse": rmse,
        "mape": mape
    })

    # ------------------ Prophet Model ------------------------
    if HAS_PROPHET:
        print("\n🚀 Model 2: Prophet")
        y_pred_prophet, forecast_obj, m = model_prophet(df_train, df_test)
        mae_p, rmse_p, mape_p = eval_metrics(df_test["y"].values, y_pred_prophet)
        print(f"Prophet MAE={mae_p:.2f} | RMSE={rmse_p:.2f} | MAPE={mape_p:.2f}%")

        # simpan forecast prophet di test set
        df_fore_prophet = df_test[["ds", "y"]].copy()
        df_fore_prophet["y_pred"] = y_pred_prophet
        df_fore_prophet.to_csv(
            os.path.join(OUT_GLOBAL, "global_forecast_prophet_test.csv"),
            index=False
        )

        plot_forecast(df_train, df_test, y_pred_prophet, model_name="prophet")

        # Simpan model metric
        results.append({
            "model": "prophet",
            "mae": mae_p,
            "rmse": rmse_p,
            "mape": mape_p
        })

        # Optional: buat forecast ke depan, misal 7 hari ke depan (168 jam)
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

    # ------------------ Save Metrics -------------------------
    df_metrics = pd.DataFrame(results)
    metrics_path = os.path.join(OUT_GLOBAL, "global_metrics.csv")
    df_metrics.to_csv(metrics_path, index=False)
    print(f"\n💾 Metrics semua model disimpan → {metrics_path}")
    print(df_metrics)

    # Simpan juga dalam bentuk JSON (kalau mau dipakai Streamlit)
    metrics_json_path = os.path.join(OUT_GLOBAL, "global_metrics.json")
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"💾 Metrics JSON disimpan → {metrics_json_path}")

    print("\n✅ Selesai: Global demand forecasting (baseline + Prophet).")


if __name__ == "__main__":
    main()
