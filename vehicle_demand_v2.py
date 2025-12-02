import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from prophet import Prophet

# ------------------------------------------------------------
# PATH CONFIG
# ------------------------------------------------------------

OUT_PREP  = "outputs"  # dari preprocessing.py
CLEAN_CSV = os.path.join(OUT_PREP, "cleaned_data.csv")

OUT_VEH = "output_vehicle_demand_v2"
os.makedirs(OUT_VEH, exist_ok=True)

# ------------------------------------------------------------
# UTILITIES
# ------------------------------------------------------------

def load_clean_data(path=CLEAN_CSV) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} tidak ditemukan. Jalankan preprocessing.py terlebih dahulu.")
    df = pd.read_csv(path, parse_dates=["datetime"])
    return df


def make_filename_safe(s: str) -> str:
    """Biar nama file aman (tanpa spasi & karakter aneh)."""
    return "".join(ch if ch.isalnum() else "_" for ch in s.lower())


def build_hourly_demand_for_vehicle(df: pd.DataFrame, vehicle_type: str) -> pd.DataFrame:
    """
    Ambil subset berdasarkan vehicle_type tertentu dan buat demand per jam.
    Output: kolom ['ds', 'y', 'vehicle_type']
    """
    sub = df[df["vehicle_type"] == vehicle_type].copy()
    if sub.empty:
        return None

    df_ts = (
        sub
        .groupby(pd.Grouper(key="datetime", freq="H"))
        .size()
        .reset_index(name="y")
        .rename(columns={"datetime": "ds"})
    )

    # Transformasi log untuk stabilitas data
    df_ts["y"] = np.log1p(df_ts["y"])  # Menggunakan log1p untuk stabilisasi

    df_ts["vehicle_type"] = vehicle_type
    df_ts = df_ts.sort_values("ds").reset_index(drop=True)

    return df_ts


def train_test_split_time(df_ts: pd.DataFrame, test_ratio: float = 0.2):
    """
    Split time-series berdasarkan urutan waktu (bukan random).
    """
    n = len(df_ts)
    n_test = int(n * test_ratio)
    n_train = n - n_test

    df_train = df_ts.iloc[:n_train].copy()
    df_test  = df_ts.iloc[n_train:].copy()

    return df_train, df_test


def baseline_naive_last(train: pd.DataFrame, test: pd.DataFrame):
    """
    Baseline: prediksi semua titik test = nilai terakhir di train.
    """
    last_value = train["y"].iloc[-1]
    y_pred = np.full(shape=len(test), fill_value=last_value)
    return y_pred


def model_prophet(train: pd.DataFrame, test: pd.DataFrame):
    """
    Train Prophet di data train, prediksi test.ds
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
    y_pred = forecast["yhat"].values

    return y_pred, forecast, m


def eval_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)   # versi kompatibel
    rmse = np.sqrt(mse)
    mape = (np.abs((y_true - y_pred) / np.maximum(y_true, 1e-6))).mean() * 100
    return mae, rmse, mape


def plot_forecast(df_train, df_test, y_pred, model_name: str, vehicle_slug: str):
    plt.figure(figsize=(12, 5))

    plt.plot(df_train["ds"], df_train["y"], label="Train", alpha=0.7)
    plt.plot(df_test["ds"], df_test["y"], label="Test (actual)", alpha=0.7)
    plt.plot(df_test["ds"], y_pred, label=f"Predicted ({model_name})", linewidth=2)

    plt.title(f"Hourly Demand Forecast - {vehicle_slug} ({model_name})")
    plt.xlabel("Time")
    plt.ylabel("Demand")
    plt.legend()
    plt.tight_layout()

    png_path = os.path.join(OUT_VEH, f"vehicle_{vehicle_slug}_forecast_plot_{model_name}.png")
    plt.savefig(png_path, dpi=300)
    plt.close()
    print(f"  💾 Plot ({model_name}) → {png_path}")


# ------------------------------------------------------------
# MAIN PER VEHICLE TYPE
# ------------------------------------------------------------

def run_vehicle_forecast():
    print("📥 Load cleaned data...")
    df = load_clean_data()
    print("Shape df_clean:", df.shape)

    if "vehicle_type" not in df.columns:
        raise ValueError("Kolom 'vehicle_type' tidak ditemukan di cleaned_data.csv")

    vehicle_types = sorted(df["vehicle_type"].dropna().unique())
    print("\n🚗 Ditemukan vehicle_type:", vehicle_types)

    all_metrics = []

    for vt in vehicle_types:
        print(f"\n============================================")
        print(f"🔎 Vehicle type: {vt}")
        print(f"============================================")

        vt_slug = make_filename_safe(vt)
        df_ts = build_hourly_demand_for_vehicle(df, vt)

        if df_ts is None or df_ts["y"].sum() == 0 or len(df_ts) < 50:
            print(f"  ⚠ Data terlalu sedikit atau kosong untuk {vt}, skip.")
            continue

        # simpan raw hourly demand per vehicle
        hour_path = os.path.join(OUT_VEH, f"vehicle_{vt_slug}_hourly.csv")
        df_ts.to_csv(hour_path, index=False)
        print(f"  💾 Hourly demand saved → {hour_path}")

        # train-test split
        df_train, df_test = train_test_split_time(df_ts, test_ratio=0.2)
        print(f"  Train: {len(df_train)} points | Test: {len(df_test)} points")

        # ------------------ Baseline -------------------
        print(f"  🚀 Baseline Naive untuk {vt}")
        y_pred_base = baseline_naive_last(df_train, df_test)
        mae, rmse, mape = eval_metrics(df_test["y"].values, y_pred_base)
        print(f"    Baseline MAE={mae:.2f} | RMSE={rmse:.2f} | MAPE={mape:.2f}%")

        # simpan forecast baseline
        df_fore_base = df_test[["ds", "y"]].copy()
        df_fore_base["y_pred"] = y_pred_base
        fore_base_path = os.path.join(OUT_VEH, f"vehicle_{vt_slug}_forecast_baseline.csv")
        df_fore_base.to_csv(fore_base_path, index=False)
        print(f"  💾 Forecast baseline → {fore_base_path}")

        plot_forecast(df_train, df_test, y_pred_base, "baseline", vt_slug)

        all_metrics.append({
            "vehicle_type": vt,
            "model": "baseline_naive",
            "mae": mae,
            "rmse": rmse,
            "mape": mape
        })

        # ------------------ Prophet --------------------
        print(f"  🚀 Prophet untuk {vt}")
        try:
            y_pred_prophet, forecast_obj, m = model_prophet(df_train, df_test)
            mae_p, rmse_p, mape_p = eval_metrics(df_test["y"].values, y_pred_prophet)
            print(f"    Prophet MAE={mae_p:.2f} | RMSE={rmse_p:.2f} | MAPE={mape_p:.2f}%")

            # simpan forecast prophet (test period)
            df_fore_prophet = df_test[["ds", "y"]].copy()
            df_fore_prophet["y_pred"] = y_pred_prophet
            fore_prop_path = os.path.join(OUT_VEH, f"vehicle_{vt_slug}_forecast_prophet_test.csv")
            df_fore_prophet.to_csv(fore_prop_path, index=False)
            print(f"  💾 Forecast Prophet (test) → {fore_prop_path}")

            plot_forecast(df_train, df_test, y_pred_prophet, "prophet", vt_slug)

            # optional: future forecast 7 hari ke depan (168 jam)
            future_h = 24 * 7
            future_full = m.make_future_dataframe(periods=future_h, freq="H")
            future_fore = m.predict(future_full)
            future_path = os.path.join(OUT_VEH, f"vehicle_{vt_slug}_forecast_prophet_future.csv")
            future_fore[["ds", "yhat", "yhat_lower", "yhat_upper"]].to_csv(
                future_path, index=False
            )
            print(f"  💾 Future forecast Prophet → {future_path}")

            all_metrics.append({
                "vehicle_type": vt,
                "model": "prophet",
                "mae": mae_p,
                "rmse": rmse_p,
                "mape": mape_p
            })

        except Exception as e:
            print(f"  ⚠ Prophet gagal untuk {vt}: {e}")
            
    # ------------------------------------------------
    # SIMPAN METRIK SEMUA VEHICLE
    # ------------------------------------------------
    if all_metrics:
        df_metrics = pd.DataFrame(all_metrics)
        metrics_path = os.path.join(OUT_VEH, "vehicle_demand_metrics.csv")
        df_metrics.to_csv(metrics_path, index=False)
        print(f"\n💾 Metrik semua model per vehicle disimpan → {metrics_path}")
        print(df_metrics)
    else:
        print("\n⚠ Tidak ada metrik yang tersimpan (mungkin semua series terlalu pendek?).")


if __name__ == "__main__":
    run_vehicle_forecast()
