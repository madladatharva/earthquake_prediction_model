"""
earthquake_predictor_demo.py
Educational demo: predict the next earthquake's (lat, lon, mag) from the previous k events.

Not for operational use. For learning/research only.
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import math
import time
import warnings
warnings.filterwarnings("ignore")

# -----------------------
# 1) Utilities
# -----------------------
def fetch_usgs(starttime, endtime, min_magnitude=2.5, limit=20000):
    """
    Fetch earthquake events from USGS API between starttime and endtime (strings 'YYYY-MM-DD').
    Returns a DataFrame sorted by time ascending.
    """
    url = "https://earthquake.usgs.gov/fdsnws/event/1/query"
    params = {
        "format": "geojson",
        "starttime": starttime,
        "endtime": endtime,
        "minmagnitude": min_magnitude,
        "limit": limit
    }
    print(f"Fetching USGS events {starttime} → {endtime}, min_magnitude={min_magnitude}")
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    rows = []
    for f in data.get("features", []):
        prop = f.get("properties", {})
        geom = f.get("geometry", {})
        coords = geom.get("coordinates", [None, None, None])
        # coords = [lon, lat, depth]
        time_ms = prop.get("time")
        if time_ms is None: continue
        t = datetime.utcfromtimestamp(time_ms / 1000.0)
        lon, lat, depth = coords[0], coords[1], coords[2] if len(coords) > 2 else None
        mag = prop.get("mag")
        place = prop.get("place")
        rows.append((t, lat, lon, depth, mag, place))
    df = pd.DataFrame(rows, columns=["time", "latitude", "longitude", "depth", "magnitude", "place"])
    df = df.dropna(subset=["time", "latitude", "longitude", "magnitude"])
    df = df.sort_values("time").reset_index(drop=True)
    print(f"Fetched {len(df)} events.")
    return df

def haversine_km(lat1, lon1, lat2, lon2):
    """
    Haversine distance (km) between points.
    Accepts scalars or numpy arrays (elementwise).
    """
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2.0)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    R = 6371.0
    return R * c

# -----------------------
# 2) Build supervised sequences
# -----------------------
def build_sequences(df, k=5):
    """
    From chronological DataFrame df, build X and y:
    - For each index i >= k, take previous k events (i-k ... i-1) as input features,
      and event at i as target (lat, lon, mag).
    - Feature per subevent: [latitude, longitude, magnitude, time_lag_hours]
      where time_lag_hours = (time_of_last_in_window - time_of_sub_event) in hours.
    Returns:
      X_flat: (n_samples, k*4)
      X_seq: (n_samples, k, 4)  -> for LSTM
      y: (n_samples, 3)         -> [lat, lon, mag]
      times: list of target event times (for splitting/evaluation)
    """
    rows_X = []
    rows_X_seq = []
    rows_y = []
    times = []
    n = len(df)
    if n < k + 1:
        return np.array([]), np.array([]), np.array([]), []
    for i in range(k, n):
        last_time = df.loc[i-1, "time"]
        features = []
        seq = []
        valid = True
        for j in range(i-k, i):
            r = df.loc[j]
            if pd.isnull(r["magnitude"]): 
                valid = False
                break
            time_lag_hours = (last_time - r["time"]).total_seconds() / 3600.0
            feat = [r["latitude"], r["longitude"], r["magnitude"], time_lag_hours]
            features.extend(feat)
            seq.append(feat)
        if not valid:
            continue
        target = df.loc[i]
        rows_X.append(features)
        rows_X_seq.append(seq)
        rows_y.append([target["latitude"], target["longitude"], target["magnitude"]])
        times.append(target["time"])
    X_flat = np.array(rows_X, dtype=float)
    X_seq = np.array(rows_X_seq, dtype=float)
    y = np.array(rows_y, dtype=float)
    return X_flat, X_seq, y, times

# -----------------------
# 3) Training / evaluation helpers
# -----------------------
def time_split(n_samples, test_fraction=0.2):
    split = int(n_samples * (1 - test_fraction))
    return split

def evaluate_predictions(y_true, y_pred):
    """
    y_true, y_pred shape (n,3): [lat, lon, mag]
    Returns dict with mean/median loc error (km) and mag RMSE.
    """
    lat_t = y_true[:,0]; lon_t = y_true[:,1]; mag_t = y_true[:,2]
    lat_p = y_pred[:,0]; lon_p = y_pred[:,1]; mag_p = y_pred[:,2]
    dists = haversine_km(lat_t, lon_t, lat_p, lon_p)
    mag_rmse = math.sqrt(mean_squared_error(mag_t, mag_p))
    return {
        "mean_location_km": float(np.mean(dists)),
        "median_location_km": float(np.median(dists)),
        "mag_rmse": float(mag_rmse)
    }

# -----------------------
# 4) Main demo flow
# -----------------------
def main_demo():
    # -------- configure --------
    # Fetch one year of data (adjust as needed). If internet blocked, load saved CSV.
    end = datetime.utcnow()
    start = end - timedelta(days=365*2)   # 2 years
    start_str = start.strftime("%Y-%m-%d")
    end_str = end.strftime("%Y-%m-%d")
    min_mag = 2.5
    k = 6  # number of previous events to use
    test_frac = 0.2

    # -------- get data --------
    df = fetch_usgs(start_str, end_str, min_magnitude=min_mag)
    print("Example rows:\n", df.head().to_string(index=False))
    if len(df) < k + 10:
        raise RuntimeError("Not enough events fetched. Try lowering min_magnitude or expanding time window.")

    # -------- build sequences --------
    X_flat, X_seq, y, times = build_sequences(df, k=k)
    print(f"Built {len(X_flat)} supervised samples (k={k}).")

    # -------- train/test split (time-based) --------
    split = time_split(len(X_flat), test_fraction=test_frac)
    X_train_flat, X_test_flat = X_flat[:split], X_flat[split:]
    X_train_seq, X_test_seq = X_seq[:split], X_seq[split:]
    y_train, y_test = y[:split], y[split:]
    times_train, times_test = times[:split], times[split:]

    # -------- scale features (fit on train only) --------
    scaler = StandardScaler()
    # scaler expects 2D array: X_train_flat already that shape
    scaler.fit(X_train_flat)
    X_train_scaled = scaler.transform(X_train_flat)
    X_test_scaled = scaler.transform(X_test_flat)

    # -------- Baseline model: Random Forest (multi-output) --------
    print("\nTraining RandomForest baseline...")
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train_scaled, y_train)
    y_pred_rf = rf.predict(X_test_scaled)
    rf_stats = evaluate_predictions(y_test, y_pred_rf)
    print("RandomForest evaluation:", rf_stats)

    # -------- Simple Linear Regression baseline --------
    print("\nTraining LinearRegression baseline...")
    lr = LinearRegression()
    lr.fit(X_train_scaled, y_train)
    y_pred_lr = lr.predict(X_test_scaled)
    lr_stats = evaluate_predictions(y_test, y_pred_lr)
    print("LinearRegression evaluation:", lr_stats)

    # -------- Optional: LSTM (PyTorch) for sequence modeling --------
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import Dataset, DataLoader

        print("\nTraining LSTM model (this may take a while)...")
        # Prepare data for LSTM: we must scale per-feature
        n_samples, seq_len, n_feats = X_seq.shape
        reshaped = X_seq.reshape(-1, n_feats)  # (n_samples*seq_len, n_feats)
        feat_scaler = StandardScaler().fit(reshaped[:split*seq_len])  # fit only on train part
        reshaped_scaled = feat_scaler.transform(reshaped)
        X_seq_scaled = reshaped_scaled.reshape(n_samples, seq_len, n_feats)
        X_train_l = X_seq_scaled[:split]
        X_test_l = X_seq_scaled[split:]

        # torch datasets
        class EQDataset(Dataset):
            def __init__(self, X, y):
                self.X = torch.tensor(X, dtype=torch.float32)
                self.y = torch.tensor(y, dtype=torch.float32)
            def __len__(self): return len(self.X)
            def __getitem__(self, idx): return self.X[idx], self.y[idx]

        train_ds = EQDataset(X_train_l, y_train)
        test_ds = EQDataset(X_test_l, y_test)
        train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

        class LSTMRegressor(nn.Module):
            def __init__(self, input_size, hidden_size=64, num_layers=1):
                super().__init__()
                self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
                self.fc = nn.Linear(hidden_size, 3)
            def forward(self, x):
                out, _ = self.lstm(x)
                last = out[:, -1, :]
                return self.fc(last)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = LSTMRegressor(input_size=n_feats, hidden_size=64, num_layers=1).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()

        epochs = 12
        for epoch in range(epochs):
            model.train()
            total_loss = 0.0
            for xb, yb in train_loader:
                xb = xb.to(device); yb = yb.to(device)
                optimizer.zero_grad()
                out = model(xb)
                loss = criterion(out, yb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * xb.size(0)
            avg_loss = total_loss / len(train_ds)
            # quick eval on test
            model.eval()
            preds = []
            trues = []
            with torch.no_grad():
                for xb, yb in test_loader:
                    xb = xb.to(device)
                    out = model(xb).cpu().numpy()
                    preds.append(out)
                    trues.append(yb.numpy())
            preds = np.vstack(preds); trues = np.vstack(trues)
            stats = evaluate_predictions(trues, preds)
            print(f"Epoch {epoch+1}/{epochs} train_loss={avg_loss:.4f} test_mean_loc_km={stats['mean_location_km']:.2f} mag_rmse={stats['mag_rmse']:.3f}")

        y_pred_lstm = preds
        lstm_stats = evaluate_predictions(y_test, y_pred_lstm)
        print("LSTM final evaluation:", lstm_stats)

    except Exception as e:
        print("Skipping LSTM (PyTorch not available or error):", e)

    # -------- Show a few sample predictions --------
    print("\nSome sample predictions (true -> predicted) (first 10 of test):")
    for i in range(min(10, len(y_test))):
        true = y_test[i]
        pred = y_pred_rf[i]  # using RF as example
        dist = haversine_km(true[0], true[1], pred[0], pred[1])
        print(f"True (lat,lon,mag)={true}  Pred={np.round(pred,3)}  loc_err_km={dist:.2f}")

    print("\nDemo finished. Remember: educational only. See comments for next steps.")

if __name__ == "__main__":
    main_demo()
