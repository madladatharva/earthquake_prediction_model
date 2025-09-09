import requests
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader, TensorDataset

# -------------------------------
# 1. Fetch Earthquake Data
# -------------------------------
def fetch_earthquakes(start, end, min_mag=3.0):
    print(f"Fetching USGS data {start} to {end}...")
    all_data = []
    start_year = int(start[:4])
    end_year = int(end[:4])

    for year in range(start_year, end_year + 1):
        url = (
            f"https://earthquake.usgs.gov/fdsnws/event/1/query?"
            f"format=geojson&starttime={year}-01-01&endtime={year}-12-31&minmagnitude={min_mag}"
        )
        print(f"Fetching {year} ...")
        try:
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            data = r.json()
            for feat in data.get("features", []):
                props = feat["properties"]
                geom = feat["geometry"]["coordinates"]
                all_data.append({
                    "time": props["time"],
                    "lat": geom[1],
                    "lon": geom[0],
                    "depth": geom[2],
                    "mag": props["mag"]
                })
        except Exception as e:
            print(f"⚠️ Skipping year {year}, error: {e}")
            continue

    df = pd.DataFrame(all_data)
    df["time"] = pd.to_datetime(df["time"], unit="ms")
    return df.dropna()

# -------------------------------
# 2. Define Model
# -------------------------------
class EarthquakeModel(nn.Module):
    def __init__(self):
        super(EarthquakeModel, self).__init__()
        self.fc1 = nn.Linear(3, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 3)  # predict lat, lon, mag
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

# -------------------------------
# 3. Main
# -------------------------------
if __name__ == "__main__":
    df = fetch_earthquakes("2010-01-01", "2025-09-09", min_mag=3.0)
    print(f"Total events fetched: {len(df)}")

    # ⚡ Limit to 10k random samples
    df = df.sample(10000, random_state=42)

    # ✅ Force numeric types and drop invalid rows
    df = df.astype({"lat": "float32", "lon": "float32", "depth": "float32", "mag": "float32"})
    df = df.dropna()

    print(df.head())

    # Features: (lat, lon, depth), Targets: (lat, lon, mag)
    X = df[["lat", "lon", "depth"]].values
    y = df[["lat", "lon", "mag"]].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor),
                              batch_size=32, shuffle=True)

    model = EarthquakeModel()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # -------------------------------
    # Training
    # -------------------------------
    num_epochs = 5  # ⚡ faster
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            outputs = model(xb)
            loss = criterion(outputs, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}")

    # -------------------------------
    # Evaluation
    # -------------------------------
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test_tensor)
        test_loss = criterion(y_pred, y_test_tensor).item()
        r2 = r2_score(y_test, y_pred.numpy())

    print("\n📊 Evaluation Results:")
    print(f"Test MSE Loss: {test_loss:.4f}")
    print(f"R² Score (accuracy of fit): {r2:.4f}")
