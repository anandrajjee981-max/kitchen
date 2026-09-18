import pandas as pd
import numpy as np


def _build_hourly_features(orders: pd.DataFrame) -> pd.DataFrame:
   
    df = orders.copy()
    df["hour_bucket"] = df["timestamp"].dt.floor("h")

    hourly = (
        df.groupby(["dish_id", "dish_name", "hour_bucket"])["qty"]
        .sum()
        .reset_index()
    )
    hourly["hour_of_day"] = hourly["hour_bucket"].dt.hour
    hourly["day_of_week"] = hourly["hour_bucket"].dt.dayofweek
    hourly["is_weekend"] = (hourly["day_of_week"] >= 5).astype(int)
    return hourly


class DemandForecaster:
   

    def __init__(self):
        self.models = {}       # dish_id -> trained model
        self.dish_names = {}   # dish_id -> name

    def fit(self, orders: pd.DataFrame):
        from sklearn.ensemble import RandomForestRegressor

        hourly = _build_hourly_features(orders)

        for dish_id, group in hourly.groupby("dish_id"):
            self.dish_names[dish_id] = group["dish_name"].iloc[0]

            X = group[["hour_of_day", "day_of_week", "is_weekend"]]
            y = group["qty"]

            model = RandomForestRegressor(
                n_estimators=100, max_depth=6, random_state=42
            )
            model.fit(X, y)
            self.models[dish_id] = model

        return self

    def predict_next_hours(self, target_time: pd.Timestamp, hours_ahead: int = 6) -> pd.DataFrame:
       
        rows = []
        for h in range(hours_ahead):
            t = target_time + pd.Timedelta(hours=h)
            features = pd.DataFrame(
                [{
                    "hour_of_day": t.hour,
                    "day_of_week": t.dayofweek,
                    "is_weekend": int(t.dayofweek >= 5),
                }]
            )
            for dish_id, model in self.models.items():
                pred = max(0.0, float(model.predict(features)[0]))
                rows.append(
                    {
                        "timestamp": t,
                        "dish_id": dish_id,
                        "dish_name": self.dish_names[dish_id],
                        "predicted_qty": round(pred, 1),
                    }
                )
        return pd.DataFrame(rows)


class SimpleAverageForecaster:
   

    def __init__(self):
        self.lookup = None  # (dish_id, hour_of_day, is_weekend) -> avg qty
        self.dish_names = {}

    def fit(self, orders: pd.DataFrame):
        hourly = _build_hourly_features(orders)
        self.dish_names = (
            hourly[["dish_id", "dish_name"]]
            .drop_duplicates()
            .set_index("dish_id")["dish_name"]
            .to_dict()
        )
        self.lookup = (
            hourly.groupby(["dish_id", "hour_of_day", "is_weekend"])["qty"]
            .mean()
            .to_dict()
        )
        return self

    def predict_next_hours(self, target_time: pd.Timestamp, hours_ahead: int = 6) -> pd.DataFrame:
        rows = []
        for h in range(hours_ahead):
            t = target_time + pd.Timedelta(hours=h)
            is_weekend = int(t.dayofweek >= 5)
            for dish_id, name in self.dish_names.items():
                pred = self.lookup.get((dish_id, t.hour, is_weekend), 0.0)
                rows.append(
                    {
                        "timestamp": t,
                        "dish_id": dish_id,
                        "dish_name": name,
                        "predicted_qty": round(pred, 1),
                    }
                )
        return pd.DataFrame(rows)


if __name__ == "__main__":
    from backend.src.service.synthetic_data import generate_orders

    orders = generate_orders(days=14)

    try:
        forecaster = DemandForecaster().fit(orders)
        print("Using RandomForestRegressor forecaster\n")
    except ImportError:
        forecaster = SimpleAverageForecaster().fit(orders)
        print("sklearn not available — using SimpleAverageForecaster fallback\n")

    target = pd.Timestamp.now().floor("h")
    forecast = forecaster.predict_next_hours(target, hours_ahead=6)
    print(forecast.pivot(index="timestamp", columns="dish_name", values="predicted_qty"))
