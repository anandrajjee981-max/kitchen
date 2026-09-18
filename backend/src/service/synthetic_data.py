import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

DISHES = [
    {"dish_id": 1, "name": "Butter Chicken", "base_popularity": 1.4},
    {"dish_id": 2, "name": "Paneer Tikka", "base_popularity": 1.1},
    {"dish_id": 3, "name": "Veg Hakka Noodles", "base_popularity": 0.9},
    {"dish_id": 4, "name": "Margherita Pizza", "base_popularity": 1.3},
    {"dish_id": 5, "name": "Cold-Pressed Juice", "base_popularity": 0.6},
    {"dish_id": 6, "name": "Chicken Burger", "base_popularity": 1.0},
]


def _hourly_demand_multiplier(hour: int) -> float:
   
    if 12 <= hour <= 14:
        return 3.0
    if 19 <= hour <= 21:
        return 3.5
    if 7 <= hour <= 9:
        return 1.2
    if 0 <= hour <= 5:
        return 0.05
    return 0.8


def _weekday_multiplier(day_of_week: int) -> float:
   
    return 1.4 if day_of_week >= 5 else 1.0


def generate_orders(days: int = 14, seed: int = 42) -> pd.DataFrame:
   
    rng = np.random.default_rng(seed)
    random.seed(seed)

    start = datetime.now() - timedelta(days=days)
    rows = []

    for day in range(days):
        current_day = start + timedelta(days=day)
        for hour in range(24):
            ts_base = current_day.replace(hour=hour, minute=0, second=0, microsecond=0)
            hour_mult = _hourly_demand_multiplier(hour)
            day_mult = _weekday_multiplier(current_day.weekday())

            for dish in DISHES:
                expected = dish["base_popularity"] * hour_mult * day_mult
                # Poisson noise around the expected order count for this hour
                qty = rng.poisson(lam=max(expected, 0.01))
                if qty > 0:
                    # spread orders randomly within the hour
                    ts = ts_base + timedelta(minutes=random.randint(0, 59))
                    rows.append(
                        {
                            "timestamp": ts,
                            "dish_id": dish["dish_id"],
                            "dish_name": dish["name"],
                            "qty": qty,
                        }
                    )

    df = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
    return df


if __name__ == "__main__":
    orders = generate_orders(days=7)
    print(orders.head(15))
    print(f"\nTotal synthetic order events: {len(orders)}")
    print(f"Date range: {orders.timestamp.min()} -> {orders.timestamp.max()}")
