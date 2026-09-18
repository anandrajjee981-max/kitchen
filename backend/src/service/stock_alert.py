import pandas as pd


def simple_low_stock_check(stock_df: pd.DataFrame) -> pd.DataFrame:
   
    low = stock_df[stock_df["current_stock"] <= stock_df["reorder_threshold"]].copy()
    low["alert"] = low.apply(
        lambda r: f"⚠️ {r['dish_name']} is low — {r['current_stock']} servings left "
                  f"(reorder at {r['reorder_threshold']})",
        axis=1,
    )
    return low[["dish_id", "dish_name", "current_stock", "alert"]]


def predictive_stockout_check(
    stock_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
) -> pd.DataFrame:
    
    results = []

    for dish_id, group in forecast_df.groupby("dish_id"):
        group = group.sort_values("timestamp")
        row = stock_df[stock_df["dish_id"] == dish_id]
        if row.empty:
            continue
        remaining = float(row["current_stock"].iloc[0])
        dish_name = row["dish_name"].iloc[0]

        stockout_time = None
        for _, r in group.iterrows():
            remaining -= r["predicted_qty"]
            if remaining <= 0:
                stockout_time = r["timestamp"]
                break

        if stockout_time is not None:
            hours_until = (stockout_time - pd.Timestamp.now()).total_seconds() / 3600
            results.append(
                {
                    "dish_id": dish_id,
                    "dish_name": dish_name,
                    "current_stock": row["current_stock"].iloc[0],
                    "predicted_stockout_time": stockout_time,
                    "alert": f"🔮 {dish_name} is projected to sell out in "
                             f"~{max(hours_until, 0):.1f}h based on current demand trend",
                }
            )

    return pd.DataFrame(results)


if __name__ == "__main__":
    from backend.src.service.synthetic_data import generate_orders
    from backend.src.service.demand_forecast import DemandForecaster, SimpleAverageForecaster

    orders = generate_orders(days=14)

    # Fake current stock levels for the demo
    stock_df = pd.DataFrame(
        [
            {"dish_id": 1, "dish_name": "Butter Chicken", "current_stock": 8, "reorder_threshold": 5},
            {"dish_id": 2, "dish_name": "Paneer Tikka", "current_stock": 20, "reorder_threshold": 5},
            {"dish_id": 3, "dish_name": "Veg Hakka Noodles", "current_stock": 3, "reorder_threshold": 5},
            {"dish_id": 4, "dish_name": "Margherita Pizza", "current_stock": 15, "reorder_threshold": 5},
            {"dish_id": 5, "dish_name": "Cold-Pressed Juice", "current_stock": 30, "reorder_threshold": 5},
            {"dish_id": 6, "dish_name": "Chicken Burger", "current_stock": 6, "reorder_threshold": 5},
        ]
    )

    print("=== Simple threshold alerts ===")
    print(simple_low_stock_check(stock_df).to_string(index=False))

    try:
        forecaster = DemandForecaster().fit(orders)
    except ImportError:
        forecaster = SimpleAverageForecaster().fit(orders)

    forecast = forecaster.predict_next_hours(pd.Timestamp.now().floor("h"), hours_ahead=8)

    print("\n=== Predictive stockout alerts ===")
    predictive = predictive_stockout_check(stock_df, forecast)
    if predictive.empty:
        print("No dishes projected to sell out in the forecast window.")
    else:
        print(predictive[["dish_name", "current_stock", "alert"]].to_string(index=False))
