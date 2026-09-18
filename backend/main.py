from fastapi import FastAPI
import pandas as pd
from src.service.demand_sorting import ranked_menu
from src.service.demand_forecast import DemandForecaster, SimpleAverageForecaster
from src.service.stock_alert import simple_low_stock_check, predictive_stockout_check
from src.service.synthetic_data import generate_orders


app = FastAPI()





# --- Temporary: replace this with a real DB call once orders/stock tables exist ---

def get_orders_from_db():
    # TODO: swap this for a real query once the orders table exists
    return generate_orders(days=14)

def get_stock_from_db():
    # TODO: swap this for a real query once the stock table exists
    return pd.DataFrame([
        {"dish_id": 1, "dish_name": "Butter Chicken", "current_stock": 8, "reorder_threshold": 5},
        {"dish_id": 2, "dish_name": "Paneer Tikka", "current_stock": 20, "reorder_threshold": 5},
        {"dish_id": 3, "dish_name": "Veg Hakka Noodles", "current_stock": 3, "reorder_threshold": 5},
        {"dish_id": 4, "dish_name": "Margherita Pizza", "current_stock": 15, "reorder_threshold": 5},
        {"dish_id": 5, "dish_name": "Cold-Pressed Juice", "current_stock": 30, "reorder_threshold": 5},
        {"dish_id": 6, "dish_name": "Chicken Burger", "current_stock": 6, "reorder_threshold": 5},
    ])


# --- 1. Demand-ranked menu ---
@app.get("/api/menu/ranked")
def get_ranked_menu():
    orders = get_orders_from_db()
    ranking = ranked_menu(orders)
    return ranking.to_dict(orient="records")


# --- 2. Demand forecast (next 6 hours) ---
@app.get("/api/forecast")
def get_forecast():
    orders = get_orders_from_db()
    try:
        forecaster = DemandForecaster().fit(orders)
    except ImportError:
        forecaster = SimpleAverageForecaster().fit(orders)

    now = pd.Timestamp.now().floor("h")
    forecast = forecaster.predict_next_hours(now, hours_ahead=6)
    # convert timestamps to strings so they're JSON-serializable
    forecast["timestamp"] = forecast["timestamp"].astype(str)
    return forecast.to_dict(orient="records")


# --- 3. Stock alerts (simple + predictive) ---
@app.get("/api/alerts")
def get_alerts():
    orders = get_orders_from_db()
    stock = get_stock_from_db()

    simple_alerts = simple_low_stock_check(stock)

    try:
        forecaster = DemandForecaster().fit(orders)
    except ImportError:
        forecaster = SimpleAverageForecaster().fit(orders)

    now = pd.Timestamp.now().floor("h")
    forecast = forecaster.predict_next_hours(now, hours_ahead=8)
    predictive_alerts = predictive_stockout_check(stock, forecast)

    return {
        "simple_alerts": simple_alerts["alert"].tolist() if not simple_alerts.empty else [],
        "predictive_alerts": predictive_alerts["alert"].tolist() if not predictive_alerts.empty else [],
    }

