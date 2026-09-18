import pandas as pd
from datetime import datetime


def compute_demand_scores(
    orders: pd.DataFrame,
    now: datetime = None,
    half_life_hours: float = 12.0,
) -> pd.DataFrame:

    if now is None:
        now = orders["timestamp"].max()

    df = orders.copy()
    age_hours = (now - df["timestamp"]).dt.total_seconds() / 3600.0
    decay = 0.5 ** (age_hours / half_life_hours)
    df["weighted_qty"] = df["qty"] * decay

    scores = (
        df.groupby(["dish_id", "dish_name"])["weighted_qty"]
        .sum()
        .reset_index()
        .rename(columns={"weighted_qty": "demand_score"})
        .sort_values("demand_score", ascending=False)
        .reset_index(drop=True)
    )
    return scores


def time_of_day_boost(dish_id: int, hour: int) -> float:

    breakfast_dishes = {5}   # e.g. Cold-Pressed Juice
    dinner_dishes = {1, 4}   # e.g. Butter Chicken, Pizza

    if 7 <= hour <= 10 and dish_id in breakfast_dishes:
        return 1.5
    if 19 <= hour <= 21 and dish_id in dinner_dishes:
        return 1.3
    return 1.0


def ranked_menu(orders: pd.DataFrame, now: datetime = None) -> pd.DataFrame:
 
    if now is None:
        now = orders["timestamp"].max()

    scores = compute_demand_scores(orders, now=now)
    scores["boost"] = scores["dish_id"].apply(
        lambda d: time_of_day_boost(d, now.hour)
    )
    scores["final_score"] = scores["demand_score"] * scores["boost"]
    return scores.sort_values("final_score", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    from backend.src.service.synthetic_data import generate_orders

    orders = generate_orders(days=7)
    ranking = ranked_menu(orders)
    print("Menu ranked by current demand:\n")
    print(ranking.to_string(index=False))
