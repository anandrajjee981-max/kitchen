from fastapi import FastAPI

app = FastAPI()

app.get("/api/orders")
def get_orders():
    return {"orders": "orders_placed"}

