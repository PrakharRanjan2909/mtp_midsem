import numpy as np
import pandas as pd

# Load RUL predictions from diagnostics phase
def load_rul_predictions(file_path):
    """
    Loads RUL predictions from a CSV file.
    """
    return pd.read_csv(file_path)

# Example usage
rul_data = load_rul_predictions("rul_predictions.csv")
print(rul_data.head())



import numpy as np
import scipy.stats as stats

#The current model assumes random failures based on RUL, but in reality, tool failures may follow specific distributions.
# Different distributions (e.g., Poisson, Weibull, or Exponential) can better model real-world failure patterns.

def simulate_tool_demand(rul_data, num_days=30, distribution="poisson"):
    """
    Simulates tool demand based on RUL predictions using different statistical distributions.
    
    Parameters:
        rul_data (DataFrame): RUL data
        num_days (int): Number of days to simulate
        distribution (str): Type of failure distribution ("poisson", "weibull", "exponential")
        
    Returns:
        List of daily tool demand.
    """
    daily_demand = []

    for _ in range(num_days):
        if distribution == "poisson":
            demand = np.random.poisson(lam=len(rul_data) / 10)  # Avg failure rate
        elif distribution == "weibull":
            shape = 1.5  # Shape parameter (adjustable)
            scale = np.mean(rul_data["RUL"]) / 2
            demand = np.random.weibull(shape) * scale
        elif distribution == "exponential":
            lambda_param = 1 / np.mean(rul_data["RUL"])
            demand = np.random.exponential(scale=1 / lambda_param)
        else:
            raise ValueError("Unsupported distribution")

        daily_demand.append(int(demand))  # Convert to integer count of failed tools

    return daily_demand

# Example usage
daily_demand_poisson = simulate_tool_demand(rul_data, distribution="poisson")
daily_demand_weibull = simulate_tool_demand(rul_data, distribution="weibull")
daily_demand_exponential = simulate_tool_demand(rul_data, distribution="exponential")

print("Poisson Demand:", daily_demand_poisson)
print("Weibull Demand:", daily_demand_weibull)
print("Exponential Demand:", daily_demand_exponential)


#Instead of relying only on statistical distributions, we can train an ML model to predict future tool failures based on historical usage, RUL, and external conditions.
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def train_inventory_forecast_model(rul_data):
    """
    Train a Random Forest model to predict daily tool failures based on RUL and past failure patterns.
    """
    # Prepare features and labels
    X = rul_data[["RUL"]]  # Features: RUL
    y = rul_data["Predicted Failure Probability"] * len(rul_data)  # Target: Failures

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error of Forecast Model: {mse}")

    return model

# Train and get the model
inventory_forecast_model = train_inventory_forecast_model(rul_data)

# Predict next day's failures
predicted_failures = inventory_forecast_model.predict([[5]])  # Example: RUL = 5 days
print(f"Predicted Failures for RUL=5: {predicted_failures}")


#Instead of relying on static predictions, use real-time sensor data to update inventory decisions dynamically.
# How?

# Continuously monitor force/torque readings from tools.
# If an unexpected spike or anomaly is detected, update failure probability and adjust inventory.

import time
import random

def real_time_inventory_update(inventory, reorder_point, max_stock, model):
    """
    Simulates real-time monitoring and dynamically updates inventory levels.
    
    Parameters:
        inventory (int): Current inventory level
        reorder_point (int): Minimum stock before reordering
        max_stock (int): Maximum allowable stock
        model: Trained ML model for demand forecasting
    """
    while True:
        # Simulate real-time failure detection
        real_time_failures = int(model.predict([[random.randint(1, 10)]])[0])

        # Update inventory
        inventory -= real_time_failures
        print(f"Current Inventory: {inventory}, Detected Failures: {real_time_failures}")

        # If inventory is below reorder point, place an order
        if inventory < reorder_point:
            order_quantity = max_stock - inventory
            inventory += order_quantity
            print(f"🔄 Reordered {order_quantity} new tools. New Inventory: {inventory}")

        time.sleep(1)  # Simulate real-time update every 5 seconds

# Example usage
real_time_inventory_update(10, reorder_point=5, max_stock=15, model=inventory_forecast_model)



# import matplotlib.pyplot as plt

# def inventory_simulation(daily_demand, s=5, S=15, initial_stock=10, lead_time=3, holding_cost=2, failure_cost=50):
#     """
#     Simulates inventory changes over time and calculates total cost.
#     """
#     inventory = initial_stock
#     orders = []
#     total_cost = 0

#     inventory_levels = []
    
#     for day, demand in enumerate(daily_demand):
#         # Check if we need to place an order
#         if inventory < s:
#             order_quantity = S - inventory
#             orders.append((day + lead_time, order_quantity))  # Order arrives after lead time
        
#         # Fulfill demand
#         inventory -= demand
#         if inventory < 0:
#             total_cost += abs(inventory) * failure_cost  # Stockout penalty
#             inventory = 0
        
#         # Receive new stock if orders arrive
#         for order_day, quantity in orders:
#             if order_day == day:
#                 inventory += quantity
        
#         inventory_levels.append(inventory)
#         total_cost += inventory * holding_cost  # Add holding cost

#     # Plot inventory levels
#     plt.figure(figsize=(10,5))
#     plt.plot(range(len(daily_demand)), inventory_levels, label="Inventory Level")
#     plt.axhline(y=s, color='r', linestyle='--', label="Reorder Point (s)")
#     plt.axhline(y=S, color='g', linestyle='--', label="Max Inventory (S)")
#     plt.xlabel("Days")
#     plt.ylabel("Inventory Level")
#     plt.legend()
#     plt.title("Tool Inventory Over Time")
#     plt.show()

#     return total_cost

# # Run the simulation
# total_cost = inventory_simulation(daily_demand)
# print("Total Cost:", total_cost)



# import itertools

# def optimize_inventory(daily_demand):
#     """
#     Finds the best (s, S) policy that minimizes total cost.
#     """
#     best_cost = float('inf')
#     best_s, best_S = None, None

#     for s, S in itertools.product(range(1, 10), range(10, 20)):  # Test different values
#         cost = inventory_simulation(daily_demand, s, S)
#         if cost < best_cost:
#             best_cost = cost
#             best_s, best_S = s, S

#     return best_s, best_S, best_cost

# # Run optimization
# optimal_s, optimal_S, min_cost = optimize_inventory(daily_demand)
# print(f"Optimal (s, S): ({optimal_s}, {optimal_S}) with cost {min_cost}")

