import numpy as np
import pandas as pd



# (s, S) inventory policy, which defines:

# 𝑠
# s: Reorder point (if inventory drops below this, order new tools).
# 𝑆
# S: Maximum inventory level (how many tools we keep in stock).
# ℎ
# h: Holding cost per tool per time period.
# 𝑐
# 𝑓
# c 
# f
# ​
#  : Cost of tool failure if no replacement is available.
# 𝜆
# λ: Tool failure rate based on RUL predictions.
# 𝐿
# L: Lead time (how long new orders take to arrive).
# Objective Function (Minimize Total Cost):

# min
# ⁡
# ℎ
# ×
# 𝑆
# +
# 𝑐
# 𝑓
# ×
# 𝑃
# (
# stockout
# )
# minh×S+c 
# f
# ​
#  ×P(stockout)
# where 
# 𝑃
# (
# stockout
# )
# P(stockout) is the probability that tools fail before new ones arrive.

# optimize tool inventory levels, ensuring that downtime is minimized while avoiding excessive stock costs.

#Minimize total cost by balancing tool storage costs and failure risk due to stockouts, using RUL predictions to guide inventory decisions.

# 

# Load RUL predictions from diagnostics phase
def load_rul_predictions(file_path):
    """
    Loads RUL predictions from a CSV file.
    """
    return pd.read_csv(file_path)

# Example usage
rul_data = load_rul_predictions("rul_predictions.csv")
print(rul_data.head())



# import numpy as np

#randomly selects tools to fail based on their RUL distribution.

def simulate_tool_demand(rul_data, num_days=30):
    """
    Simulates tool demand based on RUL predictions.
    Each tool has a probability of failing within the given period.
    """
    daily_demand = []
    
    for _ in range(num_days):
        failed_tools = rul_data[rul_data["RUL"] <= np.random.randint(1, 5)]  # Randomize failures
        daily_demand.append(len(failed_tools))
    
    return daily_demand

# Example usage
daily_demand = simulate_tool_demand(rul_data)
print("Simulated Daily Demand:", daily_demand)


#If inventory drops below s, we order enough to restock to 𝑆

import matplotlib.pyplot as plt

def inventory_simulation(daily_demand, s=5, S=15, initial_stock=10, lead_time=3, holding_cost=2, failure_cost=50):
    """
    Simulates inventory changes over time and calculates total cost.
    """
    inventory = initial_stock
    orders = []
    total_cost = 0

    inventory_levels = []
    
    for day, demand in enumerate(daily_demand):
        # Check if we need to place an order
        if inventory < s:
            order_quantity = S - inventory
            orders.append((day + lead_time, order_quantity))  # Order arrives after lead time
        
        # Fulfill demand
        inventory -= demand
        if inventory < 0:
            total_cost += abs(inventory) * failure_cost  # Stockout penalty
            inventory = 0
        
        # Receive new stock if orders arrive
        for order_day, quantity in orders:
            if order_day == day:
                inventory += quantity
        
        inventory_levels.append(inventory)
        total_cost += inventory * holding_cost  # Add holding cost

    # Plot inventory levels
    plt.figure(figsize=(10,5))
    plt.plot(range(len(daily_demand)), inventory_levels, label="Inventory Level")
    plt.axhline(y=s, color='r', linestyle='--', label="Reorder Point (s)")
    plt.axhline(y=S, color='g', linestyle='--', label="Max Inventory (S)")
    plt.xlabel("Days")
    plt.ylabel("Inventory Level")
    plt.legend()
    plt.title("Tool Inventory Over Time")
    plt.show()

    return total_cost

# Run the simulation
total_cost = inventory_simulation(daily_demand)
print("Total Cost:", total_cost)

import itertools

def optimize_inventory(daily_demand):
    """
    Finds the best (s, S) policy that minimizes total cost.
    """
    best_cost = float('inf')
    best_s, best_S = None, None

    for s, S in itertools.product(range(1, 10), range(10, 20)):  # Test different values
        cost = inventory_simulation(daily_demand, s, S)
        if cost < best_cost:
            best_cost = cost
            best_s, best_S = s, S

    return best_s, best_S, best_cost

# Run optimization
optimal_s, optimal_S, min_cost = optimize_inventory(daily_demand)
print(f"Optimal (s, S): ({optimal_s}, {optimal_S}) with cost {min_cost}")

