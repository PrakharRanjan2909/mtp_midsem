import pulp
import numpy as np


def generate_demand_from_rul(rul_list, days=30, plates_per_day=50, holes_per_plate=100):
    """
    Converts RUL values (in holes) into daily demand array.
    """
    holes_per_day = plates_per_day * holes_per_plate
    demand = [0] * days

    for rul in rul_list:
        if rul <= 0:
            failure_day = 0
        else:
            failure_day = int(rul // holes_per_day)
        
        if failure_day < days:
            demand[failure_day] += 1
    return demand


# Define parameters
T = 30  # Number of time periods
C_h = 2  # Holding cost per tool per time period
C_f = 50  # Failure cost if stockout occurs
C_o = 20  # Ordering cost
M = 100  # Large constant for Big-M method


# Sample RULs in holes (you can load these from file or HMM output)
rul_predictions = [3000, 1000, 12000, 5000, 15000, 500, 2200, 3500]  # Example RULs

# Generate demand from RULs
D_t = generate_demand_from_rul(rul_predictions, days=T, plates_per_day=50, holes_per_plate=100)


# # Simulated demand from RUL predictions
# np.random.seed(42)
# D_t = np.random.randint(1, 5, size=T)  # Random demand between 1 and 4 tools per period

# Define LP model
model = pulp.LpProblem("Inventory_Optimization", pulp.LpMinimize)

# Decision Variables
s = pulp.LpVariable("Reorder_Point", lowBound=1, upBound=10, cat="Integer")
S = pulp.LpVariable("Max_Stock", lowBound=10, upBound=20, cat="Integer")
I = pulp.LpVariable.dicts("Inventory", range(T+1), lowBound=0, cat="Integer")
O = pulp.LpVariable.dicts("Order", range(T), lowBound=0, upBound=20, cat="Integer")
B = pulp.LpVariable.dicts("Stockout", range(T), lowBound=0, cat="Integer")

# NEW: Binary variable to indicate if inventory is below reorder point
X = pulp.LpVariable.dicts("Reorder_Indicator", range(T), cat="Binary")

# Objective: Minimize total cost
model += pulp.lpSum([C_h * I[t] + C_f * B[t] + C_o * O[t] for t in range(T)])

# Constraints
I[0] = S  # Initial inventory
for t in range(T):
    model += I[t+1] == I[t] + O[t] - D_t[t] - B[t]  # Inventory balance
    model += B[t] >= D_t[t] - I[t]  # Stockout condition
    model += O[t] <= S - I[t]  # Orders cannot exceed max inventory

    # Big-M Constraints for Reorder Condition
    model += I[t] - s <= M * (1 - X[t])  # If X[t] = 1, I[t] < s
    model += I[t] - s >= -M * X[t]  # If X[t] = 0, I[t] >= s

    # Order only when inventory is below s
    model += O[t] <= M * X[t]  # Ensures ordering only when X[t] = 1
    model += O[t] <= S - s  # Ensures max order quantity does not exceed S - s

# Solve the model
model.solve()

# Print results
print(f"Optimal (s, S): ({pulp.value(s)}, {pulp.value(S)}) with cost {pulp.value(model.objective)}")

# Print daily order policy
print("\nOptimal Order Policy:")
for t in range(T):
    print(f"Day {t+1}: Order {pulp.value(O[t])}, Inventory {pulp.value(I[t])}, Stockout {pulp.value(B[t])}")



import matplotlib.pyplot as plt
plt.bar(range(len(D_t)), D_t)
plt.title("Daily Tool Demand Based on RUL")
plt.xlabel("Day")
plt.ylabel("Tools Needed")
plt.grid(True)
plt.show()



