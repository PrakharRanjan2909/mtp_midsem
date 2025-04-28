from pulp import *

# Parameters
holes_per_plate = 100
tool_capacity = 10000  # max holes a tool can drill
tool_cost = 500
plate_cost = 200
tool_change_cost = 50
inventory_cost = 2
max_inventory = 20
T = 10  # planning horizon in days
M = 1000  # big-M for logical constraints

# LP Model
model = LpProblem("Joint_s_S_and_Daily_Plate_Optimization", LpMinimize)

# Decision Variables
s = LpVariable("s", lowBound=0, upBound=max_inventory, cat="Integer")
S = LpVariable("S", lowBound=0, upBound=max_inventory, cat="Integer")
plates_per_day = LpVariable.dicts("Plates", range(T), lowBound=1, upBound=100, cat="Integer")

O = LpVariable.dicts("Order", range(T), lowBound=0, upBound=max_inventory, cat="Integer")
I = LpVariable.dicts("Inventory", range(T+1), lowBound=0, upBound=max_inventory, cat="Integer")
X = LpVariable.dicts("ToolChange", range(T), cat="Binary")
P = LpVariable.dicts("PlateLoss", range(T), cat="Binary")

# Objective Function: Minimize total cost
model += lpSum([
    tool_cost * O[t] +
    inventory_cost * I[t] +
    tool_change_cost * X[t] +
    plate_cost * P[t]
    for t in range(T)
])

# Initial inventory
model += I[0] == 5

# Constraints
for t in range(T):
    # Inventory flow
    model += I[t+1] == I[t] + O[t] - X[t]

    # Reorder decision based on inventory level
    # If inventory is below s, order up to S
    model += O[t] <= max_inventory * X[t]
    # model += I[t] <= s + (S - s) * (1 - X[t])  # If X[t] = 1, I[t] < s
    model += I[t] <= s + M * (1 - X[t])


    # Plate loss if tool not changed and insufficient RUL
    holes_required = plates_per_day[t] * holes_per_plate
    model += holes_required <= tool_capacity + M * X[t]  # if X[t] = 1, no restriction
    model += holes_required >= tool_capacity - M * P[t]  # if over RUL and no change, plate fails

# Solve
model.solve()

# Output results
results = {
    "status": LpStatus[model.status],
    "total_cost": value(model.objective),
    "s": s.varValue,
    "S": S.varValue,
    "plates_per_day": [plates_per_day[t].varValue for t in range(T)],
    "orders": [O[t].varValue for t in range(T)],
    "tool_changes": [X[t].varValue for t in range(T)],
    "plate_losses": [P[t].varValue for t in range(T)],
    "inventory": [I[t].varValue for t in range(T+1)],
}

results
print("Optimization Results:")
for key, value in results.items():
    print(f"{key}: {value}")    

