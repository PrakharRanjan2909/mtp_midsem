# import pulp
# import numpy as np

# def generate_demand_from_rul(rul_list, threshold, total_life, holes_per_day, days):
#     """
#     Converts RULs (in holes) into demand per day based on replacement threshold.
#     If a tool has remaining life < threshold %, it is replaced on day 0.
#     """
#     demand = [0] * days
#     for rul in rul_list:
#         life_percent_left = (rul / total_life) * 100
#         if life_percent_left < threshold:
#             failure_day = 0  # Replace immediately
#         else:
#             failure_day = int(rul // holes_per_day)

#         if failure_day < days:
#             demand[failure_day] += 1
#     return demand

# def optimize_inventory_with_rul_policy(
#     rul_list,
#     total_life=5000,
#     days=30,
#     plates_per_day=50,
#     holes_per_plate=100,
#     thresholds=[3, 5, 7, 10, 15],
#     holding_cost=2,
#     ordering_cost=20,
#     tool_change_cost=30,
#     tool_break_cost=150
# ):
#     """
#     Runs ILP optimization for multiple RUL thresholds and finds the most cost-effective one.
#     """
#     holes_per_day = plates_per_day * holes_per_plate
#     best_threshold = None
#     best_cost = float('inf')
#     cost_results = {}

#     for threshold in thresholds:
#         demand = generate_demand_from_rul(rul_list, threshold, total_life, holes_per_day, days)

#         # ILP Model Setup
#         model = pulp.LpProblem(f"Inventory_Optimization_Threshold_{threshold}", pulp.LpMinimize)
#         I = pulp.LpVariable.dicts("Inventory", range(days + 1), lowBound=0, cat="Integer")
#         O = pulp.LpVariable.dicts("Order", range(days), lowBound=0, upBound=100, cat="Integer")
#         B = pulp.LpVariable.dicts("Break", range(days), lowBound=0, cat="Integer")
#         X = pulp.LpVariable.dicts("Reorder_Flag", range(days), cat="Binary")
#         s = pulp.LpVariable("Reorder_Point", lowBound=1, upBound=10, cat="Integer")
#         S = pulp.LpVariable("Max_Inventory", lowBound=10, upBound=30, cat="Integer")
#         M = 100  # Big-M

#         # Objective: Minimize total cost
#         model += pulp.lpSum([
#             holding_cost * I[t] +
#             ordering_cost * O[t] +
#             tool_break_cost * B[t]
#             for t in range(days)
#         ]) + tool_change_cost * sum(demand)  # Adding proactive tool replacement cost

#         # Constraints
#         I[0] = S  # Initial inventory
#         for t in range(days):
#             model += I[t+1] == I[t] + O[t] - demand[t] - B[t]
#             model += B[t] >= demand[t] - I[t]
#             model += O[t] <= S - I[t]
#             model += I[t] - s <= M * (1 - X[t])
#             model += I[t] - s >= -M * X[t]
#             model += O[t] <= M * X[t]
#             model += O[t] <= S - s

#         model.solve()
#         total_cost = pulp.value(model.objective)
#         cost_results[threshold] = total_cost

#         if total_cost < best_cost:
#             best_cost = total_cost
#             best_threshold = threshold

#     return best_threshold, best_cost, cost_results

# rul_list = [3000, 2000, 1500, 500, 1000, 800, 2500, 100]  # RULs in holes
# best_threshold, best_cost, all_costs = optimize_inventory_with_rul_policy(
#     rul_list,
#     total_life=5000,
#     thresholds=[3, 5, 7, 10, 15],
#     tool_change_cost=30,
#     tool_break_cost=150
# )

# print(f"Best Threshold: {best_threshold}%")
# print(f"Minimum Total Cost: {best_cost}")
# print("All Threshold Costs:", all_costs)

import pulp

def generate_demand_from_rul(rul_list, threshold, total_life, holes_per_day, days):
    demand = [0] * days
    for rul in rul_list:
        life_percent_left = (rul / total_life) * 100
        if life_percent_left < threshold:
            failure_day = 0
        else:
            failure_day = int(rul // holes_per_day)
        if failure_day < days:
            demand[failure_day] += 1
    return demand

def optimize_inventory_with_threshold_and_sS(
    rul_list,
    total_life=5000,
    days=30,
    plates_per_day=50,
    holes_per_plate=100,
    thresholds=[3, 5, 7, 10],
    s_range=range(1, 11),
    S_range=range(10, 21),
    holding_cost=2,
    ordering_cost=20,
    tool_change_cost=30,
    tool_break_cost=150
):
    holes_per_day = plates_per_day * holes_per_plate
    best_result = {
        'threshold': None,
        's': None,
        'S': None,
        'total_cost': float('inf')
    }
    all_results = {}

    for threshold in thresholds:
        demand = generate_demand_from_rul(rul_list, threshold, total_life, holes_per_day, days)
        best_cost_for_threshold = float('inf')
        best_s, best_S = None, None

        for s_val in s_range:
            for S_val in S_range:
                if s_val >= S_val:
                    continue  # Skip invalid combinations

                model = pulp.LpProblem("Inventory_Opt", pulp.LpMinimize)
                M = 1000

                I = pulp.LpVariable.dicts("Inventory", range(days + 1), lowBound=0, cat="Integer")
                O = pulp.LpVariable.dicts("Order", range(days), lowBound=0, upBound=100, cat="Integer")
                B = pulp.LpVariable.dicts("Break", range(days), lowBound=0, cat="Integer")
                X = pulp.LpVariable.dicts("Reorder_Flag", range(days), cat="Binary")

                # Objective function
                model += pulp.lpSum([
                    holding_cost * I[t] +
                    ordering_cost * O[t] +
                    tool_break_cost * B[t]
                    for t in range(days)
                ]) + tool_change_cost * sum(demand)

                # Initial inventory
                model += I[0] == S_val

                for t in range(days):
                    model += I[t+1] == I[t] + O[t] - demand[t] - B[t]
                    model += B[t] >= demand[t] - I[t]
                    model += O[t] <= S_val - I[t]

                    model += I[t] - s_val <= M * (1 - X[t])
                    model += I[t] - s_val >= -M * X[t]
                    model += O[t] <= M * X[t]
                    model += O[t] <= S_val - s_val

                model.solve()
                total_cost = pulp.value(model.objective)

                if total_cost < best_cost_for_threshold:
                    best_cost_for_threshold = total_cost
                    best_s, best_S = s_val, S_val

                all_results[(threshold, s_val, S_val)] = total_cost

        # Update best overall result
        if best_cost_for_threshold < best_result['total_cost']:
            best_result['threshold'] = threshold
            best_result['s'] = best_s
            best_result['S'] = best_S
            best_result['total_cost'] = best_cost_for_threshold

    return best_result, all_results

rul_list = [3000, 2000, 1500, 500, 1000, 800, 2500, 100]

best, all_combinations = optimize_inventory_with_threshold_and_sS(
    rul_list,
    thresholds=[5, 7, 10],
    s_range=range(2, 8),
    S_range=range(12, 20)
)

print("🔹 Best Overall Configuration:")
print(f"  Threshold: {best['threshold']}%")
print(f"  Reorder Point (s): {best['s']}")
print(f"  Max Inventory (S): {best['S']}")
print(f"  Total Cost: {best['total_cost']}")



import matplotlib.pyplot as plt

def plot_cost_vs_rul_threshold(cost_results):
    thresholds = sorted(cost_results.keys())
    costs = [cost_results[t] for t in thresholds]

    plt.figure(figsize=(8, 5))
    plt.plot(thresholds, costs, marker='o', linestyle='-', color='teal')
    plt.title('Total Cost vs RUL Replacement Threshold')
    plt.xlabel('RUL Threshold (%)')
    plt.ylabel('Total Cost')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_cost_bar_chart(cost_results):
    thresholds = sorted(cost_results.keys())
    costs = [cost_results[t] for t in thresholds]

    plt.figure(figsize=(8, 5))
    plt.bar([str(t) + '%' for t in thresholds], costs, color='skyblue', edgecolor='black')
    plt.title('Cost Comparison Across RUL Thresholds')
    plt.xlabel('RUL Threshold')
    plt.ylabel('Total Cost')
    plt.tight_layout()
    plt.show()


from mpl_toolkits.mplot3d import Axes3D

def plot_3d_cost_surface(all_results):
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt

    thresholds, s_vals, S_vals, costs = [], [], [], []

    for (th, s, S), cost in all_results.items():
        thresholds.append(th)
        s_vals.append(s)
        S_vals.append(S)
        costs.append(cost)

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(thresholds, s_vals, S_vals, c=costs, cmap='viridis')
    ax.set_xlabel('Threshold (%)')
    ax.set_ylabel('Reorder Point (s)')
    ax.set_zlabel('Max Inventory (S)')
    ax.set_title('Cost vs Threshold vs (s, S)')
    plt.tight_layout()
    plt.show()

plot_cost_vs_rul_threshold(all_combinations)
plot_cost_bar_chart(all_combinations)
plot_3d_cost_surface(all_combinations)
