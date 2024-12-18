from pulp import LpProblem, LpMinimize, LpVariable, lpSum, LpStatus, PULP_CBC_CMD
import numpy as np
import random


M = 1e10

two_phase = False

"""# Create the model
model = LpProblem(name="integer-programming-example", sense=LpMaximize)

# Define integer variables
x = LpVariable(name="x", lowBound=0, cat=LpInteger)
y = LpVariable(name="y", lowBound=0, cat=LpInteger)

# Add the objective function
model += 3 * x + 4 * y, "Objective"

# Add constraints
model += (2 * x + y <= 10, "Constraint_1")
model += (x + 3 * y <= 12, "Constraint_2")

# Solve the problem
status = model.solve()

# Print the solution status
print(f"Status: {LpStatus[model.status]}")

# Print the values of variables
print(f"x = {x.value()}")
print(f"y = {y.value()}")

# Print the optimal objective value
print(f"Objective Value: {model.objective.value()}")"""


def solver(cost_matrix, supply_arr, demand_arr):
    assert len(cost_matrix) == len(supply_arr), "Number of rows in cost_matrix must match supply_arr"
    assert len(cost_matrix[0]) == len(demand_arr), "Number of columns in cost_matrix must match demand_arr"
    model = LpProblem(name="transportation_problem", sense=LpMinimize)
    
    # Define decision variables
    x_matrix = [[LpVariable(name=f"x{i}_{j}", lowBound=0) for j in range(len(cost_matrix[i]))] for i in range(len(cost_matrix))]

    # Define the objective function
    objective = lpSum(x_matrix[i][j] * cost_matrix[i][j] for i in range(len(cost_matrix)) for j in range(len(cost_matrix[i])))
    model += objective, "Objective"
    
    # Add supply constraints
    for i in range(len(cost_matrix)):
        constraint_name = f"Supply_Constraint_{i}"
        constraint = lpSum(x_matrix[i]) == supply_arr[i]
        model += constraint, constraint_name
    
    # Add demand constraints
    for j in range(len(cost_matrix[0])):
        constraint_name = f"Demand_Constraint_{j}"
        constraint = lpSum([x_matrix[i][j] for i in range(len(cost_matrix))]) == demand_arr[j]
        model += constraint, constraint_name
    
    # Solve the problem
    solver = PULP_CBC_CMD(msg=False)
    status = model.solve(solver)
    
    # Print the optimal objective value
    print(f"Objective Value: {model.objective.value()}")
    
    # Print the values of the decision variables
    """for i in range(len(cost_matrix)):
        for j in range(len(cost_matrix[i])):
            print(f"x{i}{j} = {x_matrix[i][j].value()}")
"""
    return model.objective.value()


def random_instance(supply_nodes, demand_nodes, maximum_cost, maximum_amount):
    supply_arr = []
    demand_arr = []
    if supply_nodes < demand_nodes:
        for i in range(supply_nodes):
            supply_arr.append(random.randint(1, maximum_amount))
        for i in range(demand_nodes):
            demand_arr.append(random.randint(1, maximum_amount*supply_nodes//demand_nodes))
    else:
        for i in range(demand_nodes):
            demand_arr.append(random.randint(1, maximum_amount))
        for i in range(supply_nodes):
            supply_arr.append(random.randint(1, maximum_amount*demand_nodes//supply_nodes))

    if sum(supply_arr) != sum(demand_arr):
        if sum(supply_arr) < sum(demand_arr):
            excess = sum(demand_arr) - sum(supply_arr)
            # distribute the excess supply randomly among the supply nodes not violating the maximum amount
            for i in range(len(supply_arr)):
                if excess == 0:
                    break
                if supply_arr[i] + excess <= maximum_amount:
                    supply_arr[i] += excess
                    excess = 0
                else:
                    excess -= maximum_amount - supply_arr[i]
                    supply_arr[i] = maximum_amount
        else:
            excess = sum(supply_arr) - sum(demand_arr)
            # distribute the excess demand randomly among the demand nodes not violating the maximum amount
            for i in range(len(demand_arr)):
                if excess == 0:
                    break
                if demand_arr[i] + excess <= maximum_amount:
                    demand_arr[i] += excess
                    excess = 0
                else:
                    excess -= maximum_amount - demand_arr[i]
                    demand_arr[i] = maximum_amount
    random.shuffle(supply_arr)
    random.shuffle(demand_arr)
    cost_matrix = []
    for i in range(supply_nodes):
        sub_cost_matrix = []
        for j in range(demand_nodes):
            sub_cost_matrix.append(random.randint(1, maximum_cost))
        cost_matrix.append(sub_cost_matrix)


    return cost_matrix, supply_arr, demand_arr


def check_artificial_variable(c, two_phase=False):
    c_artificial = np.zeros(len(c))
    for i in range(len(c)):
        if c[i] == -M or c[i] == M:
            two_phase = True
            c_artificial[i] = -1
    return two_phase, c_artificial

def handle_singular_matrix(matrix):
    try:
        inv_matrix = np.linalg.inv(matrix)
    except np.linalg.LinAlgError:
        print("Singular matrix detected, using pseudo-inverse.")
        inv_matrix = np.linalg.pinv(matrix)
    return inv_matrix


def elementary_row_operations(matrix, d, p):
    m, n = matrix.shape
    for i in range(m):
        if i != p:
            matrix[i] = matrix[i] - d[i] / d[p] * matrix[p]
    matrix[p] = matrix[p] / d[p]
    return matrix

def revised_simplex_method(c, A, b, problem="Max"):
    # c is the cost vector
    # A is the constraint matrix
    # b is the right-hand side vector
    # x is the solution vector
    # z is the objective value
    # m is the number of constraints
    # n is the number of variables
    # B is the set of basic variables
    # N is the set of nonbasic variables
    # r is the vector of right-hand side values
    # q is the entering variable
    # p is the leaving variable
    # i is the index of the leaving variable
    # j is the index of the entering variable
    
    two_phase = False
    check_artificial_variable(c)
    if two_phase:
        c_two_phase = c
        c, two_phase = check_artificial_variable(c)

    

    m, n = A.shape
    B = np.arange(n-m, n)
    N = np.arange(0, n-m)
    base_matrix = A[:, B]
    nonbase_matrix = A[:, N]
    base_cost = c[B]
    nonbase_cost = c[N]
    inv_base = np.linalg.inv(base_matrix)
    optimality_check = np.dot(np.transpose(base_cost), inv_base)
    optimality_check = np.dot(optimality_check, nonbase_matrix) - np.transpose(nonbase_cost)
    while np.any(optimality_check < 0):
        j = np.argmin(optimality_check)
        d = np.dot(inv_base, nonbase_matrix[:, j])

        rhs = np.dot(inv_base, b)
        if np.all(d <= 0):
            return "Unbounded"
        else:
            theta = 1e10
            for i in range(m):
                if d[i] > 0:
                    theta = min(theta, rhs[i] / d[i])
                    if(theta == rhs[i] / d[i]):
                        p = i
            q = B[p]
            B[p] = j
            N[j] = q
            base_matrix = A[:, B]
            nonbase_matrix = A[:, N]
            base_cost = c[B]
            nonbase_cost = c[N]
            inv_base = elementary_row_operations(inv_base, d, p)
            optimality_check = np.dot(base_cost, inv_base)
            optimality_check = np.dot(optimality_check, nonbase_matrix) - nonbase_cost

    optimal_value = np.dot(np.transpose(base_cost), inv_base)
    optimal_value = np.dot(optimal_value, b)

    rhs = np.dot(inv_base, b)
    x = np.zeros(n)
    x[B] = rhs
    #concatenate two matrices
    if two_phase:
        identity_matrix = np.identity(len(B))
        non_basic = np.dot(inv_base, nonbase_matrix)
        indices = []
        for i in range(len(c)):
            if c[i] == -1:
                indices.append(i)
        m = m - len(indices)
        indices.sort(reverse=True)
        for i in indices:
            c_two_phase = np.delete(c_two_phase, i)
            A = np.delete(A, i, 0)
            N = np.delete(N, i)
        A = np.zeros((m, n))
        for i in range(m):
            A[i, B] = identity_matrix[i]
            A[i, N] = non_basic[i]
        b = rhs
        c = c_two_phase
        revised_simplex_method(c, A, b)

    return optimal_value, x



"""def test_revised_simplex_method():
    c = np.array([1, 1, 0, 0, -M])
    A = np.array([[2, 5, 0, 1, 0],
                  [1, 1, -1, 0, 1]
                  ])
    b = np.array([6, 2])
    optimal_value, x = revised_simplex_method(c, A, b)
    print(optimal_value)
    print(x)

    
test_revised_simplex_method()
"""
def convert_random_instance_to_lp(cost_matrix, supply_arr, demand_arr):
    # cost_matrix is the cost matrix
    # supply_arr is the supply array
    # demand_arr is the demand array
    # x is the solution vector
    # z is the objective value
    # m is the number of constraints
    # n is the number of variables
    # B is the set of basic variables
    # N is the set of nonbasic variables
    # r is the vector of right-hand side values
    # q is the entering variable
    # p is the leaving variable
    # i is the index of the leaving variable
    # j is the index of the entering variable
    m = len(supply_arr) + len(demand_arr)
    n = len(supply_arr) * len(demand_arr)
    A = np.zeros((m,m+n))
    b = np.zeros(m)
    c = np.zeros(m+n)
    c.fill(-M)
    for i in range(len(supply_arr)):
        for j in range(len(demand_arr)):
            A[i, i*len(demand_arr) + j] = 1
            A[j + len(supply_arr), i*len(demand_arr) + j] = 1
            c[i*len(demand_arr) + j] = -1 * cost_matrix[i][j]
    for i in range(m):
        A[i, n + i] = 1


    b[:len(supply_arr)] = supply_arr
    b[len(supply_arr):] = demand_arr
    return c, A, b


def test_convert_random_instance_to_lp():
    cost_matrix = np.array([[1, 2, 3],
                             [4, 5, 6]
                             ])
    supply_arr = np.array([3, 5])
    demand_arr = np.array([4, 2, 2])
    c, A, b = convert_random_instance_to_lp(cost_matrix, supply_arr, demand_arr)
    print(c)
    print(A)
    print(b)


cost_matrix, supply_arr, demand_arr = random_instance(120, 120, 10000, 1000)
print(solver(cost_matrix, supply_arr, demand_arr))
print(revised_simplex_method(*convert_random_instance_to_lp(cost_matrix, supply_arr, demand_arr)))
