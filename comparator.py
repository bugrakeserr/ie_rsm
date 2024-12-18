import re

def compare_solver_rms_results(solver_obj_value, rms_obj_value, solver_time, rms_time, tolerance=1e-5):
    """
    Compares the objective values and time taken by Solver and Revised Simplex Method (RMS).

    :param solver_obj_value: Objective value from the solver.
    :param rms_obj_value: Objective value from the Revised Simplex Method.
    :param solver_time: Time taken by the solver.
    :param rms_time: Time taken by the Revised Simplex Method.
    :param tolerance: Allowed difference between objective values for them to be considered the same.
    :return: A string comparison result.
    """
    # Compare objective values within the given tolerance
    if abs(solver_obj_value - rms_obj_value) < tolerance:
        objective_comparison = "Objective values match."
    else:
        objective_comparison = f"Objective values differ. Solver: {solver_obj_value}, RMS: {rms_obj_value}"

    # Compare time taken
    if rms_time < solver_time:
        time_comparison = f"RMS is faster by {solver_time - rms_time:.6f} seconds."
    elif rms_time > solver_time:
        time_comparison = f"Solver is faster by {rms_time - solver_time:.6f} seconds."
    else:
        time_comparison = "Both took the same time."

    # Return the final comparison message
    return f"{objective_comparison}\n{time_comparison}"

def extract_values_from_file(file_path):
    """
    Extracts Solver and RMS objective values and times from the output file.

    :param file_path: Path to the output file.
    :return: A list of tuples containing the objective values and times.
    """
    results = []
    
    with open(file_path, 'r') as file:
        content = file.read()
        
        # Use regular expressions to extract relevant data
        pattern = r"Solver's objective value is: (\d+\.\d+)\s+Time taken by Solver: ([\d\.]+)\s+Revised Simplex Method's Optimal Value is: (\d+\.\d+)\s+Time taken by Revised Simplex Method: ([\d\.]+)"
        matches = re.findall(pattern, content)
        
        for match in matches:
            solver_obj_value = float(match[0])
            solver_time = float(match[1])
            rms_obj_value = float(match[2])
            rms_time = float(match[3])
            
            results.append((solver_obj_value, rms_obj_value, solver_time, rms_time))
    
    return results

# Example Usage
file_path = 'output.txt'

# Extract the values from the file
results = extract_values_from_file(file_path)

observation_file = open("observation.txt", "w")
# Compare the results for each problem instance
for i, (solver_obj_value, rms_obj_value, solver_time, rms_time) in enumerate(results):
    observation_file.write(f"Problem Instance {i + 1}:\n")
    observation_file.write(compare_solver_rms_results(solver_obj_value, rms_obj_value, solver_time, rms_time))
    observation_file.write("\n\n")
