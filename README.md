Transportation Problem Solver and Optimization Algorithms

Project Overview:
This Python project implements two key optimization algorithms for solving transportation and linear programming problems:
1. Transportation Problem Solver using PuLP library
2. Custom Revised Simplex Method implementation

Dependencies and Installation:
Required Libraries:
- NumPy
- PuLP
- Python 3.x (3.7 or higher recommended)

Installation Steps:
1. Ensure Python is installed on your system
2. Open terminal/command prompt
3. Install dependencies using pip:

   Option 1: Individual installations
   ```
   pip install numpy
   pip install pulp
   ```

   Option 2: Using requirements file
   Create a requirements.txt with:
   ```
   numpy
   pulp
   ```
   Then run:
   ```
   pip install -r requirements.txt
   ```

Project Files:
1. solver.py: Primary optimization script
2. comparator.py: Results comparison utility

Output Configuration:
The script includes a boolean variable 'test_by_output_file' to control output method:
- When set to True: Writes results to "output.txt"
- When set to False: Prints results to terminal

Comparator Script (comparator.py):
- Compares objective values and computation times
- Extracts results from output file
- Generates an observation.txt with detailed comparisons

How to Use Comparator:
1. Run solver.py with test_by_output_file = True
2. Execute comparator.py
3. Check observation.txt for detailed comparison results

Key Comparison Metrics:
- Objective value matching
- Computation time differences
- Performance analysis for each problem instance

Performance Metrics Tracked:
- Number of supply and demand nodes
- Maximum cost and amount
- Solver objective value
- Computation time for each method

Typical Execution:
The script runs 100 random problem instances by default, comparing solver performance and optimal values.

Troubleshooting:
- Ensure all dependencies are correctly installed
- Check Python version compatibility
- Verify input data format
