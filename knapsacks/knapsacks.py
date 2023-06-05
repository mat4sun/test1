# Multiple knapsacks
from ortools.linear_solver import pywraplp
import numpy as np


def create_data_model():
    """Create the data for the example."""
    data = {}
#     weights = [48, 30, 42, 36, 36, 48, 42, 42, 36, 24, 30, 30, 42, 36, 36]
#     values = [10, 30, 25, 50, 35, 30, 15, 40, 30, 35, 45, 10, 20, 30, 25]
    weights = np.array([10637, 11647, 14475, 6423, 10792, 15980, 20434, 14021, 14487, 16458,
                        14130, 13966, 3820, 14131, 440, 14109, 14150, 13668, 19969, 19991,
                        17775, 13138, 12895, 1026]) / 1024  # Hemvideos, Gb
    values = weights
    data['weights'] = weights
    data['values'] = values
    data['items'] = list(range(len(weights)))
    data['num_items'] = len(weights)
    data['bin_capacities'] = [24, 99, 99, 99]  # Blueray M-disc capacity in Gb with 1 Gb marginal
    data['bins'] = list(range(len(data['bin_capacities'])))
    return data


def main():
    data = create_data_model()

    # Create the mip solver with the CBC backend.
    solver = pywraplp.Solver.CreateSolver('multiple_knapsack_mip', 'CBC')

    # Variables
    # x[i, j] = 1 if item i is packed in bin j.
    x = {}
    for i in data['items']:
        for j in data['bins']:
            x[(i, j)] = solver.IntVar(0, 1, 'x_%i_%i' % (i, j))

    # Constraints
    # Each item can be in at most one bin.
    for i in data['items']:
        solver.Add(sum(x[i, j] for j in data['bins']) <= 1)
    # The amount packed in each bin cannot exceed its capacity.
    for j in data['bins']:
        solver.Add(
            sum(x[(i, j)] * data['weights'][i]
                for i in data['items']) <= data['bin_capacities'][j])

    # Objective
    objective = solver.Objective()

    for i in data['items']:
        for j in data['bins']:
            objective.SetCoefficient(x[(i, j)], data['values'][i])
    objective.SetMaximization()

    status = solver.Solve()

    if status == pywraplp.Solver.OPTIMAL:
        total_weight = 0
        for j in data['bins']:
            bin_weight = 0
            bin_value = 0
            print(f'Bin {j}:')
            for i in data['items']:
                if x[i, j].solution_value() > 0:
                    print('Item', i, '- weight:', data['weights'][i], ' value:',
                          data['values'][i])
                    bin_weight += data['weights'][i]
                    bin_value += data['values'][i]
            print('Packed bin weight:', bin_weight)
            print('Packed bin value:', bin_value)
            print()
            total_weight += bin_weight
            
        print(f"---\nTotal packed value: {objective.Value()}")
        print(f"Total packed weight: {total_weight} in {len(data['bins'])} containers")
        print(f"Total weight to be packed: {sum(data['weights'])}")
        print(f"Total value for items: {sum(data['values'])}")
    else:
        print('The problem does not have an optimal solution.')


if __name__ == '__main__':
    main()
