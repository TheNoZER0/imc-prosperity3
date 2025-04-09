import os
import re
from itertools import product
from typing import Optional

from prosperity3bt.__main__ import run_backtest


def sanitise_filename(s):
    """
    Sanitise a string to be safe in filenames (e.g., avoid periods or special characters).
    """
    return re.sub(r'[^\w]', '_', str(s))

def grid_search_backtest(
    algorithm_file_path: str,
    grid: dict,
    rounds: list[str],
    data_path: Optional[str] = None,
) -> list[dict]:
    param_combinations = list(product(*grid.values()))
    param_names = list(grid.keys())

    results = []

    for combination in param_combinations:
        params = dict(zip(param_names, combination))
        print(f"\n\nRunning backtest with parameters: {params}")

        # Read original algorithm template
        with open(algorithm_file_path, 'r') as original_file:
            code = original_file.read()

        # Replace placeholders like {{PARAM}} in the code
        for param_name, param_value in params.items():
            placeholder = f"{{{{{param_name}}}}}"
            code = code.replace(placeholder, str(param_value))

        # Build a safe temp filename (no dots or illegal characters)
        filename_suffix = '_'.join(sanitise_filename(v) for v in combination)
        temp_file_path = os.path.join(
            os.path.dirname(algorithm_file_path),
            f"temp_algo_{filename_suffix}.py"
        )

        # Write the modified code to the temp file
        with open(temp_file_path, 'w') as temp_file:
            temp_file.write(code)

        # Run the backtest
        try:
            profit = run_backtest(
                algorithm_file_path=temp_file_path,
                data_path=data_path,
                rounds=rounds,
                merge_pnl=True,
                use_visualiser=False,
                skip_output=True,
                print_stdout=False,
                no_progress=True
            )
        except Exception as e:
            print(f"Backtest failed for {params}: {e}")
            profit = None

        # Save the result
        results.append({
            "params": params,
            "profit": profit
        })

        # Clean up the temp file
        os.remove(temp_file_path)

    return results

# Example usage:
grid = {
    "PARAM1": [0.4, 0.5],
    "RAINFOREST_RESIN": [1],
    "SQUID_INK": [1],
    "KELP": [1],
}

results = grid_search_backtest(
    algorithm_file_path=r"../trader.py",
    grid=grid,
    rounds=["1"],
    data_path=None
)

# Print results
print(f"\n\n\nResults for grid search:")
for result in results:
    print(f"Parameters: {result['params']} | Profit: {result['profit']}")
