import os
import re
from itertools import product
from typing import Optional
from pathlib import Path
from prosperity3bt.__main__ import cli
import io
from contextlib import redirect_stdout


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

    print(f"Total parameter combinations: {len(param_combinations)}")

    MAX_RUNS = 1  # This is the combo limit
    if len(param_combinations) > MAX_RUNS:
        print(f"Too many combinations ({len(param_combinations)}), trimming to {MAX_RUNS}")
        import random
        param_combinations = random.sample(param_combinations, MAX_RUNS)

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
        profit_summary_str = ""
        try:
            f = io.StringIO()
            with redirect_stdout(f):
                cli(
                    algorithm=Path(temp_file_path),
                    days=rounds,
                    merge_pnl=True,
                    vis=False,
                    out=None,
                    no_out=True,
                    data=Path(data_path) if data_path else None,
                    print_output=False,
                    match_trades="all",
                    no_progress=True,
                    original_timestamps=False,
                    version=False
                )
            output = f.getvalue()

            profit = None
            summary_lines = []
            found_summary = False

            for line in output.splitlines():
                if "Profit summary:" in line:
                    found_summary = True
                if found_summary:
                    summary_lines.append(line)
                if found_summary and "Total profit:" in line:
                    try:
                        profit = float(line.split("Total profit:")[-1].replace(",", "").strip())
                    except:
                        profit = None
                    break

            profit_summary_str = "\n".join(summary_lines)


        except Exception as e:
            print(f"Backtest failed for {params}: {e}")
            profit = None

        # Save the result
        results.append({
            "params": params,
            "profit": profit,
            "summary": profit_summary_str
        })

        # Clean up the temp file
        os.remove(temp_file_path)

    return results

# Example usage:
grid = {
    "resin_fair_value": [10000],
    "resin_clear_width": [0],
    "resin_take_width": [1,2,3],
    "resin_disregard_edge": [1,2,3],
    "resin_join_edge": [1,2,3],
    "resin_default_edge": [1,2,3,4,5],
    "resin_soft_position_limit": [10,11,12,13,14,15],

    "kelp_fair_value": [10000],
    "kelp_take_width": [1, 2, 3],
    "kelp_clear_width": [0],
    "kelp_disregard_edge": [1, 2, 3],
    "kelp_join_edge": [1, 2, 3], 
    "kelp_default_edge": [1, 2, 3, 4],
    
    "squink_fair_value": [10000],
    "squink_take_width": [1, 2, 3],
    "squink_clear_width": [0],
    "squink_disregard_edge": [0, 1, 2],
    "squink_join_edge": [0, 1, 2], 
    "squink_default_edge": [1, 2, 3, 4]
}


results = grid_search_backtest(
    algorithm_file_path=r"../example.py",
    grid=grid,
    rounds=["1"],
    data_path=None
)

# Print results
print(f"\n\n\nResults for grid search:")
for result in results:
    print(f"Parameters: {result['params']}")
    print(f"{result['summary']}")
    print("-" * 40)
