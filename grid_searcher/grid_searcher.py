import os
import re
import io
import random
from pathlib import Path
from itertools import product
from contextlib import redirect_stdout
from concurrent.futures import ProcessPoolExecutor, as_completed
from prosperity3bt.__main__ import cli  # Assuming this remains unchanged
import multiprocessing
import numpy as np

def sanitise_filename(s):
    """Sanitise a string to be safe in filenames (e.g., avoid special characters)."""
    return re.sub(r'[^\w]', '_', str(s))

def run_backtest(combination, param_names, algorithm_file_path, rounds, data_path):
    try:
        params = dict(zip(param_names, combination))
        print(f"Running backtest with parameters: {params}")

        # Read original algorithm template.
        with open(algorithm_file_path, 'r') as original_file:
            code = original_file.read()

        # Replace placeholders like {{PARAM}} in the code.
        for param_name, param_value in params.items():
            placeholder = f"{{{{{param_name}}}}}"
            code = code.replace(placeholder, str(param_value))

        # Create a temporary file name.
        filename_suffix = '_'.join(sanitise_filename(v) for v in combination)
        temp_file_path = os.path.join(
            os.path.dirname(algorithm_file_path),
            f"temp_algo_{filename_suffix}.py"
        )

        # Write the modified code to a temporary file.
        with open(temp_file_path, 'w') as temp_file:
            temp_file.write(code)

        # Run backtest using the CLI.
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
                except Exception:
                    profit = None
                break
        profit_summary_str = "\n".join(summary_lines)
        result = {"params": params, "profit": profit, "summary": profit_summary_str}

    except Exception as e:
        print(f"Exception in run_backtest for parameters {params}: {e}")
        result = {"params": params, "profit": None, "summary": f"Error: {e}"}
    finally:
        # Clean up the temporary file.
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
    return result

def grid_search_backtest(algorithm_file_path, grid, rounds, data_path=None):
    # Create a list of parameter combinations.
    param_combinations = list(product(*grid.values()))
    param_names = list(grid.keys())

    print(f"Total parameter combinations: {len(param_combinations)}")
    
    MAX_RUNS = 5000  # Trim parameter combinations if too many.
    if len(param_combinations) > MAX_RUNS:
        print(f"Too many combinations ({len(param_combinations)}), trimming to {MAX_RUNS}")
        param_combinations = random.sample(param_combinations, MAX_RUNS)
    
    results = []
    
    # Use the "spawn" context for better compatibility.
    mp_context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(mp_context=mp_context, max_workers=8) as executor:
        futures = [executor.submit(run_backtest, comb, param_names, algorithm_file_path, rounds, data_path)
                   for comb in param_combinations]
        
        for future in as_completed(futures):
            results.append(future.result())
    
    return results

if __name__ == '__main__':
    # The freeze_support() call is necessary if you plan to freeze the code (e.g., with PyInstaller).
    multiprocessing.freeze_support()  
    
    # Example grid setup.
    grid = {
        # "threshold_b1": [i for i in range(40, 75)],
        # "threshold_b2": [i for i in range(20, 60)]
        # "threshold_p": [i for i in range(1,100)]
        # "cross_z": [i for i in np.arange(0, 3, 0.2)],
        # "cross_ema": [i for i in range(5,100,2)],
        #"alpha": [i for i in np.arange(0.1, 1, 0.05)],
        #"thresh": [i for i in np.arange(0,100, 4)]
        #"threshold_volc": [i for i in np.arange(31, 37, 0.05)],
        #"hold": [i for i in range(5, 20, 5)],
        #"arb": [i for i in range(5, 40, 2)],
        #"longp": [i for i in range(10, 50, 2)]
        # "volthresh": [i for i in np.arange(0.5, 1.5, 0.1)],
        # "csi": [i for i in np.arange(20, 50, 5)],
        #"panic": [i for i in np.arange(1, 10, 0.2)],
        # explore from 1e‑7 up to 4e‑7 in five steps
        #"vol_buy":  np.linspace(1e-7, 4e-7, num=10).tolist(),  
        # explore from 3e‑7 up to 7e‑7 in five steps
        #"vol_sell": np.linspace(3e-7, 7e-7, num=10).tolist()
        "tol": np.linspace(1.0e-7, 1e-3, num=100).tolist(),
    }   
    
    results = grid_search_backtest(
        algorithm_file_path=r"../algos/new2.py",
        grid=grid,
        rounds=["5"],
        data_path=None
    )
    
    # Filter out unsuccessful runs.
    valid_results = [r for r in results if r['profit'] is not None]
    
    if valid_results:
        best_result = max(valid_results, key=lambda x: x['profit'])
        print("\n\nResult with Highest Total Profit:")
        print(f"Parameters: {best_result['params']}")
        print(best_result['summary'])
    else:
        print("No successful backtest results were obtained.")
