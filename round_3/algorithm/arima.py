import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt


# --- Step 1: Differencing Function ---
def difference(series, d=1):
    """
    Difference the series to make it stationary.
    """
    diffed = series.copy()
    for _ in range(d):
        diffed = diffed[1:] - diffed[:-1]
    return diffed


# --- Step 2: Inverse Difference Function ---
def inverse_difference(last_original, diffs):
    """
    Reverse differencing to recover the original scale.
    """
    result = [last_original]
    for d in diffs:
        result.append(result[-1] + d)
    return result[1:]  # Skip the first original


# --- Step 3: ARMA Log-Likelihood Function ---
def arma_loglik(params, series, p, q):
    """
    Log-likelihood function for ARMA(p, q) model.
    """
    n = len(series)
    ar_params = params[:p]
    ma_params = params[p:p + q]
    sigma = params[-1]

    eps = np.zeros(n)
    for t in range(max(p, q), n):
        ar_term = sum(ar_params[i] * series[t - i - 1] for i in range(p))
        ma_term = sum(ma_params[i] * eps[t - i - 1] for i in range(q))
        eps[t] = series[t] - ar_term - ma_term

    loglik = -0.5 * n * np.log(2 * np.pi * sigma ** 2) - 0.5 * np.sum(eps ** 2) / sigma ** 2
    return -loglik  # Minimize negative log-likelihood


# --- Step 4: Fit ARIMA Model ---
def fit_arima(series, p=1, d=1, q=1):
    """
    Fit an ARIMA model to the time series data.
    """
    # Step 1: Difference the data
    diffed = difference(series, d)
    diffed = diffed[max(p, q):]  # Skip initial lags due to differencing

    # Step 2: Initial guess for parameters
    init_params = np.random.randn(p + q + 1) * 0.1  # AR parameters + MA parameters + sigma

    # Step 3: Minimize the negative log-likelihood
    result = minimize(arma_loglik, init_params, args=(diffed, p, q), method='L-BFGS-B')

    return result.x  # Fitted parameters (AR, MA, sigma)


# --- Step 5: Forecasting Function ---
def forecast_arima(series, p, d, q, params, steps=1):
    """
    Forecast the future steps using the ARIMA model.
    """
    # Extract the parameters
    ar_params = params[:p]
    ma_params = params[p:p + q]
    sigma = params[-1]

    # Get the last value to reverse differencing
    last_value = series[-1]

    # Generate forecasts
    forecast = []
    for _ in range(steps):
        ar_term = sum(ar_params[i] * series[-i - 1] for i in range(p))
        ma_term = sum(ma_params[i] * (forecast[-i - 1] if i < len(forecast) else 0) for i in range(q))
        forecast_value = ar_term + ma_term
        forecast.append(forecast_value)

    # Reverse differencing
    forecast_original_scale = inverse_difference(last_value, forecast)

    return forecast_original_scale


# --- Step 6: Plotting Helper Function ---
def plot_series(original, forecast, title="ARIMA Forecast"):
    """
    Plot the original and forecasted series.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(original, label="Original Series")
    plt.plot(range(len(original), len(original) + len(forecast)), forecast, label="Forecast", color='red')
    plt.legend()
    plt.title(title)
    plt.show()


# --- Step 7: Example Usage with Synthetic Data ---
if __name__ == "__main__":
    # Generate synthetic data (for testing purposes)
    np.random.seed(42)
    time_series = np.cumsum(np.random.randn(100))  # Random walk

    # Fit ARIMA(p=1, d=1, q=1) to the series
    p, d, q = 1, 1, 1
    params = fit_arima(time_series, p, d, q)

    # Forecast the next 10 values
    forecast = forecast_arima(time_series, p, d, q, params, steps=10)

    # Plot the results
    plot_series(time_series, forecast, title="ARIMA(1,1,1) Forecast")
