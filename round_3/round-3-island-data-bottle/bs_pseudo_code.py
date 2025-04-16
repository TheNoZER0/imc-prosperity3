import math
import numpy as np


# === Black-Scholes Model for Coupon Pricing ===

def norm_cdf(x):
    """Approximate the standard normal CDF using the Abramowitz and Stegun approximation."""
    # Coefficients for the approximation
    coeffs = [0.319381530, -0.356563782, 1.781477937, -1.821255978, 1.330274429]
    p = 0.2316419
    c = 0.3989422804014337  # 1 / sqrt(2 * pi)

    # Calculate the sign of x and make x positive for the approximation
    sign = 1 if x >= 0 else -1
    x = abs(x)

    # Approximation calculation
    t = 1 / (1 + p * x)
    poly = sum(c * t ** (i + 1) for i, c in enumerate(coeffs))
    result = 1 - c * math.exp(-x ** 2 / 2) * poly

    return 0.5 * (1 + sign * result)


def bs_coupon_price(spot, strike, time_to_expiry, risk_free_rate, volatility):
    """Calculate the expected price of a coupon using the Black-Scholes model."""
    if volatility == 0 or time_to_expiry == 0:
        return max(spot - strike, 0)  # Handle edge cases

    sqrt_T = np.sqrt(time_to_expiry)
    d1 = (np.log(spot / strike) + (risk_free_rate + 0.5 * volatility ** 2) * time_to_expiry) / (volatility * sqrt_T)
    d2 = d1 - volatility * sqrt_T

    # Calculate the option price using Black-Scholes formula
    price = spot * norm_cdf(d1) - strike * np.exp(-risk_free_rate * time_to_expiry) * norm_cdf(d2)
    return price


# === Bisection Method for Implied Volatility ===

def implied_volatility(market_price, spot, strike, round_number, timestamp, tol=1e-6, max_iter=1000):
    """
    Calculate the implied volatility using the bisection method.

    Args:
        market_price: Market price of the coupon.
        spot: Current spot price of volcanic rock.
        strike: Strike price of the coupon.
        round_number: Current round number.
        timestamp: Current timestamp in the round.
        tol: Tolerance for convergence.
        max_iter: Maximum number of iterations for the bisection method.
    """
    risk_free_rate = 0  # Assuming a risk-free rate of 0
    time_to_expiry = ((7_000_000 - ((round_number - 1) * 1_000_000 + timestamp)) / 1_000_000) / 365.25

    # Objective function for the bisection method
    def objective(volatility):
        return bs_coupon_price(spot, strike, time_to_expiry, risk_free_rate, volatility) - market_price

    # Initial bounds for volatility
    lower_bound = 1e-6  # Small number for the lower bound
    upper_bound = 1.0  # Reasonable upper bound for volatility

    # Check if the market price is between the bounds (edge cases)
    if objective(lower_bound) * objective(upper_bound) > 0:
        return np.nan  # No root found within the bounds

    # Perform the bisection method
    for _ in range(max_iter):
        mid_point = (lower_bound + upper_bound) / 2
        f_mid = objective(mid_point)

        if abs(f_mid) < tol:  # Found the solution within tolerance
            return mid_point
        elif objective(lower_bound) * f_mid < 0:  # Root is in the left half
            upper_bound = mid_point
        else:  # Root is in the right half
            lower_bound = mid_point

    return np.nan  # No solution found within the maximum iterations