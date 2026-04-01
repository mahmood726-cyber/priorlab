import numpy as np


def linear_pool(pdfs, grid, weights=None):
    """Linear opinion pool: weighted arithmetic mean of densities."""
    n = len(pdfs)
    if weights is None:
        weights = [1.0 / n] * n
    result = np.zeros_like(grid, dtype=float)
    for pdf, w in zip(pdfs, weights):
        result += w * np.array(pdf)
    return result


def log_pool(pdfs, grid, weights=None):
    """Logarithmic opinion pool: weighted geometric mean, normalized."""
    n = len(pdfs)
    if weights is None:
        weights = [1.0 / n] * n
    log_result = np.zeros_like(grid, dtype=float)
    for pdf, w in zip(pdfs, weights):
        arr = np.array(pdf)
        arr = np.maximum(arr, 1e-300)  # avoid log(0)
        log_result += w * np.log(arr)
    result = np.exp(log_result)
    # Normalize
    integral = np.trapezoid(result, grid)
    if integral > 0:
        result /= integral
    return result
