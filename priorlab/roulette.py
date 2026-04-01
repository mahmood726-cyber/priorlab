import numpy as np
from priorlab.models import ElicitedQuantiles, RouletteBins


def roulette_to_histogram(bins):
    """Convert chip counts to probability histogram."""
    total = sum(bins.chips)
    if total <= 0:
        probs = [0.0] * len(bins.chips)
    else:
        probs = [c / total for c in bins.chips]
    midpoints = []
    for i in range(len(bins.chips)):
        mid = (bins.bin_edges[i] + bins.bin_edges[i + 1]) / 2
        midpoints.append(mid)
    return {"midpoints": midpoints, "probabilities": probs}


def roulette_to_quantiles(bins):
    """Convert roulette histogram to approximate quantiles via CDF interpolation."""
    hist = roulette_to_histogram(bins)
    midpoints = hist["midpoints"]
    probs = hist["probabilities"]

    # Build CDF from histogram
    cum_prob = []
    running = 0.0
    for p in probs:
        running += p
        cum_prob.append(running)

    def interp_quantile(target_p):
        for i, cp in enumerate(cum_prob):
            if cp >= target_p:
                if i == 0:
                    return midpoints[0]
                frac = (target_p - cum_prob[i - 1]) / (cp - cum_prob[i - 1]) if cp != cum_prob[i - 1] else 0
                return midpoints[i - 1] + frac * (midpoints[i] - midpoints[i - 1])
        return midpoints[-1]

    return ElicitedQuantiles(
        lower=interp_quantile(0.05),
        q1=interp_quantile(0.25),
        median=interp_quantile(0.50),
        q3=interp_quantile(0.75),
        upper=interp_quantile(0.95),
    )


def _pseudo_quantiles_from_pdf(grid, pdf):
    """Extract approximate quantiles from a density evaluated on a grid."""
    cdf = np.cumsum(pdf)
    dx = grid[1] - grid[0] if len(grid) > 1 else 1.0
    cdf = cdf * dx
    cdf = cdf / cdf[-1]  # normalize

    def interp(target):
        idx = np.searchsorted(cdf, target)
        idx = min(idx, len(grid) - 1)
        return float(grid[idx])

    return ElicitedQuantiles(
        lower=interp(0.05), q1=interp(0.25), median=interp(0.50),
        q3=interp(0.75), upper=interp(0.95),
    )
