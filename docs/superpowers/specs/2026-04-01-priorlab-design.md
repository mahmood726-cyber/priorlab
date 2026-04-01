# PriorLab — Interactive Bayesian Prior Elicitation Studio

**Date**: 2026-04-01
**Status**: Approved
**Target Journal**: Statistics in Medicine
**Location**: `C:\Models\PriorLab\`

## Summary

Browser-based interactive tool for Bayesian prior elicitation. Implements SHELF (Sheffield Elicitation Framework) methods: quantile elicitation, roulette method, and visual distribution building. Fits 6 distribution families, supports multi-expert aggregation, and exports priors as JSON compatible with Bayesian MA tools. No existing browser tool offers this — SHELF is paper/Excel only.

## Architecture

- **Python engine** (`priorlab/`): Distribution fitting + aggregation, scipy
- **Browser app** (`app/priorlab.html`): Single-file HTML with Plotly.js, interactive
- **Test suite** (`tests/`): pytest, 25+ tests
- **TruthCert**: Hash-linked provenance

## Data Model

### ElicitedQuantiles
```python
@dataclass
class ElicitedQuantiles:
    lower: float            # P5 or P10 (plausible lower bound)
    q1: float               # P25 (first quartile)
    median: float           # P50
    q3: float               # P75 (third quartile)
    upper: float            # P95 or P90 (plausible upper bound)
    lower_p: float = 0.05   # probability for lower bound
    upper_p: float = 0.95   # probability for upper bound
```

### RouletteBins
```python
@dataclass
class RouletteBins:
    bin_edges: list[float]      # N+1 edges defining N bins
    chips: list[int]            # N chip counts (one per bin)
    total_chips: int = 0        # auto-computed sum
```

### FittedDistribution
```python
@dataclass
class FittedDistribution:
    family: str                 # "normal"/"lognormal"/"beta"/"gamma"/"halfcauchy"/"t"
    params: dict[str, float]    # family-specific parameters
    ks_distance: float          # Kolmogorov-Smirnov distance from elicited quantiles
    aic: float                  # Akaike information criterion (for comparison)
    x_grid: list[float]         # evaluation points
    pdf_values: list[float]     # PDF at each x_grid point
    cdf_at_quantiles: dict[str, float]  # CDF at elicited quantile points
```

### ExpertPrior
```python
@dataclass
class ExpertPrior:
    expert_id: str
    label: str
    quantiles: ElicitedQuantiles | None = None
    roulette: RouletteBins | None = None
    best_fit: FittedDistribution | None = None
    all_fits: list[FittedDistribution] = field(default_factory=list)
```

### AggregatedPrior
```python
@dataclass
class AggregatedPrior:
    method: str                 # "linear_pool" or "log_pool"
    weights: list[float]        # expert weights (default equal)
    x_grid: list[float]
    pdf_values: list[float]     # aggregated PDF
    fitted: FittedDistribution | None = None  # best-fit to aggregated
```

### PriorLabResult
```python
@dataclass
class PriorLabResult:
    experts: list[ExpertPrior]
    aggregated: AggregatedPrior | None = None
    parameter_name: str = "theta"
    parameter_description: str = ""
    export_json: dict = field(default_factory=dict)
    input_hash: str = ""
    certification: str = ""
```

## Statistical Methods

### 1. Quantile Elicitation

Expert specifies 5 quantiles: P5 (or P10), P25, P50, P75, P95 (or P90).

**Validation**:
- Must be monotonically increasing: lower < q1 < median < q3 < upper
- Probabilities must be in (0,1)

### 2. Roulette Method (SHELF)

Expert distributes N chips (default 20) across bins:
- Define range [lower, upper] and split into K bins (default 10)
- Expert places chips to represent belief (more chips = higher probability)
- Convert to histogram: `p_bin = chips_in_bin / total_chips`
- Fit distribution to the histogram midpoints + probabilities

### 3. Distribution Fitting

Fit 6 families to elicited quantiles via maximum likelihood / quantile matching:

| Family | Parameters | Fitting Method |
|--------|-----------|----------------|
| Normal | mu, sigma | Closed-form from median + IQR |
| Log-Normal | mu, sigma (log-scale) | Closed-form from log(median) + log(IQR) |
| Beta | alpha, beta | Method of moments from quantiles, scaled to [lower, upper] |
| Gamma | shape, scale | Method of moments from mean + variance |
| Half-Cauchy | scale | Fit to median (scale = median) |
| Student-t | df, loc, scale | Grid search df in [1,30], then loc/scale from quantiles |

**Fitting algorithm** (quantile matching via least squares):
```
minimize sum((F_inv(p_i) - q_i)^2) over parameters
```
Where `F_inv` is the inverse CDF and `(p_i, q_i)` are the elicited quantile pairs.

**Goodness-of-fit**:
- KS distance: max|F(q_i) - p_i| across all elicited quantile points
- AIC: -2*loglik + 2*k (using histogram or quantile pseudo-likelihood)
- Best fit = minimum KS distance

### 4. Prior-Posterior Preview

Given a hypothetical dataset (theta_data, se_data), compute:
- **Normal prior**: conjugate update → `posterior = N(mu_post, sigma2_post)` where
  - `sigma2_post = 1 / (1/sigma2_prior + 1/se_data^2)`
  - `mu_post = sigma2_post * (mu_prior/sigma2_prior + theta_data/se_data^2)`
- For non-Normal priors: numerical grid approximation
  - `posterior(theta) ∝ prior(theta) × likelihood(data|theta)`
  - Normalize on grid

### 5. Multi-Expert Aggregation

**Linear pooling** (default): `p_agg(theta) = sum(w_i * p_i(theta))` — arithmetic mean of densities.

**Logarithmic pooling**: `p_agg(theta) ∝ prod(p_i(theta)^w_i)` — geometric mean, then normalize.

Default weights: equal (`w_i = 1/n_experts`). User can adjust.

After aggregation, fit a parametric distribution to the aggregated density (re-use fitting algorithm on pseudo-quantiles extracted from the aggregated CDF).

### 6. Export Format

JSON compatible with BayesianMA tool:
```json
{
  "parameter": "theta",
  "description": "Treatment effect (log odds ratio)",
  "family": "normal",
  "params": {"mu": -0.3, "sigma": 0.15},
  "elicitation_method": "quantile",
  "n_experts": 3,
  "aggregation": "linear_pool",
  "provenance": {
    "tool": "PriorLab v0.1.0",
    "date": "2026-04-01",
    "hash": "a3f8..."
  }
}
```

R code snippet:
```r
# Prior for theta (Treatment effect)
prior_mu <- -0.3
prior_sigma <- 0.15
# dnorm(theta, mean = prior_mu, sd = prior_sigma)
```

Python code snippet:
```python
from scipy.stats import norm
prior = norm(loc=-0.3, scale=0.15)
```

## Browser App Tabs (5)

### Tab 1: Elicitation
- Parameter name + description input
- Toggle: Quantile method / Roulette method / Visual (click-to-shape)
- **Quantile**: 5 input fields (P5, P25, P50, P75, P95) with validation
- **Roulette**: Slider for range [lower, upper], grid of bins, click to add/remove chips, chip count display
- **Visual**: Plotly density curve that user can drag control points on
- Live density preview as user enters values
- "Add Expert" button for multi-expert mode

### Tab 2: Distribution Fit
- Table: 6 rows (one per family), columns: family, params, KS distance, AIC
- Best-fit highlighted
- Overlay plot: elicited quantiles (vertical lines) + all 6 fitted densities
- Select button to choose preferred distribution (defaults to best KS)

### Tab 3: Prior-Posterior Preview
- Input: hypothetical data (observed effect, SE, or sample size)
- Plotly overlay: prior density (blue), likelihood (gray), posterior density (red)
- Summary: prior mean, posterior mean, posterior CI, shrinkage %
- Slider: vary hypothetical data to see posterior shift in real-time

### Tab 4: Multi-Expert
- Expert list with individual prior summaries
- Aggregation method toggle: Linear / Logarithmic pooling
- Weight sliders per expert
- Overlay plot: individual expert densities (thin lines) + aggregated (thick line)
- Fit parametric distribution to aggregated density

### Tab 5: Export & Report
- Selected prior summary (family, params, source)
- JSON export (clipboard + download)
- R code snippet
- Python code snippet
- TruthCert bundle
- Methods paragraph for manuscript

## Visualizations (4 Plotly charts)

1. **Elicitation preview**: Live density from current inputs (updates as user types)
2. **Distribution comparison**: 6 fitted densities overlaid with elicited quantile markers
3. **Prior-posterior overlay**: Prior (blue) + likelihood (gray) + posterior (red)
4. **Expert comparison**: Individual densities (thin) + aggregated (thick)

## Built-in Examples

### 1. Treatment Effect (Log Odds Ratio)
- Context: "What is the plausible effect of a new drug on mortality?"
- Quantiles: P5=-0.8, P25=-0.4, P50=-0.2, P75=0.0, P95=0.3
- Expected best fit: Normal(-0.2, 0.28)

### 2. Heterogeneity (Tau²)
- Context: "How much between-study variation do you expect?"
- Quantiles: P5=0.001, P25=0.02, P50=0.08, P75=0.20, P95=0.50
- Expected best fit: Half-Cauchy or Log-Normal

### 3. Baseline Risk
- Context: "What is the plausible event rate in the control group?"
- Quantiles: P5=0.02, P25=0.05, P50=0.10, P75=0.20, P95=0.40
- Expected best fit: Beta(2, 18) or Log-Normal

## Test Coverage (25+ tests)

### fitting.py (8 tests)
- Normal fit from symmetric quantiles recovers mu/sigma
- Log-Normal fit from right-skewed quantiles
- Beta fit bounded [0,1] for baseline risk
- Gamma fit for positive-only parameter (tau²)
- Half-Cauchy fit to heavy-tailed prior
- Student-t fit with low df for robust prior
- KS distance is zero for perfect match
- Best fit selection from 6 candidates

### roulette.py (4 tests)
- Uniform chips → fitted prior close to uniform
- Concentrated chips → narrow fitted prior
- Chip normalization sums to 1
- Empty bins produce zero probability

### aggregation.py (4 tests)
- Linear pool of identical priors returns same prior
- Log pool of identical priors returns same prior
- Equal weights sum to 1
- Aggregated density integrates to ~1

### preview.py (4 tests)
- Normal conjugate update matches closed-form
- Posterior tighter than prior (more data = less uncertainty)
- Strong prior + weak data → posterior near prior
- Weak prior + strong data → posterior near data

### pipeline.py (5+ tests)
- End-to-end on each example
- Export JSON contains required fields
- Certification PASS for complete elicitation
- Multi-expert aggregation produces result

## File Structure

```
C:\Models\PriorLab\
  priorlab/
    __init__.py
    models.py           # All dataclasses
    fitting.py           # Distribution fitting (6 families, quantile matching)
    roulette.py          # Roulette method (chips → histogram → fit)
    aggregation.py       # Multi-expert pooling (linear, logarithmic)
    preview.py           # Prior-posterior preview (conjugate + grid)
    export.py            # JSON, R code, Python code generation
    pipeline.py          # run_priorlab() orchestrator
    certifier.py         # TruthCert
  tests/
    conftest.py
    test_fitting.py
    test_roulette.py
    test_aggregation.py
    test_preview.py
    test_pipeline.py
  app/
    priorlab.html        # Single-file browser app
  data/
    treatment_effect.json
    heterogeneity.json
    baseline_risk.json
  setup.py
  README.md
  LICENSE
```

## Out of Scope (v1)

- Copula-based multivariate elicitation
- Time-series prior updating (sequential elicitation sessions)
- Hierarchical prior structures
- Real-time collaboration (multiple experts simultaneously)
- Calibration training for experts (probability calibration quizzes)
- Integration with Stan/JAGS model code generation
- Non-parametric prior specification (Dirichlet process)
