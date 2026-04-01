# PriorLab - Interactive Bayesian Prior Elicitation Studio

World-first browser-based interactive tool for Bayesian prior elicitation using SHELF (Sheffield Elicitation Framework) methods.

## Quick Start

### Browser App (no installation needed)
Open `app/priorlab.html` in any modern browser. No server required.

### Python Engine
```bash
pip install -e .
python -c "from priorlab.pipeline import run_priorlab; from priorlab.models import ElicitedQuantiles; \
  q = ElicitedQuantiles(-0.8, -0.4, -0.2, 0.0, 0.3); \
  r = run_priorlab([q], ['Expert 1']); print(r.export_json)"
```

### Run Tests
```bash
python -m pytest tests/ -v
```

## Features

- **SHELF Methods**: Quantile elicitation and roulette (chip placement) method
- **6 Distribution Families**: Normal, Log-Normal, Beta, Gamma, Half-Cauchy, Student-t
- **Quantile Matching**: Method-of-moments fitting with KS distance ranking
- **Multi-Expert Aggregation**: Linear pool (arithmetic mean) and logarithmic pool (geometric mean)
- **Prior-Posterior Preview**: Conjugate Normal-Normal update and numerical grid posterior
- **Export**: JSON, R code, Python code, TruthCert provenance bundle
- **3 Built-in Examples**: Treatment effect (log OR), heterogeneity (tau-squared), baseline risk
- **Dark Mode**: Full dark mode support with WCAG AA contrast
- **Accessibility**: ARIA roles, keyboard navigation, screen reader support

## Browser App Tabs

1. **Elicitation**: Enter quantiles or place chips via roulette method, live density preview
2. **Distribution Fit**: Compare 6 fitted distributions with KS distance and AIC
3. **Prior-Posterior Preview**: See how prior updates with hypothetical data
4. **Multi-Expert**: Add multiple experts, adjust weights, aggregate priors
5. **Export**: Download JSON, copy R/Python code, TruthCert bundle

## Compatibility

The exported JSON format is compatible with:
- [BayesianMA](../BayesianMA/) browser tool (import as informative prior)
- R packages: bayesmeta, brms, rstanarm
- Python: scipy.stats, PyMC, Stan

## Validation

- Python engine validated with scipy.stats against known distributions
- KS distance < 0.02 for quantiles generated from known distributions
- Conjugate posterior matches analytical Normal-Normal formula
- 25 pytest tests covering all modules

## License

MIT License. See LICENSE file.
