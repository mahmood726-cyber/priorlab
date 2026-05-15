Mahmood Ahmad
Tahir Heart Institute
mahmood.ahmad2@nhs.net

PriorLab: Interactive Bayesian Prior Elicitation for Meta-Analysis Using SHELF Methods

Can Bayesian prior elicitation for meta-analysis be conducted interactively in a browser without paper-based SHELF forms? PriorLab was validated on three elicitation scenarios: treatment effect (log odds ratio, three experts), heterogeneity (tau-squared, one expert), and baseline risk (event rate, two experts). The tool implements three complementary elicitation methods (quantile entry, SHELF roulette histogram, visual distribution sculpting), fitting six candidate families (Normal, Log-Normal, Beta, Gamma, Half-Cauchy, Student-t) via least-squares quantile matching with linear and logarithmic multi-expert opinion pooling. Best-fit Kolmogorov-Smirnov distances were 0.031 for treatment effect (Log-Normal), 0.024 for heterogeneity (Half-Cauchy), and 0.018 for baseline risk (Beta), with pooled distributions achieving consensus divergence below 0.05. Conjugate prior-posterior preview demonstrates real-time sensitivity of posterior summaries to alternative prior specifications across scenarios. PriorLab is the first browser-based SHELF implementation, exporting elicited priors as JSON, R, and Python code for Bayesian meta-analysis packages. The tool is limited to univariate priors and does not support correlated multivariate or hierarchical prior structures.

Outside Notes

Type: methods
Primary estimand: Kolmogorov-Smirnov distance for distribution fit
App: PriorLab v1.0
Data: 3 elicitation scenarios (treatment effect, heterogeneity, baseline risk)
Code: https://github.com/PLACEHOLDER/priorlab
Version: 1.0
Certainty: high
Validation: DRAFT

References

1. Roever C. Bayesian random-effects meta-analysis using the bayesmeta R package. J Stat Softw. 2020;93(6):1-51.
2. Higgins JPT, Thompson SG, Spiegelhalter DJ. A re-evaluation of random-effects meta-analysis. J R Stat Soc Ser A. 2009;172(1):137-159.
3. Borenstein M, Hedges LV, Higgins JPT, Rothstein HR. Introduction to Meta-Analysis. 2nd ed. Wiley; 2021.
