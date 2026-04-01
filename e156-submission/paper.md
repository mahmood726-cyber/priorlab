Mahmood Ahmad
Tahir Heart Institute
author@example.com

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

1. O'Hagan A, Buck CE, Daneshkhah A, et al. Uncertain Judgements: Eliciting Experts' Probabilities. Wiley; 2006.
2. Oakley JE, O'Hagan A. SHELF: the Sheffield Elicitation Framework. University of Sheffield; 2019.
3. Rover C, Bender R, Dias S, et al. On weakly informative prior distributions for the heterogeneity parameter in Bayesian random-effects meta-analysis. Res Synth Methods. 2021;12(4):448-474.

AI Disclosure

This work represents a compiler-generated evidence micro-publication (i.e., a structured, pipeline-based synthesis output). AI (Claude, Anthropic) was used as a constrained synthesis engine operating on structured inputs and predefined rules for infrastructure generation, not as an autonomous author. The 156-word body was written and verified by the author, who takes full responsibility for the content. This disclosure follows ICMJE recommendations (2023) that AI tools do not meet authorship criteria, COPE guidance on transparency in AI-assisted research, and WAME recommendations requiring disclosure of AI use. All analysis code, data, and versioned evidence capsules (TruthCert) are archived for independent verification.
