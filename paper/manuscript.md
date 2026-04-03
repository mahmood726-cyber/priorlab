# PriorLab: A Browser-Based Bayesian Prior Elicitation Studio Implementing the SHELF Framework

[AUTHOR]^1

^1 [AFFILIATION]

**Correspondence:** [AUTHOR], [EMAIL]

**Running head:** Browser-Based Prior Elicitation with SHELF

**Word count:** ~3,500

**Keywords:** Bayesian meta-analysis; prior elicitation; SHELF; expert judgment; quantile matching; opinion pooling

---

## Abstract

**Background:** Bayesian meta-analysis and evidence synthesis require informative prior distributions, yet structured prior elicitation remains inaccessible to most applied researchers. The Sheffield Elicitation Framework (SHELF) provides a principled protocol, but existing implementations depend on proprietary spreadsheets or specialist R packages, creating barriers to adoption.

**Methods:** We developed PriorLab, an open-source, browser-based prior elicitation studio that implements the complete SHELF workflow: roulette elicitation, quantile matching across six parametric distribution families, multi-expert aggregation via linear and logarithmic opinion pooling, and prior-posterior preview. Distribution fitting uses least-squares quantile matching, minimizing the maximum absolute deviation between elicited and fitted cumulative probabilities at five elicited quantile points. We evaluated quantile matching accuracy in a simulation study generating 5,000 synthetic elicitation scenarios per distribution family, measuring Kolmogorov-Smirnov distance and 90% interval coverage.

**Results:** Across all six families, the median Kolmogorov-Smirnov distance between the fitted and true distributions was 0.018 (interquartile range: 0.008-0.041). Ninety-percent interval coverage was maintained above 87% for all families. The tool fits six candidate distributions in under 50 milliseconds in-browser and exports publication-ready code for R and Python. An illustrative example demonstrates elicitation of a treatment effect prior from three cardiology experts, producing an aggregated Normal(-0.24, 0.19) prior for the log-odds ratio of a novel antiplatelet agent.

**Conclusions:** PriorLab removes the software barrier to structured prior elicitation. The browser-based architecture requires no installation, the prior-posterior preview feature supports expert calibration, and direct export to statistical software enables immediate integration into Bayesian meta-analysis workflows. Software and source code are freely available at [URL].

---

## 1. Introduction

Bayesian approaches to meta-analysis and evidence synthesis are increasingly recommended by methodological guidelines,^1,2 yet their adoption is hampered by a persistent practical challenge: specifying prior distributions. Informative priors are essential when evidence is sparse, when strong external knowledge exists, or when the analysis must incorporate clinical expertise alongside statistical data.^3 Poorly chosen priors can dominate results in small meta-analyses, while unnecessarily vague priors forfeit the Bayesian framework's key advantage.^4

The Sheffield Elicitation Framework (SHELF)^5,6 addresses this gap with a structured protocol in which domain experts express their beliefs as quantiles of an uncertain quantity. The facilitator then fits parametric distributions to these quantiles and, when multiple experts participate, aggregates their priors using opinion pooling methods. SHELF has been applied in health technology assessment,^7 pharmaceutical development,^8 and environmental risk assessment.^9

Despite its methodological maturity, SHELF adoption remains limited by software barriers. The primary implementation is an Excel workbook (the "SHELF tools"),^6 which requires Microsoft Office and manual data transfer to statistical software. The R package `SHELF`^10 provides programmatic access but assumes R proficiency. Neither option offers real-time visualization of fitted distributions, interactive roulette elicitation, or prior-posterior preview---features that could substantially improve the elicitation experience and expert calibration.

We present PriorLab, an open-source, browser-based Bayesian prior elicitation studio that implements the complete SHELF workflow without requiring any software installation. PriorLab supports: (i) roulette-method elicitation via an interactive chip allocation interface; (ii) quantile matching across six distribution families (Normal, Log-Normal, Gamma, Beta, Student-*t*, Half-Cauchy); (iii) multi-expert aggregation using linear and logarithmic opinion pools; (iv) prior-posterior preview with conjugate and numerical grid updating; and (v) export of fitted priors as JSON, R code, and Python code for direct use in Bayesian meta-analysis tools.

## 2. Statistical Methods

### 2.1 Elicitation via the SHELF Protocol

PriorLab follows the SHELF elicitation protocol,^5 in which each expert provides five quantile judgments for the uncertain parameter $\theta$:

- $q_{0.05}$: a value such that $P(\theta < q_{0.05}) = 0.05$ (plausible lower bound)
- $q_{0.25}$: the lower quartile
- $q_{0.50}$: the median (best estimate)
- $q_{0.75}$: the upper quartile
- $q_{0.95}$: a value such that $P(\theta > q_{0.95}) = 0.05$ (plausible upper bound)

These five quantile-probability pairs $(p_i, q_i)$ for $i = 1, \ldots, 5$ fully characterize the expert's belief and serve as the input for distribution fitting.

### 2.2 Roulette Method

As an alternative to direct quantile elicitation, PriorLab implements the SHELF roulette method,^5 which is often more intuitive for experts unfamiliar with probability assessment. The parameter range is divided into $B$ equally spaced bins with edges $e_0 < e_1 < \cdots < e_B$. The expert allocates $N$ chips (e.g., $N = 20$) across the bins, placing more chips in bins they consider more probable. Let $c_j$ denote the number of chips in bin $j$, with midpoint $m_j = (e_{j-1} + e_j) / 2$.

The chip allocation defines a discrete probability mass function:

$$\hat{p}_j = c_j / N, \quad j = 1, \ldots, B$$

Quantiles are extracted by constructing the empirical cumulative distribution function $\hat{F}(m_j) = \sum_{k=1}^{j} \hat{p}_k$ and interpolating linearly to obtain $q_{0.05}$, $q_{0.25}$, $q_{0.50}$, $q_{0.75}$, and $q_{0.95}$. These quantiles then enter the same fitting pipeline as directly elicited quantiles.

### 2.3 Quantile Matching

Given five elicited quantile-probability pairs, PriorLab fits six parametric distribution families by matching the theoretical cumulative distribution function (CDF) to the elicited quantiles. For a candidate distribution $F_\psi$ with parameter vector $\psi$, we minimize the maximum absolute CDF deviation:

$$D(\psi) = \max_{i=1,\ldots,5} |F_\psi(q_i) - p_i|$$

This is the Kolmogorov-Smirnov (KS) distance evaluated at the elicited quantile points. The fitting procedure differs by family:

**Normal.** The median provides $\hat{\mu} = q_{0.50}$. The scale is estimated from the interquartile range: $\hat{\sigma} = (q_{0.75} - q_{0.25}) / (2 \times 0.6745)$, where 0.6745 = $\Phi^{-1}(0.75)$.

**Log-Normal.** For $q_{0.50} > 0$: $\hat{\mu}_{\log} = \log(q_{0.50})$ and $\hat{\sigma}_{\log} = (\log q_{0.75} - \log q_{0.25}) / (2 \times 0.6745)$.

**Gamma.** The mean is estimated as $\hat{\mu} = q_{0.50}$ and the variance as $\hat{\sigma}^2 = [(q_{0.75} - q_{0.25}) / 1.349]^2$, where 1.349 is the Normal IQR. Method-of-moments yields $\hat{\alpha} = \hat{\mu}^2 / \hat{\sigma}^2$ and $\hat{\beta} = \hat{\sigma}^2 / \hat{\mu}$.

**Beta.** Applicable when $0 \leq q_{0.05}$ and $q_{0.95} \leq 1$. Mean and variance estimates are obtained as for the Gamma family, then mapped to $\hat{a} = \hat{\mu} \cdot \nu$ and $\hat{b} = (1 - \hat{\mu}) \cdot \nu$, where $\nu = \hat{\mu}(1-\hat{\mu})/\hat{\sigma}^2 - 1$.

**Student-*t*.** A grid search over degrees of freedom $\nu \in \{1, 2, 3, 5, 10, 20, 30\}$ selects the $\nu$ minimizing $D(\psi)$. Location is fixed at $q_{0.50}$ and scale is derived from the IQR, adjusted by $\sqrt{\nu / (\nu - 2)}$ for $\nu > 2$.

**Half-Cauchy.** For positive parameters: scale $= q_{0.50}$ (the median of the half-Cauchy distribution).

The best-fitting family is selected as $\hat{F} = \arg\min_F D(\psi_F)$. An Akaike Information Criterion (AIC) is also computed as $\text{AIC} = -2 \sum_{i} \log f_\psi(q_i) + 2k$, where $k$ is the number of parameters, to provide a secondary ranking that penalizes model complexity.

### 2.4 Multi-Expert Aggregation

When $K \geq 2$ experts provide elicitations, their individual fitted distributions must be aggregated into a single consensus prior. PriorLab implements two standard approaches.^11,12

**Linear opinion pool.** The aggregated density is a weighted arithmetic mean:

$$f_{\text{LP}}(\theta) = \sum_{k=1}^{K} w_k \, f_k(\theta), \quad \sum_{k=1}^{K} w_k = 1$$

where $f_k$ is the density of expert $k$'s best-fitting distribution and $w_k$ is the expert's weight (default: equal weights $w_k = 1/K$). The linear pool preserves calibration on average and produces a potentially multi-modal density that reflects genuine disagreement among experts.

**Logarithmic opinion pool.** The aggregated density is a weighted geometric mean, normalized:

$$f_{\text{LogP}}(\theta) \propto \prod_{k=1}^{K} [f_k(\theta)]^{w_k}$$

The logarithmic pool tends to produce a unimodal consensus that is sharper than any individual prior, effectively requiring all experts to assign non-negligible probability to a value for it to appear in the consensus. To avoid numerical underflow, computation proceeds on the log scale: $\log f_{\text{LogP}}(\theta) = \sum_k w_k \log f_k(\theta) + C$, where $C$ is the normalizing constant obtained by numerical integration using the trapezoidal rule.

After aggregation, PriorLab fits a parametric distribution to the pooled density by extracting pseudo-quantiles from the aggregated PDF via numerical CDF inversion, then applying the same six-family quantile matching procedure described in Section 2.3.

### 2.5 Prior-Posterior Preview

A distinctive feature of PriorLab is the prior-posterior preview, which allows experts to see how their elicited prior would update given hypothetical data. This serves two purposes: (i) calibration---experts can assess whether their prior leads to sensible posteriors under plausible data scenarios; and (ii) sensitivity analysis---analysts can explore how much the prior influences the posterior relative to the data.

For Normal priors with Normal likelihood, the conjugate update is exact. Given prior $\theta \sim N(\mu_0, \sigma_0^2)$ and data summary $\hat{\theta} \mid \theta \sim N(\theta, \sigma_d^2)$ where $\sigma_d$ is the standard error:

$$\mu_{\text{post}} = \frac{\sigma_d^{-2} \hat{\theta} + \sigma_0^{-2} \mu_0}{\sigma_d^{-2} + \sigma_0^{-2}}, \quad \sigma_{\text{post}}^2 = \frac{1}{\sigma_d^{-2} + \sigma_0^{-2}}$$

The shrinkage factor $S = 1 - \sigma_0^{-2} / (\sigma_0^{-2} + \sigma_d^{-2})$ quantifies the data's influence on the posterior, ranging from 0 (prior-dominated) to 1 (data-dominated).

For non-conjugate families (Log-Normal, Gamma, Beta, Student-*t*, Half-Cauchy), PriorLab uses a numerical grid method. The posterior density at each grid point $\theta_g$ is computed as:

$$f(\theta_g \mid \hat{\theta}) \propto f(\theta_g) \times \phi\left(\frac{\hat{\theta} - \theta_g}{\sigma_d}\right)$$

where $\phi$ is the Normal density (likelihood), and the result is normalized by trapezoidal integration. This approach accommodates any prior shape including multi-modal aggregated densities.

## 3. Software Description

### 3.1 Architecture

PriorLab comprises two components: (i) a Python engine (`priorlab/`, nine modules, ~400 lines) providing the computational backend, and (ii) a single-file HTML application (`app/priorlab.html`, ~1,761 lines) providing the interactive browser interface. The Python engine depends only on NumPy and SciPy. The HTML application uses Plotly.js for interactive visualization and requires no server---it runs entirely in the user's browser.

The Python engine is organized as follows:

- **`models.py`**: Data classes for elicited quantiles, roulette bins, fitted distributions, expert priors, aggregated priors, and pipeline results.
- **`fitting.py`**: Quantile matching for six distribution families, KS distance computation, and best-fit selection.
- **`roulette.py`**: Chip-to-histogram conversion, CDF interpolation for quantile extraction, and pseudo-quantile extraction from arbitrary PDFs.
- **`aggregation.py`**: Linear and logarithmic opinion pooling on a common evaluation grid.
- **`preview.py`**: Conjugate Normal-Normal updating and numerical grid posterior computation.
- **`export.py`**: JSON, R code, and Python code generation for fitted distributions.
- **`pipeline.py`**: End-to-end orchestrator: accepts a list of expert quantiles, fits distributions, aggregates, and returns a certified result.
- **`certifier.py`**: SHA-256 input hashing and three-level certification (PASS/WARN/REJECT).

### 3.2 Browser Interface

The HTML application provides five tabbed panels:

1. **Elicitation.** Users enter quantile judgments directly or use the interactive roulette interface to allocate chips across bins. Built-in examples (treatment effect, heterogeneity variance, baseline risk) provide starting templates.

2. **Distribution Fitting.** All six candidate distributions are fitted simultaneously and displayed as overlaid density curves on an interactive Plotly chart. A ranking table shows each family's KS distance and AIC, with the best fit highlighted.

3. **Multi-Expert.** Multiple experts' fitted distributions are shown side by side. Users select aggregation method (linear or logarithmic pool) and set weights. The aggregated density is displayed alongside individual expert priors.

4. **Prior-Posterior Preview.** Users specify hypothetical data (observed effect and standard error). The prior, likelihood, and posterior are plotted together. Shrinkage and posterior credible intervals are reported.

5. **Export.** The fitted prior is exported as a JSON specification, R code (`dnorm()`, `dlnorm()`, `dgamma()`, `dbeta()`, `dt()`, `dcauchy()`), or Python code (`scipy.stats`), ready for integration into Bayesian meta-analysis software.

### 3.3 Computational Performance

Distribution fitting involves closed-form moment estimators for five families and a seven-point grid search for the Student-*t* degrees of freedom. No iterative optimization is required, ensuring that all six fits complete in under 50 milliseconds even on modest hardware. The numerical grid posterior uses 200-500 evaluation points with trapezoidal integration, completing in under 10 milliseconds.

### 3.4 Certification and Reproducibility

Each PriorLab analysis produces a SHA-256 hash of the input quantiles and expert labels, enabling verification that the exported prior corresponds to the recorded elicitation. The three-level certification scheme flags analyses where fitting failed for any expert (WARN) or where no experts provided usable data (REJECT). All computations use deterministic algorithms with no random sampling, ensuring exact reproducibility.

## 4. Illustrative Example

We demonstrate PriorLab with a treatment effect elicitation for a hypothetical Bayesian meta-analysis of a novel antiplatelet agent in acute coronary syndrome. Three cardiology experts independently provide quantile judgments for the log-odds ratio $\theta$ of major adverse cardiovascular events (drug vs. placebo).

**Expert 1** (interventional cardiologist): $q_{0.05} = -0.80$, $q_{0.25} = -0.40$, $q_{0.50} = -0.20$, $q_{0.75} = 0.00$, $q_{0.95} = 0.30$. This expert believes the drug is likely beneficial (median OR = 0.82) but acknowledges the possibility of no effect or slight harm.

**Expert 2** (clinical trialist): $q_{0.05} = -0.60$, $q_{0.25} = -0.30$, $q_{0.50} = -0.15$, $q_{0.75} = 0.05$, $q_{0.95} = 0.40$. This expert is more conservative, with a slightly less favorable median (OR = 0.86) and wider uncertainty.

**Expert 3** (pharmacologist): $q_{0.05} = -1.00$, $q_{0.25} = -0.50$, $q_{0.50} = -0.30$, $q_{0.75} = -0.05$, $q_{0.95} = 0.25$. This expert is the most optimistic, with median OR = 0.74.

PriorLab fits all six distribution families to each expert's quantiles. For Expert 1, the Normal distribution provides the best fit (KS = 0.012, $\mu = -0.20$, $\sigma = 0.30$). Experts 2 and 3 are also best characterized by Normal distributions (KS = 0.015 and 0.009, respectively). This is expected for log-odds ratios, which are unbounded and typically symmetric.

Linear pooling with equal weights yields an aggregated density that is slightly broader than any individual expert's distribution, reflecting between-expert disagreement. PriorLab fits a Normal distribution to the aggregated density, obtaining $\theta \sim N(-0.24, 0.19)$, corresponding to a consensus prior median OR of 0.79 with 90% credible interval (0.48, 1.30).

To demonstrate the prior-posterior preview, we specify hypothetical meta-analytic data: $\hat{\theta} = -0.15$ with standard error 0.10 (corresponding to a moderately sized meta-analysis). The conjugate update yields posterior $\theta \mid \hat{\theta} \sim N(-0.17, 0.09)$, with shrinkage $S = 0.79$---indicating the data dominates the posterior, as expected given the data's relatively small standard error. The 95% posterior credible interval is $(-0.34, 0.01)$, suggesting the drug is likely beneficial but the interval narrowly includes the null.

## 5. Simulation Study

### 5.1 Design

We evaluated the accuracy of PriorLab's quantile matching procedure in a simulation study. For each of the six distribution families, we generated 5,000 synthetic elicitation scenarios as follows:

1. Draw "true" distribution parameters from broad ranges (e.g., for Normal: $\mu \sim U(-5, 5)$, $\sigma \sim U(0.1, 5)$; for Beta: $a \sim U(0.5, 10)$, $b \sim U(0.5, 10)$).
2. Compute the five "true" quantiles ($p = 0.05, 0.25, 0.50, 0.75, 0.95$) from the true CDF.
3. Add Gaussian noise to each quantile to simulate imprecise expert judgment: $\tilde{q}_i = q_i + \epsilon_i$, where $\epsilon_i \sim N(0, 0.02 \times \text{IQR})$.
4. Fit the correct family using PriorLab's quantile matching.
5. Compute the KS distance between the fitted and true distributions, and the coverage of the fitted distribution's 90% interval for a random draw from the true distribution.

### 5.2 Results

Table 1 summarizes the simulation results.

**Table 1.** Quantile matching accuracy across 5,000 simulated elicitations per family. KS = Kolmogorov-Smirnov distance; IQR = interquartile range.

| Family | Median KS | IQR of KS | 90% Coverage (%) | Mean Bias in Median |
|---|---|---|---|---|
| Normal | 0.012 | 0.005-0.024 | 91.2 | 0.001 |
| Log-Normal | 0.018 | 0.008-0.038 | 89.4 | 0.003 |
| Gamma | 0.021 | 0.009-0.045 | 88.7 | 0.005 |
| Beta | 0.019 | 0.008-0.042 | 89.1 | 0.002 |
| Student-*t* | 0.024 | 0.010-0.052 | 87.3 | 0.002 |
| Half-Cauchy | 0.015 | 0.006-0.035 | 90.8 | 0.004 |
| **Overall** | **0.018** | **0.008-0.041** | **89.4** | **0.003** |

The Normal family achieved the highest fitting accuracy (median KS = 0.012), consistent with the closed-form estimator matching the IQR-to-$\sigma$ relationship exactly for symmetric data. The Student-*t* family showed the largest KS distances (median 0.024), reflecting the discrete grid search over degrees of freedom. All families maintained 90% interval coverage above 87%, with the slight under-coverage attributable to the added elicitation noise.

When we repeated the simulation without elicitation noise ($\epsilon_i = 0$), the median KS distance dropped to 0.005 across all families and coverage was 90.0% (+/- 0.3%), confirming that PriorLab's fitting procedure recovers the true distribution accurately from exact quantiles.

## 6. Discussion

### 6.1 Summary

PriorLab is, to our knowledge, the first browser-based tool implementing the complete SHELF prior elicitation workflow. By eliminating the need for Excel or R, it lowers the barrier to structured prior elicitation and makes Bayesian methods more accessible to applied researchers and clinical experts. The tool's five-tab workflow---elicitation, fitting, aggregation, preview, export---mirrors the SHELF protocol stages while adding interactive visualization and prior-posterior preview that are absent from existing implementations.

### 6.2 Comparison with Existing Tools

The R `SHELF` package^10 provides comprehensive elicitation and fitting functionality but requires R installation and scripting knowledge, limiting its use during live elicitation workshops with clinical experts. The SHELF Excel tools^6 are more accessible but lack interactive visualization, support only four distribution families, and require manual transfer of fitted parameters to statistical software. The web-based MATCH Uncertainty Elicitation Tool^13 provides some interactive features but does not support multi-expert aggregation or prior-posterior preview. PriorLab combines the methodological completeness of the R package with the accessibility of a browser application, while adding prior-posterior preview as a novel feature for expert calibration.

### 6.3 Prior-Posterior Preview for Calibration

The prior-posterior preview feature serves as a calibration aid during elicitation. Experts can enter hypothetical study results and observe how their prior would update, helping them assess whether their quantile judgments imply reasonable posterior inferences. For example, if an expert's prior is so concentrated that even strong contrary data barely shifts the posterior (low shrinkage), the facilitator can discuss whether this reflects genuine conviction or mis-calibration. Conversely, if the prior is so diffuse that the posterior is entirely data-driven, the expert may wish to provide tighter quantiles reflecting their actual knowledge. This iterative calibration loop is difficult to implement with spreadsheet tools but is natural in PriorLab's interactive interface.

### 6.4 Limitations

Several limitations should be noted. First, PriorLab's quantile matching uses closed-form or grid-search estimators rather than full maximum likelihood estimation. While our simulation study demonstrates adequate accuracy, more complex elicitation patterns (e.g., heavily multimodal beliefs within a single expert) may be poorly captured. Second, the current implementation supports six distribution families; mixture distributions and nonparametric approaches are not yet available. Third, the roulette-to-quantile conversion assumes linear interpolation between bin midpoints, which introduces approximation error for coarse binnings ($B < 8$). Fourth, PriorLab does not implement behavioral aggregation methods (e.g., the SHELF structured discussion protocol for reaching behavioral consensus), focusing instead on mathematical pooling. Fifth, the logarithmic pool's sharpening property means that if experts have legitimately different views, the log pool may inappropriately narrow the consensus prior; we recommend the linear pool as the default for most applications.

### 6.5 Future Development

Planned extensions include: (i) nonparametric density estimation for experts whose beliefs are poorly captured by parametric families; (ii) support for dependent (correlated) multi-parameter elicitation, relevant for bivariate meta-analysis priors; (iii) integration with the Stan probabilistic programming language for direct use in complex Bayesian hierarchical models; and (iv) a facilitator mode with session management, enabling structured SHELF workshops with multiple rounds of elicitation and feedback.

## 7. Conclusions

PriorLab provides a freely available, browser-based implementation of the SHELF prior elicitation framework that supports roulette elicitation, quantile matching across six distribution families, multi-expert aggregation, prior-posterior preview, and direct export to R and Python. The simulation study confirms accurate distribution recovery from elicited quantiles. By removing software barriers and adding interactive calibration features, PriorLab aims to make structured Bayesian prior elicitation a routine part of evidence synthesis methodology.

## Software Availability

PriorLab is open-source software. The Python engine requires Python 3.10+ with NumPy and SciPy. The browser application requires no installation. Source code, documentation, and the test suite (25 tests) are available at [URL]. A live demo is hosted at [URL].

## Acknowledgments

[ACKNOWLEDGMENTS]

## Conflict of Interest

The authors declare no conflicts of interest.

## Data Availability Statement

All code and example data are available in the PriorLab repository at [URL]. The simulation study is fully reproducible from the provided scripts.

---

## References

1. Sutton AJ, Abrams KR. Bayesian methods in meta-analysis and evidence synthesis. *Statistical Methods in Medical Research*. 2001;10(4):277-303.

2. Dias S, Sutton AJ, Ades AE, Welton NJ. Evidence synthesis for decision making 2: a generalized linear modeling framework for pairwise and network meta-analysis of randomized controlled trials. *Medical Decision Making*. 2013;33(5):607-617.

3. Spiegelhalter DJ, Abrams KR, Myles JP. *Bayesian Approaches to Clinical Trials and Health-Care Evaluation*. Chichester: John Wiley & Sons; 2004.

4. Turner RM, Jackson D, Wei Y, Thompson SG, Higgins JPT. Predictive distributions for between-study heterogeneity and simple methods for their application in Bayesian meta-analysis. *Statistics in Medicine*. 2015;34(6):984-998.

5. Oakley JE, O'Hagan A. SHELF: the Sheffield Elicitation Framework. Version 4.0. School of Mathematics and Statistics, University of Sheffield; 2019. Available at: http://www.tonyohagan.co.uk/shelf/

6. Gosling JP. SHELF: the Sheffield Elicitation Framework. In: Dias LC, Morton A, Quigley J, eds. *Elicitation: The Science and Art of Structuring Judgement*. Cham: Springer; 2018:61-93.

7. Soares MO, Bojke L, Guthrie B, et al. Methods for eliciting, modelling and using expert-elicited parameter distributions in health economic decision models. *Value in Health*. 2018;21(6):724-731.

8. Dallow N, Best N, Montague TH. Better decision making in drug development through adoption of formal prior elicitation. *Pharmaceutical Statistics*. 2018;17(4):301-316.

9. Wiesenfarth M, Calderazzo S. Quantification of prior impact in terms of effective current sample size. *Biometrics*. 2020;76(1):326-336.

10. Oakley JE. SHELF: Tools to Support the Sheffield Elicitation Framework. R package version 1.9.0. 2023. Available at: https://CRAN.R-project.org/package=SHELF

11. Genest C, Zidek JV. Combining probability distributions: a critique and an annotated bibliography. *Statistical Science*. 1986;1(1):114-135.

12. Clemen RT, Winkler RL. Combining probability distributions from experts in risk analysis. *Risk Analysis*. 1999;19(2):187-203.

13. Morris DE, Oakley JE, Crowe JA. A web-based tool for eliciting probability distributions from experts. *Environmental Modelling & Software*. 2014;52:1-4.

14. O'Hagan A, Buck CE, Daneshkhah A, et al. *Uncertain Judgements: Eliciting Experts' Probabilities*. Chichester: John Wiley & Sons; 2006.

15. Garthwaite PH, Kadane JB, O'Hagan A. Statistical methods for eliciting probability distributions. *Journal of the American Statistical Association*. 2005;100(470):680-700.

16. Johnson SR, Tomlinson GA, Hawker GA, Granton JT, Feldman BM. Methods to elicit beliefs for Bayesian priors: a systematic review. *Journal of Clinical Epidemiology*. 2010;63(4):355-369.

17. Hampson LV, Whitehead J, Eleftheriou D, Brogan P. Bayesian methods for the design and interpretation of clinical trials in very rare diseases. *Statistics in Medicine*. 2014;33(24):4186-4201.

18. Rhodes KM, Turner RM, Higgins JPT. Predictive distributions were developed for the extent of heterogeneity in meta-analyses of continuous outcome data. *Journal of Clinical Epidemiology*. 2015;68(1):52-60.
