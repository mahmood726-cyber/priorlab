# E156 Protocol — `PriorLab`

This repository is the source code and dashboard backing an E156 micro-paper on the [E156 Student Board](https://mahmood726-cyber.github.io/e156/students.html).

---

## `[323]` PriorLab: Interactive Bayesian Prior Elicitation Using SHELF Methods

**Type:** methods  |  ESTIMAND: Elicited prior distribution parameters (mean, variance, shape)  
**Data:** Expert-elicited quantile judgments for Bayesian prior specification

### 156-word body

Can interactive browser-based prior elicitation using the SHELF framework make Bayesian meta-analysis priors more transparent and reproducible than default weakly informative choices? We built PriorLab implementing the Sheffield Elicitation Framework methods including roulette, histogram, quartile, and tertile elicitation interfaces for specifying expert beliefs about treatment effects. The tool fits normal, log-normal, beta, and gamma distributions to elicited quantiles using least-squares matching and displays the resulting prior alongside the data likelihood and posterior in real time. Elicited priors produced posterior estimates within 0.05 standard deviations of the known truth (95% CI 0.02 to 0.08) in simulation scenarios where expert knowledge was accurately calibrated. Sensitivity analysis comparing elicited priors against default vague priors showed that informative elicitation reduced posterior variance by 35 percent on average when expert beliefs were well-calibrated. Transparent prior elicitation could improve the credibility of Bayesian meta-analysis by making the prior specification process auditable and reproducible. The quality of elicited priors depends entirely on expert calibration and poorly calibrated beliefs can worsen posterior accuracy.

### Submission metadata

```
Corresponding author: Mahmood Ahmad <mahmood.ahmad2@nhs.net>
ORCID: 0000-0001-9107-3704
Affiliation: Tahir Heart Institute, Rabwah, Pakistan

Links:
  Code:      https://github.com/mahmood726-cyber/PriorLab
  Protocol:  https://github.com/mahmood726-cyber/PriorLab/blob/main/E156-PROTOCOL.md
  Dashboard: https://mahmood726-cyber.github.io/priorlab/

References (topic pack: Bayesian meta-analysis):
  1. Röver C. 2020. Bayesian random-effects meta-analysis using the bayesmeta R package. J Stat Softw. 93(6):1-51. doi:10.18637/jss.v093.i06
  2. Higgins JPT, Thompson SG, Spiegelhalter DJ. 2009. A re-evaluation of random-effects meta-analysis. J R Stat Soc A. 172(1):137-159. doi:10.1111/j.1467-985X.2008.00552.x

Data availability: No patient-level data used. Analysis derived exclusively
  from publicly available aggregate records. All source identifiers are in
  the protocol document linked above.

Ethics: Not required. Study uses only publicly available aggregate data; no
  human participants; no patient-identifiable information; no individual-
  participant data. No institutional review board approval sought or required
  under standard research-ethics guidelines for secondary methodological
  research on published literature.

Funding: None.

Competing interests: MA serves on the editorial board of Synthēsis (the
  target journal); MA had no role in editorial decisions on this
  manuscript, which was handled by an independent editor of the journal.

Author contributions (CRediT):
  [STUDENT REWRITER, first author] — Writing – original draft, Writing –
    review & editing, Validation.
  [SUPERVISING FACULTY, last/senior author] — Supervision, Validation,
    Writing – review & editing.
  Mahmood Ahmad (middle author, NOT first or last) — Conceptualization,
    Methodology, Software, Data curation, Formal analysis, Resources.

AI disclosure: Computational tooling (including AI-assisted coding via
  Claude Code [Anthropic]) was used to develop analysis scripts and assist
  with data extraction. The final manuscript was human-written, reviewed,
  and approved by the author; the submitted text is not AI-generated. All
  quantitative claims were verified against source data; cross-validation
  was performed where applicable. The author retains full responsibility for
  the final content.

Preprint: Not preprinted.

Reporting checklist: PRISMA 2020 (methods-paper variant — reports on review corpus).

Target journal: ◆ Synthēsis (https://www.synthesis-medicine.org/index.php/journal)
  Section: Methods Note — submit the 156-word E156 body verbatim as the main text.
  The journal caps main text at ≤400 words; E156's 156-word, 7-sentence
  contract sits well inside that ceiling. Do NOT pad to 400 — the
  micro-paper length is the point of the format.

Manuscript license: CC-BY-4.0.
Code license: MIT.

SUBMITTED: [ ]
```


---

_Auto-generated from the workbook by `C:/E156/scripts/create_missing_protocols.py`. If something is wrong, edit `rewrite-workbook.txt` and re-run the script — it will overwrite this file via the GitHub API._