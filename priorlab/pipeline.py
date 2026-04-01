import numpy as np
from priorlab.models import PriorLabResult, ExpertPrior, AggregatedPrior
from priorlab.fitting import fit_all_distributions, select_best_fit
from priorlab.aggregation import linear_pool
from priorlab.export import build_export_json
from priorlab.certifier import compute_input_hash, certify


def run_priorlab(quantiles_list, expert_labels=None, parameter_name="theta",
                 parameter_description="", aggregation_method="linear_pool"):
    n = len(quantiles_list)
    if expert_labels is None:
        expert_labels = [f"Expert {i+1}" for i in range(n)]

    experts = []
    for i, eq in enumerate(quantiles_list):
        fits = fit_all_distributions(eq)
        best = select_best_fit(fits) if fits else None
        experts.append(ExpertPrior(
            expert_id=f"E{i+1}",
            label=expert_labels[i],
            quantiles=eq,
            best_fit=best,
            all_fits=fits,
        ))

    # Aggregation (if multiple experts)
    aggregated = None
    if n >= 2:
        # Build common grid from first expert's best fit
        grids = [np.array(e.best_fit.x_grid) for e in experts if e.best_fit]
        if grids:
            grid = grids[0]
            pdfs = [np.array(e.best_fit.pdf_values) for e in experts if e.best_fit]
            agg_pdf = linear_pool(pdfs, grid)

            # Fit parametric to aggregated
            from priorlab.roulette import _pseudo_quantiles_from_pdf
            try:
                pseudo_q = _pseudo_quantiles_from_pdf(grid, agg_pdf)
                agg_fits = fit_all_distributions(pseudo_q)
                agg_best = select_best_fit(agg_fits) if agg_fits else None
            except Exception:
                agg_best = None

            aggregated = AggregatedPrior(
                method=aggregation_method,
                weights=[1.0 / n] * n,
                x_grid=grid.tolist(),
                pdf_values=agg_pdf.tolist(),
                fitted=agg_best,
            )

    input_hash = compute_input_hash(quantiles_list, expert_labels)
    cert = certify(experts)

    result = PriorLabResult(
        experts=experts,
        aggregated=aggregated,
        parameter_name=parameter_name,
        parameter_description=parameter_description,
        input_hash=input_hash,
        certification=cert,
    )

    result.export_json = build_export_json(result, parameter_name, parameter_description)

    return result
