def build_export_json(result, parameter_name="theta", parameter_description=""):
    best = None
    if result.aggregated and result.aggregated.fitted:
        best = result.aggregated.fitted
    elif result.experts and result.experts[0].best_fit:
        best = result.experts[0].best_fit

    if best is None:
        return {"error": "No fitted distribution available"}

    return {
        "parameter": parameter_name,
        "description": parameter_description,
        "family": best.family,
        "params": best.params,
        "ks_distance": best.ks_distance,
        "n_experts": len(result.experts),
        "aggregation": result.aggregated.method if result.aggregated else "single_expert",
        "provenance": {
            "tool": "PriorLab v0.1.0",
            "hash": result.input_hash,
        },
    }


def generate_r_code(export_json):
    family = export_json.get("family", "normal")
    params = export_json.get("params", {})
    param_name = export_json.get("parameter", "theta")

    if family == "normal":
        return f"# Prior for {param_name}\nprior <- dnorm(x, mean = {params.get('mu', 0)}, sd = {params.get('sigma', 1)})"
    if family == "lognormal":
        return f"# Prior for {param_name}\nprior <- dlnorm(x, meanlog = {params.get('mu_log', 0)}, sdlog = {params.get('sigma_log', 1)})"
    if family == "gamma":
        return f"# Prior for {param_name}\nprior <- dgamma(x, shape = {params.get('shape', 1)}, scale = {params.get('scale', 1)})"
    if family == "beta":
        return f"# Prior for {param_name}\nprior <- dbeta(x, shape1 = {params.get('alpha', 1)}, shape2 = {params.get('beta', 1)})"
    if family == "halfcauchy":
        return f"# Prior for {param_name}\nprior <- dcauchy(x, location = 0, scale = {params.get('scale', 1)}) * 2 * (x >= 0)"
    if family == "t":
        return f"# Prior for {param_name}\nprior <- dt((x - {params.get('loc', 0)}) / {params.get('scale', 1)}, df = {params.get('df', 3)}) / {params.get('scale', 1)}"
    return f"# Prior for {param_name} ({family})\n# Params: {params}"


def generate_python_code(export_json):
    family = export_json.get("family", "normal")
    params = export_json.get("params", {})
    param_name = export_json.get("parameter", "theta")

    if family == "normal":
        return f"from scipy.stats import norm\nprior = norm(loc={params.get('mu', 0)}, scale={params.get('sigma', 1)})  # {param_name}"
    if family == "lognormal":
        return f"from scipy.stats import lognorm\nimport numpy as np\nprior = lognorm(s={params.get('sigma_log', 1)}, scale=np.exp({params.get('mu_log', 0)}))  # {param_name}"
    if family == "gamma":
        return f"from scipy.stats import gamma\nprior = gamma(a={params.get('shape', 1)}, scale={params.get('scale', 1)})  # {param_name}"
    if family == "beta":
        return f"from scipy.stats import beta\nprior = beta(a={params.get('alpha', 1)}, b={params.get('beta', 1)})  # {param_name}"
    if family == "halfcauchy":
        return f"from scipy.stats import halfcauchy\nprior = halfcauchy(scale={params.get('scale', 1)})  # {param_name}"
    if family == "t":
        return f"from scipy.stats import t\nprior = t(df={params.get('df', 3)}, loc={params.get('loc', 0)}, scale={params.get('scale', 1)})  # {param_name}"
    return f"# Prior for {param_name} ({family}): {params}"
