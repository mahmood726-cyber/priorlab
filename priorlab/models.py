from dataclasses import dataclass, field


@dataclass
class ElicitedQuantiles:
    lower: float
    q1: float
    median: float
    q3: float
    upper: float
    lower_p: float = 0.05
    upper_p: float = 0.95


@dataclass
class RouletteBins:
    bin_edges: list[float] = field(default_factory=list)
    chips: list[int] = field(default_factory=list)
    total_chips: int = 0

    def __post_init__(self):
        self.total_chips = sum(self.chips)


@dataclass
class FittedDistribution:
    family: str
    params: dict[str, float] = field(default_factory=dict)
    ks_distance: float = 0.0
    aic: float = 0.0
    x_grid: list[float] = field(default_factory=list)
    pdf_values: list[float] = field(default_factory=list)
    cdf_at_quantiles: dict[str, float] = field(default_factory=dict)


@dataclass
class ExpertPrior:
    expert_id: str
    label: str
    quantiles: ElicitedQuantiles | None = None
    roulette: RouletteBins | None = None
    best_fit: FittedDistribution | None = None
    all_fits: list[FittedDistribution] = field(default_factory=list)


@dataclass
class AggregatedPrior:
    method: str = "linear_pool"
    weights: list[float] = field(default_factory=list)
    x_grid: list[float] = field(default_factory=list)
    pdf_values: list[float] = field(default_factory=list)
    fitted: FittedDistribution | None = None


@dataclass
class PriorLabResult:
    experts: list[ExpertPrior] = field(default_factory=list)
    aggregated: AggregatedPrior | None = None
    parameter_name: str = "theta"
    parameter_description: str = ""
    export_json: dict = field(default_factory=dict)
    input_hash: str = ""
    certification: str = ""
