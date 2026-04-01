import hashlib
import json


def compute_input_hash(quantiles_list, expert_labels):
    data = []
    for q, label in zip(quantiles_list, expert_labels):
        data.append({"label": label, "median": q.median, "q1": q.q1, "q3": q.q3})
    raw = json.dumps(data, sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def certify(experts):
    if not experts:
        return "REJECT"
    if any(e.best_fit is None for e in experts):
        return "WARN"
    return "PASS"
