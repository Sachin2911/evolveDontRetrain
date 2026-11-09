"""Common-pool retention and one fixed contextual selector per unique member."""

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit


def quality(candidate):
    rows = candidate["outcomes"]
    if not candidate["complete"] or not rows:
        raise ValueError("Partial candidate panels cannot be nominated")
    return (
        sum(r["safe_completion"] for r in rows),
        sum(r["success"] for r in rows),
        -sum(r["violation"] for r in rows),
    )


def useful_safe(row):
    return (
        row["condition"] != "clean"
        and row["safe_completion"]
        and not row["initial_success"]
        and not row["descriptor"]["stationary"]
    )


def retain(candidates, native_outcomes, *, minimum_safe=2, clean_loss=0):
    native_clean = sum(r["success"] for r in native_outcomes if r["condition"] == "clean")
    eligible = []
    for candidate in candidates:
        if not candidate["complete"]:
            continue
        rows = candidate["outcomes"]
        safe = sum(useful_safe(r) for r in rows)
        clean = sum(r["success"] for r in rows if r["condition"] == "clean")
        if safe >= minimum_safe and clean >= native_clean - clean_loss:
            eligible.append(candidate)
    ordered = sorted(eligible, key=lambda c: (tuple(-v for v in quality(c)), c["id"]))
    if len(ordered) < 3:
        return {
            "passed": False,
            "reason": "insufficient useful candidates",
            "eligible": len(ordered),
        }
    diverse = [ordered[0]]
    remaining = ordered[1:]
    while len(diverse) < 3:

        def order(candidate):
            distance = min(
                np.linalg.norm(
                    np.asarray(candidate["descriptor"]) - np.asarray(member["descriptor"])
                )
                for member in diverse
            )
            return (-float(distance), tuple(-v for v in quality(candidate)), candidate["id"])

        chosen = min(remaining, key=order)
        diverse.append(chosen)
        remaining = [c for c in remaining if c["id"] != chosen["id"]]
    score_ids = ["native"] + [c["id"] for c in ordered[:3]]
    diverse_ids = ["native"] + [c["id"] for c in diverse]
    return {
        "passed": True,
        "eligible": len(ordered),
        "score_only": score_ids,
        "diverse": diverse_ids,
        "union": sorted(set(score_ids + diverse_ids)),
    }


def fit_predictor(contexts, labels, regularization=1.0):
    x, y = np.asarray(contexts, dtype=float), np.asarray(labels, dtype=float)
    if x.ndim != 2 or y.shape != (len(x),) or not len(x):
        raise ValueError("Invalid calibration matrix")
    if not np.isfinite(x).all() or not np.isin(y, [0, 1]).all():
        raise ValueError("Invalid calibration values")
    if np.all(y == y[0]):
        return {
            "kind": "constant",
            "probability": float((y.sum() + 1) / (len(y) + 2)),
            "episodes": len(y),
            "features": x.shape[1],
        }
    design = np.column_stack([np.ones(len(x)), x])

    def objective(weights):
        logits = design @ weights
        penalty = regularization * np.dot(weights[1:], weights[1:]) / 2
        loss = np.sum(np.logaddexp(0, logits) - y * logits) + penalty
        gradient = design.T @ (expit(logits) - y)
        gradient[1:] += regularization * weights[1:]
        return float(loss), gradient

    result = minimize(
        objective,
        np.zeros(design.shape[1]),
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": 1000, "ftol": 1e-12, "gtol": 1e-8},
    )
    if not np.isfinite(result.x).all():
        raise ValueError("Selector fit failed: " + str(result.message))
    # The penalized objective is strongly convex. A line-search stop at the optimum
    # is accepted only when the full gradient confirms convergence.
    gradient = float(np.max(np.abs(objective(result.x)[1])))
    if not result.success and gradient > 1e-5:
        raise ValueError("Selector fit failed: " + str(result.message))
    return {
        "kind": "logistic",
        "weights": result.x.tolist(),
        "episodes": len(y),
        "features": x.shape[1],
        "regularization": regularization,
        "optimizer_message": str(result.message),
        "gradient_max_abs": gradient,
    }


def predict(predictor, context):
    x = np.asarray(context, dtype=float)
    if x.shape != (predictor["features"],) or not np.isfinite(x).all():
        raise ValueError("Selector context differs from the frozen contract")
    if predictor["kind"] == "constant":
        return predictor["probability"]
    return float(expit(np.dot(predictor["weights"], np.r_[1.0, x])))


def select(members, predictors, context):
    # Native wins an exact tie; remaining ties use stable candidate identity.
    return min(members, key=lambda m: (-predict(predictors[m], context), m != "native", m))
