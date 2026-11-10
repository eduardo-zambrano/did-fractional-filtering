#!/usr/bin/env python3
"""Exact finite-K illustration; Python 3 + NumPy only. No inference claims."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def weights(d: float, k: int) -> np.ndarray:
    p = np.ones(k + 1)
    for j in range(1, k + 1):
        p[j] = p[j - 1] * (j - 1 - d) / j
    return p


def gap(z: np.ndarray, g: np.ndarray) -> np.ndarray:
    nt = g.sum(axis=1)
    nc = (~g).sum(axis=1)
    if np.any(nt == 0) or np.any(nc == 0):
        raise RuntimeError("Empty treatment/control group; choose another design.")
    return (z * g).sum(axis=1) / nt - (z * ~g).sum(axis=1) / nc


def write_results_table(summaries: list[dict], destination: Path) -> None:
    """Write the appendix table from case-grouped simulation summaries."""
    selection_labels = {
        "A: selection on state": "Pretreatment state",
        "B: selection on future innovation": "Future innovation",
        "C: selection on permanent effect": "Permanent component",
    }
    lines = [r"\begin{tabular}{llrrrr}", r"\toprule",
             r"Selection on & Estimator & Mean & Bias & RMSE & MCSE(mean) \\", r"\midrule"]
    previous = None
    for s in summaries:
        case = s["case"]
        if previous is not None and previous != case:
            lines.append(r"\addlinespace")
        label = selection_labels[case] if case != previous else ""
        values = " & ".join(f"{float(s[key]):.3f}" for key in
                            ("mean", "bias", "rmse", "mcse_mean"))
        lines.append(f"{label} & {s['estimator']} & {values} " + r"\\")
        previous = case
    lines += [r"\bottomrule", r"\end{tabular}"]
    destination.write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--replications", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20261104)
    ap.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "output")
    args = ap.parse_args()
    n, k, d0, burn, pre, holdout, batch_size = 300, 10, .45, 400, 120, 20, 25
    tau, drift = 1., .02
    p = weights(d0, k)
    ar = -p[1:]
    grid = np.linspace(0., 1., 101)
    grid_p = np.stack([weights(float(d), k) for d in grid])
    rng = np.random.default_rng(args.seed)
    outcomes: dict[tuple[str, str], list[np.ndarray]] = {}
    rows = []
    d_hats, alpha_gaps = [], []
    max_recurrence_error = 0.
    total_t = burn + pre + 1
    # x=0 before the simulation; this initial condition does not affect the exact
    # finite-K innovation equation. Burn-in merely reduces start-up transients.
    for start in range(0, args.replications, batch_size):
        b = min(batch_size, args.replications - start)
        eps = rng.normal(size=(total_t, b, n))
        x = np.zeros_like(eps)
        for t in range(total_t):
            x[t] = eps[t]
            for j in range(1, min(k, t) + 1):
                x[t] += ar[j - 1] * x[t - j]
        observed_dates = np.arange(-pre + 1, 2)
        y0 = x[burn:] + drift * observed_dates[:, None, None]
        # A and B have no unit effect. These are OBSERVED, untreated outcomes.
        # Training dates -109,...,-20 have all K lags in the observed history.
        # Profile out a separate common time intercept at each training date.
        history = y0[:pre]
        centered = history - history.mean(axis=2, keepdims=True)
        lag_design = np.stack([centered[k - j:pre - holdout - j]
                               for j in range(k + 1)], axis=0)
        lag_design = lag_design.transpose(2, 0, 1, 3).reshape(b, k + 1, -1)
        gram = lag_design @ lag_design.transpose(0, 2, 1)
        sse = np.einsum("gj,bjl,gl->bg", grid_p, gram, grid_p)
        selected = sse.argmin(axis=1)
        dhat = grid[selected]
        d_hats.append(dhat)
        # d is now locked, before assignment is generated or post outcomes used.
        eta_a = rng.normal(size=(b, n))
        eta_b = rng.normal(size=(b, n))
        ga = x[-2] + eta_a >= 0
        gb = eps[-1] + eta_b >= 0
        end_lags = np.stack([y0[-1 - j] for j in range(k + 1)], axis=1)
        filtered = np.einsum("j,bjn->bn", p, end_lags)
        feasible = np.einsum("bj,bjn->bn", grid_p[selected], end_lags)
        # Algebraic check: filtering x must recover eps, up to roundoff.
        x_filtered = sum(p[j] * x[-1 - j] for j in range(k + 1))
        max_recurrence_error = max(max_recurrence_error,
                                  float(np.max(np.abs(x_filtered - eps[-1]))))
        transforms = {"Levels": y0[-1], "First differences": y0[-1] - y0[-2],
                      "Oracle fractional": filtered, "Feasible fractional": feasible}
        for case, group in (("A: selection on state", ga), ("B: selection on future innovation", gb)):
            for method, z in transforms.items():
                estimate = tau + gap(z, group)
                outcomes.setdefault((case, method), []).append(estimate)
                for r, est in enumerate(estimate):
                    rows.append((start + r + 1, case, method, float(est), float(dhat[r])))
        # C isolates the fixed-effect problem: assignment depends only on alpha
        # plus noise. No feasible-d estimator is claimed for this separate DGP.
        alpha = rng.normal(size=(b, n))
        gc = alpha + rng.normal(size=(b, n)) >= 0
        alpha_gaps.append(gap(alpha, gc))
        case_c = "C: selection on permanent effect"
        transforms_c = {"Levels": y0[-1] + alpha,
                        "First differences": y0[-1] - y0[-2],
                        "Oracle fractional": filtered + p.sum() * alpha}
        for method, z in transforms_c.items():
            estimate = tau + gap(z, gc)
            outcomes.setdefault((case_c, method), []).append(estimate)
            for r, est in enumerate(estimate):
                rows.append((start + r + 1, case_c, method, float(est), ""))
        print(f"Completed {start + b}/{args.replications}", flush=True)
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    with (out / "replication_estimates.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["replication", "case", "estimator", "estimate", "d_hat"])
        writer.writerows(rows)
    summaries = []
    for (case, method), parts in outcomes.items():
        v = np.concatenate(parts)
        err = v - tau
        mse = float(np.mean(err ** 2))
        rmse = np.sqrt(mse)
        summaries.append(dict(case=case, estimator=method, replications=len(v),
                              mean=float(v.mean()), bias=float(err.mean()),
                              rmse=float(rmse),
                              mcse_mean=float(v.std(ddof=1) / np.sqrt(len(v))),
                              mcse_rmse=float(np.std(err ** 2, ddof=1) /
                                              np.sqrt(len(v)) / (2 * rmse))))
    with (out / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    dvec = np.concatenate(d_hats)
    alpha_gap = float(np.concatenate(alpha_gaps).mean())
    config = dict(seed=args.seed, replications=args.replications, n=n, K=k, d0=d0,
                  burn_in=burn, observed_preperiods=pre, calibration_holdout=holdout,
                  calibration_dates=[-109, -20], calibration_outcomes=90,
                  d_grid=[0., 1., .01], batch_size=batch_size, tau=tau, drift=drift,
                  alpha_in_A_B=0, numpy_version=np.__version__,
                  filtered_constant_multiplier=float(p.sum()),
                  max_recurrence_error=max_recurrence_error,
                  mean_d_hat=float(dvec.mean()), sd_d_hat=float(dvec.std(ddof=1)),
                  rmse_d_hat=float(np.sqrt(np.mean((dvec - d0) ** 2))),
                  grid_boundary_frequency=float(np.mean((dvec == 0) | (dvec == 1))),
                  case_C_mean_alpha_gap=alpha_gap,
                  case_C_predicted_bias_from_sample_alpha_gap=float(p.sum() * alpha_gap),
                  case_C_population_bias=float(p.sum() * 2 / np.sqrt(np.pi)),
                  case_B_population_bias=float(2 / np.sqrt(np.pi)))
    (out / "configuration.json").write_text(json.dumps(config, indent=2) + "\n")
    write_results_table(summaries, out / "results_table.tex")
    assert max_recurrence_error < 1e-11, "DGP does not match exact finite-K filter"
    assert all(np.isfinite(s["mean"]) for s in summaries)
    print(json.dumps(config, indent=2))
    for s in summaries:
        print(f"{s['case'][0]} | {s['estimator']:20} | mean {s['mean']:.4f} | "
              f"bias {s['bias']:.4f} | RMSE {s['rmse']:.4f} | MCSE {s['mcse_mean']:.4f}")


if __name__ == "__main__":
    main()
