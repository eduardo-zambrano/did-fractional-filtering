# Difference-in-differences with fractional filtering

Replication code and simulation results for Eduardo Zambrano’s *Difference-in-differences with fractional filtering* (2026). The original submission was titled *Fractional parallel trends*.

| In the online appendix | Experiment | Results |
| --- | --- | --- |
| Appendices B and C, Table C.1 | An exact finite-lag model with three treatment assignment rules and pretreatment calibration | [Full-precision table](exact_finite_k/output/summary.csv) · [LaTeX table](exact_finite_k/output/results_table.tex) |
| Appendix A.4 | The original truncation experiment with 20, 30, and 40 filter lags | [Results by lag length](fractional_did_mc_results_with_K_robustness.csv) |

## Reproduce Table C.1

Run this command from the repository root.

```sh
python3 exact_finite_k/run_exact_finite_k.py
```

The archived outputs have been reproduced with Python 3.12 and NumPy 2.3.5. The script uses seed `20261104`, runs 1,000 replications, and writes its results to `exact_finite_k/output/`. The first-period treatment effect is one in all three designs. They differ in whether treatment depends on the pretreatment state, the future innovation, or a permanent component. See the [experiment notes](exact_finite_k/README.md) for the assignment rules, calibration procedure, and output files.

## Reproduce the truncation results in Appendix A.4

Run this command from the repository root.

```sh
python3 run_fractional_did_mc.py
```

This experiment requires NumPy and pandas. It runs 5,000 replications with seed `20260502` and writes two CSVs in the working directory. Appendix A.4 uses the [results by lag length](fractional_did_mc_results_with_K_robustness.csv). Under selection on the pretreatment state, the estimated biases of the oracle fractional estimator are 0.044, 0.027, and 0.017 for 20, 30, and 40 lags. The [original main table](fractional_did_mc_results.csv) reports the 30-lag specification.

## Citation and license

```bibtex
@misc{zambrano_fractional_parallel_trends,
  author = {Zambrano, Eduardo},
  title  = {Difference-in-differences with fractional filtering},
  year   = {2026},
  note   = {Working paper}
}
```

This repository uses the [MIT license](LICENSE).
