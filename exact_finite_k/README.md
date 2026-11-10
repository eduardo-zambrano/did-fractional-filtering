# Reproducing Table C.1

Appendix B describes pretreatment calibration. Table C.1 in Appendix C compares estimators under three assignment rules. The first two designs use calibration. All three use 300 units, a first-period treatment effect of one, and the same transient state process with ten lags and generating value `d0 = 0.45`.

## Run the simulation

The archived outputs have been reproduced with Python 3.12 and NumPy 2.3.5. Run these commands from this directory.

```sh
python3 -m pip install 'numpy==2.3.5'
python3 run_exact_finite_k.py
```

The default seed is `20261104` and the default number of replications is 1,000. The script writes the results into the adjacent `output/` directory. To reproduce the results separately, pass a new directory with `--output`.

| File | Contents |
| --- | --- |
| [summary.csv](output/summary.csv) | Full-precision means, biases, root mean squared errors (RMSEs), and Monte Carlo standard errors for all 11 estimator rows. |
| [results_table.tex](output/results_table.tex) | Formatted Table C.1. Typesetting the table requires the LaTeX package `booktabs`. |
| [replication_estimates.csv](output/replication_estimates.csv) | Individual estimates and, where calibration is used, the value of `d_hat` for each replication. |
| [configuration.json](output/configuration.json) | Simulation settings, calibration diagnostics, and analytical bias benchmarks. |

## Assignment and calibration

| Table C.1 label | Assignment rule | First-period implication |
| --- | --- | --- |
| Pretreatment state | Select on the date-0 state plus noise. | The untreated gap partially closes. The filter using the generating coefficients accounts for the predictable change. |
| Future innovation | Select on the date-1 innovation plus noise. | The groups’ mean innovations differ. All four estimators have population bias of about 1.128. |
| Permanent component | Add an independent permanent unit effect and select on that effect plus noise. | First differencing removes the unit effect. The oracle fractional filter retains part of the permanent gap and has population bias of about 0.245. |

The CSV case identifiers begin with A, B, and C in the order shown above. Each assignment rule uses independent standard normal noise.

After 400 burn-in dates, the simulation observes 120 untreated dates and one treatment date. The first two designs have no permanent unit effects. They share the same predictive model and pretreatment histories, so they use the same calibrated filter within each replication. Selection on the future innovation breaks filtered mean balance even when the predictive model is correctly specified.

Calibration holds the lag length at ten and minimizes pooled squared one-step prediction errors over 90 pretreatment dates (−109 through −20), removing a common mean at each date. It searches from zero to one in steps of 0.01 and chooses the smallest minimizer. The last 20 pretreatment dates are excluded from calibration. In the table, “Feasible fractional” uses this estimated filter and “Oracle fractional” uses the generating value. The permanent-component design uses the oracle to isolate the bias that remains even when the generating value is known.
