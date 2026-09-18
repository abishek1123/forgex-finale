# Superseded result files

These two CSVs were produced by an earlier version of `tools/noise_sweep.py`
against a **different test set**. Their bicubic column reads **24.150** at
sigma 0.05, where every current sweep reads **22.742**.

A baseline column that differs between two result files means the two files
cannot be compared — the baseline is the checksum. See
`docs/ENGINEERING_LOG.md` 11.2, which is the entry about this exact mistake.

They are kept because the log refers to them, and deleting the evidence for a
correction is not a correction. **Do not plot or quote them.** The valid
replacements are `docs/noise_sweep_preal1.csv` and
`docs/noise_sweep_long120.csv`.
