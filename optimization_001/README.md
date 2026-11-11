# SwapGrasp optimization experiments

This isolated copy evaluates sampling, hard same-image ranking, and a frozen-detector candidate scorer.
See OPTIMIZATION.zh-CN.md for the fixed protocol, limitations, and inference command.
Detector runs and reports live in ../runs/optimization_001; the original submission remains in ../runs/submission_001.

Two matched seeds, 3500 steps per detector, RGB plus text inference, with loss curves and TensorBoard logs.
The new holdout excludes this round's training images, but originated from the previous training pool.
These are diagnostic experiments rather than an official benchmark or independent confirmation.

Completed mean Top-1: original 50.38%, balanced 50.00%, hard_swap 49.70%, reranked 50.66%.
Reranking improves balanced by 0.66 percentage points on average, with incomplete statistical support.
Against original, the 0.27-point mean increase is inconclusive; Top-5 and paired success decline.
All runs, stricter thresholds, confidence intervals and limitations are retained in the report.

Run the relevant tests with ../.venv/bin/python -m pytest -q tests/test_core.py tests/test_data_model.py tests/test_optimization.py tests/test_extended_metrics.py tests/test_prompt_specificity.py tests/test_adapter.py.
The source contains legacy helper scripts and REPRODUCE.zh-CN.md from submission_001; OPTIMIZATION.zh-CN.md documents this optimization run.
