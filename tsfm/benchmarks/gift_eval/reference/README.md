# GIFT-Eval reference files

Copied from https://github.com/SalesforceAIResearch/gift-eval at commit
`69c919e87a5046013bab25c7826b82f6eb5fa2fd` (Apache-2.0, Copyright (c) 2024 Salesforce, Inc.):

- `dataset_properties.json` — `notebooks/dataset_properties.json` (domain, frequency, variates per dataset)
- `<model>.csv` — `results/<model>/all_results.csv` (official per-config scores)

Used by `src/tsfm/benchmarks/gift_eval.py` for config names and by `scripts/eval_gift.py`
for comparisons. Our re-implementation reproduces `Seasonal_Naive.csv` on all 97 configs
(MASE within 1e-7, ND within 0.1%).
