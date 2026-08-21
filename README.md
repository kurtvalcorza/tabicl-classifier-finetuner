# tabicl-classifier-finetuner

DIMER fine-tuner for TabICLv2 classification using `tabicl.FinetunedTabICLClassifier`.

Key behavior:

- requires CUDA; no silent zero-shot fallback
- pins `tabicl[finetune]==2.1.1`
- uses `tabicl-classifier-v2-20260212.ckpt`
- creates stratified validation when `val.csv` is absent
- stratified-caps training to the configured row limit
- scores `test.csv` when present, after checkpoint selection
- writes `best.ckpt`, `training_context.parquet`, and `artifact.json`
- reloads the fine-tuned checkpoint and predicts before reporting success
- records dataset and checkpoint hashes in `result.json`

`dimer-pipeline.json` defines the DIMER workbench fields. Pairs with `tabicl-classifier-dataset-validator`; full documentation is in `tabicl-classifier-pipeline`.

## DIMER Pipeline Builder integration

The repository builds directly with the **repository root as the Docker build
context**; the Dockerfile invokes root-level `train.py`. Root layout:
`Dockerfile`, `train.py`, `requirements.txt`, `README.md`, `dimer-pipeline.json`.

- Every `datasetPreprocessing` key is read from `DIMER_PREPROCESSING_ARGS_JSON`
  and every `modelFinetuning` key from `DIMER_HYPERPARAMETERS_JSON`, 1:1 — no
  manifest control is silently ignored (guarded by
  `tests/test_train.py::test_manifest_matches_env_consumption`).
- `model_id` is intentionally **not** a manifest parameter; the DIMER Base Model
  selection is authoritative.
- The image bakes `DIMER_TASK_TYPE=tabular_classification` as the fallback for
  DIMER Custom / Other pipelines.
