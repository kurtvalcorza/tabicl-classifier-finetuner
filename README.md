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
