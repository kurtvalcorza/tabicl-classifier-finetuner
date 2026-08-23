# DIMER fine-tuner contract

This repository owns the TabICLv2 **classification** fine-tuning worker. It does not modify DIMER Workbench orchestration.

## Normal execution

When `GPU_BURST_MODE` is not enabled, the container delegates directly to the existing `train.main()` contract:

- dataset: `DIMER_DATASET_DIR` (default `/data/dataset`)
- output: `DIMER_OUTPUT_DIR`
- result: `DIMER_RESULT_PATH`
- preprocessing: `DIMER_PREPROCESSING_ARGS_JSON`
- fine-tuning controls: `DIMER_HYPERPARAMETERS_JSON`
- completion: `DIMER_DONE_CALLBACK`

## Current on-prem GPU-burst compatibility

The current on-prem GPU path can provide MinIO/S3 coordinates without mounting `/data`. When `GPU_BURST_MODE=true`, the container entrypoint:

1. redirects dataset/output/result paths into `DIMER_BURST_SCRATCH_ROOT` (default `/dev/shm/dimer`);
2. downloads `GPU_BURST_DATASET_PREFIX` from `GPU_BURST_S3_BUCKET`;
3. runs the unchanged `train.py` core;
4. verifies every declared result artifact exists and matches its declared size;
5. uploads the complete output tree, including `best.ckpt`, `training_context.parquet`, `artifact.json`, evaluation/log/progress files;
6. optionally writes the checkpoint to `GPU_BURST_MODEL_KEY` as a legacy single-model compatibility alias;
7. uploads `GPU_BURST_RESULT_KEY` **last**, so the result object acts as the durable-bundle commit marker;
8. only then calls `DIMER_DONE_CALLBACK`.

`DIMER_BURST_MAX_SCRATCH_BYTES` bounds staged dataset plus produced output (default 4 GiB).

A publication failure converts an otherwise successful training run into a failure result; a model that exists only in ephemeral scratch is not reported as successful.

## TabICL is a model bundle

The deployable serving state is not the checkpoint alone. At minimum it requires:

- `checkpoints/best.ckpt`
- `training_context.parquet`
- `artifact.json`

`result.json.artifacts.modelArtifact` remains populated for current DIMER compatibility, but `best.ckpt` is not a self-contained TabICL deployment artifact. `artifact.json` plus its referenced context/checkpoint define the serving contract.

## Base-model selection: current boundary

The worker's default base is the repository-pinned TabICLv2 classifier checkpoint. An explicit `DIMER_BASE_MODEL_PATH` operator override is honored and recorded by digest/source.

The current DIMER fine-tuning job also supplies `DIMER_MODEL_CONFIG_JSON`, but this worker does not interpret that object as a checkpoint path. Therefore DIMER must not claim that an arbitrary UI-selected base model controls this worker unless the platform provides an actual mounted `DIMER_BASE_MODEL_PATH` (or a future agreed equivalent). No DIMER change is implemented here.

## DIMER-side follow-ups (documentation only)

DIMER should:

1. treat TabICL output as a first-class multi-file model bundle across export, promotion, download, and deployment;
2. transfer artifact roles generically from `result.json.artifacts` rather than assuming a YOLO-style `best.pt` is sufficient;
3. provision/enable an appropriate CUDA GPU execution path for this GPU-only fine-tuner;
4. define a real selected-base-model handoff if the Workbench UI exposes multiple base models for this pipeline;
5. version/validate fine-tuner result envelopes and artifact descriptors.

No `dimer-backend` changes are part of this repository change.
