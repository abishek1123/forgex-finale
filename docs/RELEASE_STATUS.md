# Current shared-head release

The default model is `mx120-s0-shared/best_frontier.pt`, from the completed 120-epoch run. The selected validation checkpoint is epoch index 65. The later combined-data fine-tuning attempt is excluded.

- Architecture: shared head, 53 channels, 16 blocks, 1,346,360 parameters.
- Trained exits: 3, 6, 10, 13, 16 (`--knob 1..5`).
- Deployment checkpoint SHA-1: `40add39d927e5c265a7adbee22321932f7c48791`.
- Training checkpoint SHA-256: `cf66f9f10f5dbfd517b8758bcd9a56daafa91b0dbf28b99c65626c5b1f67f8ea`.
- Root deployment weights are byte-identical to `deployment/h100_shared/models_native_r1/model_weights.pt`.
- Recorded full-depth PSNR: 23.82509 dB on the 297-image set. Local CPU FP32 promotion re-score: 23.82490 dB. Sources: `deployment/h100_shared/evidence/multiexit_297.csv`, `docs/shared_promotion.json`.

## Runtime and engines

`python run.py INPUT OUTPUT` loads the shared model. Compatible CUDA requests use the matching native H100 engines; incompatible or missing TensorRT falls back to same-depth PyTorch. CPU works without TensorRT.

`python run.py INPUT OUTPUT --knob 5 --require-trt` is the strict hardware acceptance command. It fails on missing CUDA/TensorRT, incompatible versions, unsupported sizes, mismatched weights or engine errors. A successful strict run must print `backend=tensorrt`. `--no-trt` forces PyTorch. `--timing`, `--profile`, TTA and explicit precision modes use PyTorch and cannot be combined with `--require-trt`.

The bundled engines target Linux H100 80GB HBM3, CUDA 12.8, TensorRT 10.13.3.9 and PyTorch 2.8.0+cu128. They are native builds, not universal plans. Supported square input sizes are 32/128/256/512; profiles support up to batch 16 at 128, batch 8 at 256, batch 1 at 32/512. The root dispatcher splits larger batches to fit these profiles.

The original H100 numerical-build, fallback and sample-quality records are preserved in `deployment/h100_shared`. A new H100 hardware run of the updated root integration was not possible after the pod was terminated. A local hash or mocked-routing check does not establish engine deserialization on hardware.

## Evidence boundaries

The 100-image comparison from the 1,197 supplied pairs is a numerical equivalence check, not an independent generalization test. The 2.71x/3.26x speedups are warm GPU-only measurements at 128/256, batch 1. They exclude process startup and file I/O. No robust end-to-end speedup is claimed.

Round 1, Round 2, previous gate release and earlier TensorRT 11/H100 NVL records are historical. Their hashes, parameter counts, quality metrics and timings must not be presented as this shared release's results. See the current root README and this file before using an older document.
