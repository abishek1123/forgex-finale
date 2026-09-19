# Shared-head model and native H100 TensorRT engines

This bundle preserves the selected **original shared-head checkpoint** and the five TensorRT engines built and checked on the H100. The later combined-data fine-tuning attempt is excluded.

| Knob | Residual blocks executed | Engine |
|---|---:|---|
| 1 | 3 | `forgex_exit03_mixed_fp16.engine` |
| 2 | 6 | `forgex_exit06_mixed_fp16.engine` |
| 3 | 10 | `forgex_exit10_mixed_fp16.engine` |
| 4 | 13 | `forgex_exit13_mixed_fp16.engine` |
| 5 | 16 | `forgex_exit16_mixed_fp16.engine` |

Each prefix uses the trained shared reconstruction head. These are five inference engines, not five degradation generators. The full model contains 1,346,360 parameters.

## Files and identity

- `checkpoints/shared_best_frontier.pt`: original training checkpoint, including training state. SHA-256: `cf66f9f10f5dbfd517b8758bcd9a56daafa91b0dbf28b99c65626c5b1f67f8ea`.
- `models_native_r1/model_weights.pt`: inference-only weights and configuration exported from that checkpoint; used for same-depth PyTorch fallback.
- `models_native_r1/*.engine`: five compiled native H100 engines.
- `models_native_r1/manifest.json`: original checkpoint identity, architecture hashes, engine hashes, profiles and numerical build probes.
- `models_native_r1/source/arch`: the matching architecture source.
- `models_native_r1/*.onnx`: the five export graphs.
- `SHA256SUMS.json`: byte identities for every file shipped in this bundle.

The nested `models_native_r1` directory is an unchanged build snapshot, except that Python caches and an unused editor backup were omitted. Its original README and manifest status predate the subsequent sample quality and fallback checks. This README records the later state; no full 1,197-image evaluation is claimed.

## Run on the matching H100 environment

Recorded environment: Linux, NVIDIA H100 80GB HBM3, driver 580.126.09, PyTorch 2.8.0+cu128, CUDA 12.8, TensorRT 10.13.3.9. The complete environment inventory is `models_native_r1/requirements-lock.txt`; it includes unrelated pod packages and is a record, not a minimal installer.

From the repository root, with the prepared H100 Python environment activated:

```bash
python deployment/h100_shared/verify_bundle.py
python deployment/h100_shared/run.py /path/to/NoisyLR /path/to/restored --engines deployment/h100_shared/models_native_r1 --knob 5 --batch 8 --backend tensorrt
```

Use `--backend auto` to permit visible TensorRT-to-PyTorch fallback at the same knob depth, with CPU recovery if CUDA is unavailable. Use `--backend tensorrt` when measuring TensorRT: any backend failure raises instead of silently reporting a PyTorch run as TensorRT. A JSON report beside the output directory records actual backends and fallback counts.

Inputs are finite grayscale `.npy` arrays, shaped HxW or HxWx1. Supported square input sides are **32, 128, 256, 512**, with 2x output sides. TensorRT profiles support batches up to 16 at 128, up to 8 at 256, and batch 1 at 32/512. Use `--batch 1` for a folder containing 32/512 inputs. Mixed input sizes are grouped by shape.

These are **native H100** engines. GPU, platform or TensorRT incompatibility may require rebuilding. The fallback is provided; universal binary portability is not claimed. The runner imports PyTorch and loads folder inputs into host memory, so warm device timing is not cold-start or folder throughput.

The repository's root `models/model.pt` has been promoted to these inference weights. Root `run.py` recognizes the shared-head architecture and optionally routes compatible requests to these engines. The strict bundle runner above remains available for isolated TensorRT evaluation. Root integration was checked locally on CPU and with injected dispatch tests; the frozen engine arithmetic was validated earlier on the H100.

## Recorded validation

- All five exports passed 17 ONNX probes and subsequent TensorRT numerical probes. See the original manifest for individual errors and shapes.
- `models_native_r1/fallback_validation.json` records successful missing-engine/runtime-failure injections at all five depths and four supported sizes, plus a CPU-only case.
- The user-provided H100 quality log records 100 images at every depth for both backends, with zero unintended fallback. The largest absolute difference in aggregate mean PSNR was below 0.00028 dB. The sample is from organizer-provided training data; this is backend agreement evidence, not an independent generalization score.
- `evidence/multiexit_297.csv` records the earlier 297-image model comparison. The selected shared frontier checkpoint reaches 23.82509 dB at depth 16 on that set. Do not compare that directly with a different dataset's mean.
- `evidence/warm_latency.csv` is transcribed without rounding from the user-provided H100 timing output: batch 1, 30 warmups, 100 iterations, non-default CUDA stream; PyTorch FP32 with TF32 disabled versus TensorRT mixed FP16. Full-depth GPU medians are 3.11038 vs 1.14797 ms at 128 (**2.71x**) and 9.26171 vs 2.83704 ms at 256 (**3.26x**).
- Fresh-process folder measurements were variable. No reliable end-to-end speedup is asserted from those measurements.

## Rebuild only when needed

Use a **new output directory**; do not overwrite the validated artifact directory. The builder expects a paired 1,197-image directory for its real-image probes.

```bash
cd deployment/h100_shared
python build_engines.py --root models_native_r1/source --ckpt checkpoints/shared_best_frontier.pt --data /path/to/test1197 --hardware native --out models_rebuilt
```

The builder is the exact version used on the H100, including the optimization-profile return-value fix. An `ampere_plus` option exists in the builder, but that mode is not the set of engines preserved here and has not been validated by these results.

## Local checks

```bash
cd deployment/h100_shared
python verify_bundle.py
python -m unittest test_profiles test_routing
```

The routing tests use injected backends when PyTorch is absent; they validate routing behavior, not GPU arithmetic. GPU validation evidence comes from the H100 records above.
