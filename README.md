# ForgeX - shared-head SEM restoration

Team ForgeX: Abishek SR, Anmol BA, Hardik - VIT Vellore.
SEMICON India 2026, KLA PS01.

The default `models/model.pt` is now the completed **120-epoch shared-head model**, selected from `mx120-s0-shared/best_frontier.pt`. The later combined-data fine-tuning attempt is excluded.

## Run

```bash
pip install -r requirements-inference.txt
python run.py INPUT_DIR OUTPUT_DIR
python run.py INPUT_DIR OUTPUT_DIR --knob 1
python run.py INPUT_DIR OUTPUT_DIR --knob 5 --no-trt
```

On the matching H100/TensorRT environment, the default runner automatically uses the bundled engine for the selected exit. Missing or incompatible TensorRT falls back to the same shared model in PyTorch. CPU inference remains available. `--no-trt` explicitly disables acceleration. Existing `--depth`, `--weights`, `--tta`, batch, timing and CPU options remain accepted. TTA, timing/profile and explicit precision modes use PyTorch. The minimal PyTorch package remains self-contained in `run.py` and `models/model.pt`.

Input: grayscale float32 `.npy`, HxW or HxWx1. Output: float32 at twice the input height and width, same filename/channel convention, finite and clamped to [0,1]. A dataset directory containing `NoisyLR/` is accepted. The PyTorch path retains its previous spatial-size support; bundled TensorRT supports square inputs 32/128/256/512. Unsupported TensorRT sizes use PyTorch.

## Five trained operating points

| Knob | Blocks executed | PSNR on 297 images | H100 warm GPU ms, input 128 |
|---|---:|---:|---:|
| 1 | 3 | 23.56245 | 0.3697 |
| 2 | 6 | 23.65255 | 0.5483 |
| 3 | 10 | 23.75688 | 0.7906 |
| 4 | 13 | 23.80755 | 0.9717 |
| 5 | 16 | 23.82509 | 1.1480 |

One 1,346,360-parameter backbone uses a shared reconstruction head, supervised at depths 3/6/10/13/16. These are five compiled inference graphs, not five degradation generators. `--depth` retains the older interface; only these five exits have the new measured quality table and engine builds.

`python run.py --list-knob` displays the current datasheet. `--budget-ms` uses its H100/128/batch-1 warm GPU figures as estimates; it is not an end-to-end deadline and must be recalibrated for other hardware, sizes or backends.

## TensorRT deployment

The repository includes all five native H100 engines, the selected training checkpoint, inference weights, ONNX exports, architecture, build manifest and validation records in [`deployment/h100_shared`](deployment/h100_shared/README.md).

The recorded environment is **H100 80GB HBM3, PyTorch 2.8.0+cu128, CUDA 12.8, TensorRT 10.13.3.9**. For that Linux/CUDA target:

```bash
pip install -r requirements-h100.txt
python deployment/h100_shared/verify_bundle.py
python run.py INPUT_DIR OUTPUT_DIR --knob 5
```

Native engine compatibility depends on the GPU, platform and TensorRT runtime. Other environments may need a rebuild. No universal portability is claimed. To require TensorRT and fail visibly if it cannot run, use the bundle's strict runner:

```bash
python deployment/h100_shared/run.py INPUT_DIR OUTPUT_DIR --engines deployment/h100_shared/models_native_r1 --knob 5 --batch 8 --backend tensorrt
```

## Evidence and verification

- [`docs/shared_promotion.json`](docs/shared_promotion.json): local CPU re-score before promotion: 297 images, 23.8249 dB; the recorded H100-era shared score is 23.8251 dB. Root and reference architectures agree exactly at all five exits.
- [`multiexit_297.csv`](deployment/h100_shared/evidence/multiexit_297.csv): all control/shared/adapter checkpoint comparisons.
- [`warm_latency.csv`](deployment/h100_shared/evidence/warm_latency.csv): H100 batch-1 timing. Full-depth GPU inference is 2.71x / 3.26x faster at input 128 / 256 than PyTorch FP32 with TF32 disabled. These exclude startup and file I/O; no reliable end-to-end speedup is asserted.
- Original [engine manifest](deployment/h100_shared/models_native_r1/manifest.json) and [fallback validation](deployment/h100_shared/models_native_r1/fallback_validation.json).
- The 100-image numerical comparison agreed within 0.00028 dB aggregate mean PSNR across backends; that sample comes from organizer-provided training data and is not a new independent test set.

```bash
python tools/verify_shipped.py /path/to/297_image_dataset
python tools/package_check.py --data /path/to/297_image_dataset
python -m unittest discover -s tools -p test_shared_release.py
```

The checkpoint SHA-1 is `40add39d927e5c265a7adbee22321932f7c48791`. The source training checkpoint SHA-256 is `cf66f9f10f5dbfd517b8758bcd9a56daafa91b0dbf28b99c65626c5b1f67f8ea`. Checkpoints and engines are matched by hashes. Historical Round 1/2 and previous gate-release documents retain their original measurements and are not descriptions of the current default model.

Previous release details: [historical gate README](docs/PREVIOUS_GATE_README.md).
