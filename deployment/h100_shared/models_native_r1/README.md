# ForgeX H100 deployment development package v3

Build with `bash build_h100.sh` on the prepared H100 pod. This uses the selected shared checkpoint in restored_backup and eight real LR samples from test1197 plus numerical probes. It is deployment verification, not full-dataset quality evaluation.

Five true prefixes: 3,6,10,13,16. Profiles: 128 squared batches 1–16, 256 squared batches 1–8, general 32–512 sides at batch one. Application accepts only square 32/128/256/512 inputs. Mixed precision keeps sensitive layers FP32; initial numerical thresholds are max absolute error 0.02 and RMSE 0.002, pending full quality evaluation. Build stops on failure. A separate `--fp32 --out models_fp32` build is available for diagnosis.

Portable build uses `build_engines.py --hardware ampere_plus --out models_portable`. Same runtime dependencies are required. No engine has yet been GPU-validated merely by creating this package.

Folder inference: `python run.py INPUT OUTPUT --engines models_native --knob 5 --batch 8 --backend tensorrt`. Strict TensorRT mode never conceals fallback. Auto mode retries batch failures one image at a time and can fall back to same-depth PyTorch. Each output is atomically replaced, names and singleton channels preserved. Reports are outside output folders.

This package still explicitly selects an engine directory. Automatic native/portable selection and evaluator-side rebuilding are pending final integration. The previous auto-build draft is not used here. PyTorch is imported on this development inference path; import-free final runner optimization is pending. Do not label this a completed submission.
