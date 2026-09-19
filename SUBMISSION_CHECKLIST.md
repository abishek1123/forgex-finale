# Shared-release submission checklist

- Default checkpoint matches SHA-1 `40add39d927e5c265a7adbee22321932f7c48791`.
- `python deployment/h100_shared/verify_bundle.py` verifies the frozen checkpoint, engine, source and ONNX bytes.
- `python tools/doctor.py` checks the current checkpoint and restores a synthetic image.
- `python tools/verify_shipped.py DATASET` checks the 297 outputs using true FP32 and a maximum absolute tolerance of 1e-5 across CPU/CUDA implementations.
- `python tools/package_check.py --data DATASET` checks the full folder contract and isolated minimal package.
- `python -m unittest discover -s tools -p test_shared_release.py` checks all exits and engine routing/failure behavior.
- On the actual matching H100: run `python tools/audit_h100.py INPUT_DIR OUTPUT_DIR` to require real TensorRT execution at all five exits. Local checks do not replace this hardware test.
- Keep the full repository for TensorRT use. Copying only `run.py` and `models/model.pt` retains PyTorch inference, not the optional compiled engines.
- Install matching Linux H100 dependencies from `requirements-h100.txt`; generic or CPU inference uses its corresponding requirements file.
- Use current metrics from `docs/RELEASE_STATUS.md`; older documents are research history.
- The later combined-data fine-tuning attempt is excluded.

Historical checklist: [previous checklist](docs/history/previous_SUBMISSION_CHECKLIST.md).
