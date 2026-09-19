# ForgeX project instructions

Read [docs/RELEASE_STATUS.md](docs/RELEASE_STATUS.md), [HACKATHON_CONTEXT.md](HACKATHON_CONTEXT.md) and the root README for the current deployment. [docs/HANDOFF.md](docs/HANDOFF.md) retains the historical research record.

The default is the trained shared-head five-exit model. Checkpoint SHA-1: `40add39d927e5c265a7adbee22321932f7c48791`. Exits: 3/6/10/13/16. Do not substitute the Round 2 checkpoint or the previous gate checkpoint. Use the CSVs and release-status page for current quality and performance values.

## Working rules

- Every numerical claim must identify its dataset, checkpoint and evidence file. Historical test sets are not interchangeable.
- Change `models/model.pt` only through the guarded `tools/swap.py` workflow. The engine bundle and checkpoint must remain matched by hashes.
- Do not silently re-key old engines to new weights. Preserve the original build manifest.
- Do not retrain or rebuild engines unless the user asks or a demonstrated compatibility issue requires it.
- Run `tools/verify_shipped.py`, `tools/package_check.py --data DATASET` and relevant release tests before shipping code changes.
- Use an activated virtual environment. Python 3.12 is the recorded H100 environment. `requirements.txt` is inference-only; `requirements-h100.txt` is for the matching Linux H100 runtime; optional scoring/training dependencies are in `requirements-training.txt`.
- Respect the user's explicitly selected repository and branch. No force-push, no push-all, no rewriting published history. Explicit release instructions override the historical branch workflow in CONTRIBUTING.md.
- Never commit training images, private keys, credentials, caches or arbitrary experimental checkpoints. The explicitly published shared checkpoint, five engines and matching frozen bundle are the exception already selected for this release.
- No vendor or assistant attribution in repository content or commits.

## Evaluation entry point

`python run.py INPUT OUTPUT` keeps same filenames, writes float32 finite clamped outputs at 2x size, and preserves an input singleton channel. `--knob 1..5` selects the five trained exits. Use `--require-trt` to prove actual engine use and `--no-trt` to force PyTorch. Do not describe a successful fallback as TensorRT acceleration.
