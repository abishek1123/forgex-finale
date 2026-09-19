# Current release state

See [RELEASE_STATUS](docs/RELEASE_STATUS.md) for the model, hashes, five exits, runtime requirements and evidence boundaries.

The selected release is the original completed shared-head run, not the skipped combined-data fine-tune. `models/model.pt` is the matching inference-only checkpoint. Five native H100 engines and all needed architecture files are committed under `deployment/h100_shared`.

Current source of quality: `deployment/h100_shared/evidence/multiexit_297.csv`.
Current source of measured latency: `deployment/h100_shared/evidence/warm_latency.csv`.
Checkpoint promotion verification: `docs/shared_promotion.json`.

The H100 pod has been terminated. Historical H100 validation exists; a fresh root-runner hardware acceptance run remains unavailable. Local CPU and laptop CUDA checks are separate evidence.

Earlier chronological notes are preserved in [history](docs/history/previous_HACKATHON_CONTEXT.md). They describe earlier release states and do not override this page.
