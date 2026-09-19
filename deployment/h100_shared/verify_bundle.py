"""Verify the frozen H100 bundle without importing CUDA, PyTorch or TensorRT."""
import hashlib
import json
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    root = Path(__file__).resolve().parent
    recorded = json.loads((root / 'SHA256SUMS.json').read_text())
    for name, expected in recorded.items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise RuntimeError(f'Missing or invalid bundle path: {name}')
        if sha256(path) != expected:
            raise RuntimeError(f'Checksum mismatch: {name}')
    model_dir = root / 'models_native_r1'
    manifest = json.loads((model_dir / 'manifest.json').read_text())
    identity = manifest['identity']
    if sha256(root / 'checkpoints/shared_best_frontier.pt') != identity['checkpoint_sha256']:
        raise RuntimeError('Checkpoint does not match the engine build')
    if set(manifest['engines']) != {'3', '6', '10', '13', '16'}:
        raise RuntimeError('Expected all five exit engines')
    for entry in manifest['engines'].values():
        for kind in ('engine', 'onnx'):
            if sha256(model_dir / entry[kind]) != entry[kind + '_sha256']:
                raise RuntimeError(f'Build manifest mismatch: {entry[kind]}')
    for name, expected in identity['source_hashes'].items():
        if sha256(model_dir / 'source/arch' / name) != expected:
            raise RuntimeError(f'Architecture mismatch: {name}')
    for name, field in [('knob_runtime.py', 'runtime_sha256'), ('build_engines.py', 'builder_sha256')]:
        if sha256(root / name) != identity[field]:
            raise RuntimeError(f'Build source mismatch: {name}')
    print(f'PASS: {len(recorded)} files; original shared checkpoint and all five matching H100 engines.')
    print(f'Target: {identity["gpu"]}; TensorRT {identity["tensorrt"]}')


if __name__ == '__main__':
    main()
