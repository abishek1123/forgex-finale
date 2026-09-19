"""Optional, checkpoint-bound H100 acceleration for the existing folder runner."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

_runtime = None
_stream = None
_failed = set()
_identity_cache = {}
DEPTHS = (3, 6, 10, 13, 16)


def digest(path):
    stat = path.stat()
    key = (str(path), stat.st_size, stat.st_mtime_ns)
    if key not in _identity_cache:
        _identity_cache[key] = hashlib.sha256(path.read_bytes()).hexdigest()
    return _identity_cache[key]


def try_restore(model, arrays, options, root):
    """Return owned CPU arrays, or None to use the caller's same-depth model."""
    global _runtime, _stream
    depth = len(model.body)
    required = getattr(options, 'require_trt', False)
    if options.no_trt or options.half or not options.fp16 or options.profile:
        return None
    if depth not in DEPTHS or depth in _failed or not arrays:
        if required:
            raise RuntimeError('Required TensorRT exit is unavailable or the batch is empty')
        return None
    shape = arrays[0].shape
    if len(shape) != 2 or shape[0] != shape[1] or shape[0] not in (32, 128, 256, 512):
        if required:
            raise RuntimeError('Required TensorRT engines support square inputs 32/128/256/512 only')
        return None
    directory = Path(root) / 'deployment/h100_shared/models_native_r1'
    weights = Path(options.weights) if options.weights else Path(root) / 'models/model.pt'
    try:
        # A user-supplied checkpoint must never accidentally run another set of engines.
        if digest(weights) != digest(directory / 'model_weights.pt'):
            if required:
                raise RuntimeError('Selected weights do not match the TensorRT deployment weights')
            return None
        import torch
        import tensorrt as trt
        manifest = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
        if trt.__version__ != manifest['identity']['tensorrt']:
            raise RuntimeError('TensorRT version differs from the validated engine build')
        if torch.cuda.get_device_capability(0) != tuple(manifest['identity']['compute_capability']):
            raise RuntimeError('GPU compute capability differs from the native engine build')
        if _runtime is None:
            package = directory.parent
            sys.path.insert(0, str(package))
            try:
                spec = importlib.util.spec_from_file_location('_forgex_shared_runtime', package / 'knob_runtime.py')
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
            finally:
                sys.path.pop(0)
            _runtime = module.ForgeXKnob(directory)
            _stream = torch.cuda.Stream()
        limit = {32: 1, 128: 16, 256: 8, 512: 1}[shape[0]]
        outputs = []
        with torch.cuda.stream(_stream):
            for start in range(0, len(arrays), limit):
                restored, info = _runtime.restore_batch(arrays[start:start + limit],
                    DEPTHS.index(depth) + 1, backend='tensorrt')
                outputs.extend(restored)
        _stream.synchronize()
        if depth not in getattr(try_restore, '_announced', set()):
            print(f'  backend=tensorrt shared depth={depth}; native H100 engine', flush=True)
            try_restore._announced = getattr(try_restore, '_announced', set()) | {depth}
        return outputs
    except Exception as exc:
        if required:
            raise RuntimeError(f'Required TensorRT inference failed at depth {depth}: {exc}') from exc
        _failed.add(depth)
        print(f'  TensorRT unavailable at depth {depth}; using same-depth PyTorch: {type(exc).__name__}: {exc}',
              file=sys.stderr, flush=True)
        return None
