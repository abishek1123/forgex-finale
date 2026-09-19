"""Explicit hardware acceptance test; never count PyTorch fallback as TensorRT."""
import argparse
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input_dir', type=Path)
    parser.add_argument('output_dir', type=Path)
    a = parser.parse_args()
    if platform.system() != 'Linux' or not torch.cuda.is_available() or 'H100' not in torch.cuda.get_device_name(0):
        raise SystemExit('This acceptance test requires a Linux NVIDIA H100. No fallback is accepted.')
    import tensorrt as trt
    if trt.__version__ != '10.13.3.9':
        raise SystemExit('Use requirements-h100.txt: TensorRT 10.13.3.9 is required.')
    subprocess.run([sys.executable, str(ROOT/'deployment/h100_shared/verify_bundle.py')], cwd=ROOT, check=True)
    source = a.input_dir/'NoisyLR' if (a.input_dir/'NoisyLR').is_dir() else a.input_dir
    files = sorted(source.glob('*.npy'))[:4]
    if not files:
        raise SystemExit('No input .npy images found')
    a.output_dir.mkdir(parents=True, exist_ok=True)
    report = dict(passed=False, gpu=torch.cuda.get_device_name(0), torch=torch.__version__,
                  cuda=torch.version.cuda, tensorrt=trt.__version__, exits=[])
    report_path = a.output_dir/'h100_acceptance.json'
    report_path.write_text(json.dumps(report, indent=2))
    with tempfile.TemporaryDirectory(prefix='forgex_h100_') as temporary:
        temp = Path(temporary)
        inputs = temp/'input'
        inputs.mkdir()
        for i, file in enumerate(files):
            shutil.copy2(file, inputs/f'real_{i:02d}.npy')
        rng = np.random.default_rng(42)
        for size in (32, 128, 256, 512):
            np.save(inputs/f'probe_{size}.npy', rng.random((size, size), dtype=np.float32))
        for knob, depth in enumerate((3, 6, 10, 13, 16), 1):
            for backend, flags in [('trt', ['--require-trt']), ('torch', ['--no-trt', '--no-fp16'])]:
                out = temp/f'{backend}_{depth}'
                command = [sys.executable, str(ROOT/'run.py'), str(inputs), str(out), '--knob', str(knob), '--batch', '8', *flags]
                result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
                log = result.stdout + result.stderr
                (a.output_dir/f'{backend}_depth{depth}.log').write_text(log)
                if result.returncode or (backend == 'trt' and 'backend=tensorrt' not in log):
                    raise RuntimeError(f'{backend} depth {depth} failed. Inspect saved log.')
            metrics = []
            for file in sorted(inputs.glob('*.npy')):
                x = np.load(file)
                y = np.load(temp/f'trt_{depth}'/file.name)
                ref = np.load(temp/f'torch_{depth}'/file.name)
                expected = (2*x.shape[0], 2*x.shape[1]) + x.shape[2:]
                if y.shape != expected or y.dtype != np.float32 or not np.isfinite(y).all() or y.min() < 0 or y.max() > 1:
                    raise RuntimeError(f'Invalid output contract: depth {depth}, {file.name}')
                delta = y.astype(np.float64) - ref.astype(np.float64)
                maximum, rmse = float(np.abs(delta).max()), float(np.sqrt(np.mean(delta**2)))
                if maximum > .02 or rmse > .002:
                    raise RuntimeError(f'Numerical parity failed: depth {depth}, {file.name}: {maximum}, {rmse}')
                metrics.append(dict(file=file.name, max_abs=maximum, rmse=rmse))
            report['exits'].append(dict(depth=depth, backend='tensorrt', files=metrics))
            report_path.write_text(json.dumps(report, indent=2))
            print(f'PASS depth={depth}: real inputs and 32/128/256/512 probes', flush=True)
    report['passed'] = True
    report_path.write_text(json.dumps(report, indent=2))
    print(report_path)


if __name__ == '__main__':
    main()
