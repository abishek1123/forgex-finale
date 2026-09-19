"""Guarded promotion of the selected shared checkpoint and its matching engines."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    bundle = ROOT / 'deployment/h100_shared'
    source = bundle / 'models_native_r1/model_weights.pt'
    checkpoint = bundle / 'checkpoints/shared_best_frontier.pt'
    manifest = json.loads((bundle / 'models_native_r1/manifest.json').read_text())
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == manifest['identity']['checkpoint_sha256']
    deployed = torch.load(source, map_location='cpu', weights_only=True)
    trained = torch.load(checkpoint, map_location='cpu', weights_only=False)
    assert deployed['config'] == trained['config']
    assert deployed['state_dict'].keys() == trained['state_dict'].keys()
    assert all(torch.equal(v, trained['state_dict'][k]) for k, v in deployed['state_dict'].items())
    model, name = run.build_model(deployed['config'], deployed['state_dict'])
    assert name == 'shared_gate'
    model.load_state_dict(deployed['state_dict'], strict=True)
    model.eval()
    sys.path.insert(0, str(bundle))
    from knob_runtime import ForgeXKnob
    reference = ForgeXKnob(bundle / 'models_native_r1').load_model('cpu')
    torch.manual_seed(7)
    probe = torch.rand(1, 1, 32, 32)
    body = model.body
    with torch.inference_mode():
        for depth in (3, 6, 10, 13, 16):
            model.body = torch.nn.Sequential(*list(body)[:depth])
            error = (model(probe) - reference.forward_depth(probe, depth)).abs().max().item()
            assert error <= 1e-6, (depth, error)
            print(f'PASS root/shared reference depth={depth} max_abs={error}', flush=True)
        model.body = body
        ids = sorted((args.data / 'GT').glob('*.npy'))
        assert len(ids) == 297, 'Promotion requires the recorded 297-image dataset'
        psnrs = []
        for i, gt_path in enumerate(ids, 1):
            x = torch.from_numpy(np.load(args.data / 'NoisyLR' / gt_path.name).astype(np.float32))[None, None]
            gt = torch.from_numpy(np.load(gt_path).astype(np.float32))[None, None]
            y = model(x).clamp(0, 1)
            mse = float((y - gt).square().mean())
            psnrs.append(-10 * np.log10(max(mse, 1e-12)))
            if i % 50 == 0:
                print(f'Promotion score: {i}/297', flush=True)
    with (bundle / 'evidence/multiexit_297.csv').open() as stream:
        row = next(r for r in csv.DictReader(stream) if r['id'] == 'shared-frontier' and r['depth'] == '16')
    score, expected = float(np.mean(psnrs)), float(row['psnr'])
    assert abs(score - expected) < 0.003, (score, expected)
    destination = ROOT / 'models/model.pt'
    previous = ROOT / 'checkpoints/pre_shared_model.pt'
    previous.parent.mkdir(exist_ok=True)
    if destination.exists() and not previous.exists():
        shutil.copyfile(destination, previous)
    temp = destination.with_suffix('.pt.tmp')
    shutil.copyfile(source, temp)
    os.replace(temp, destination)
    record = dict(model='mx120-s0-shared/best_frontier.pt', n=297, psnr=score,
                  expected_psnr=expected, deploy_sha1=hashlib.sha1(destination.read_bytes()).hexdigest(),
                  checkpoint_sha256=manifest['identity']['checkpoint_sha256'],
                  method='CPU FP32, four threads, clamped output; root model agrees with reference at all five exits')
    (ROOT / 'docs/shared_promotion.json').write_text(json.dumps(record, indent=2) + '\n')
    print('PROMOTED:', json.dumps(record), flush=True)


if __name__ == '__main__':
    main()
