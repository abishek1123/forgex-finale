"""Batched folder inference. GPU build validation is required before release."""
import argparse
from collections import defaultdict, Counter
import json
from pathlib import Path
import time
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument('input')
    p.add_argument('output')
    p.add_argument('--engines', required=True)
    p.add_argument('--knob', type=int, choices=range(1, 6), default=5)
    p.add_argument('--backend', choices=('auto','tensorrt','pytorch','cpu'), default='auto')
    p.add_argument('--batch', type=int, default=1)
    a = p.parse_args()
    if a.batch < 1:
        p.error('--batch must be positive')
    from knob_runtime import ForgeXKnob
    runtime = ForgeXKnob(a.engines)
    source, destination = Path(a.input).resolve(), Path(a.output).resolve()
    if source == destination or source in destination.parents:
        p.error('Output must be outside input directory')
    groups = defaultdict(list)
    start = time.perf_counter()
    for path in sorted(source.rglob('*.npy')):
        arr = np.load(path, allow_pickle=False)
        channel = arr.ndim == 3 and arr.shape[-1] == 1
        if channel:
            arr = arr[..., 0]
        groups[arr.shape].append((path, arr, channel))
    if not groups:
        raise RuntimeError('No .npy inputs found')
    records = []
    for shape, items in groups.items():
        for offset in range(0, len(items), a.batch):
            batch = items[offset:offset+a.batch]
            outputs, infos = runtime.restore_batch([item[1] for item in batch], a.knob, a.backend)
            for (path, _, channel), result, info in zip(batch, outputs, infos):
                target = destination/path.relative_to(source)
                target.parent.mkdir(parents=True, exist_ok=True)
                if channel:
                    result = result[..., None]
                temporary = target.with_suffix('.npy.tmp')
                with temporary.open('wb') as stream:
                    np.save(stream, result, allow_pickle=False)
                temporary.replace(target)
                records.append(dict(file=path.relative_to(source).as_posix(), **info))
    seconds = time.perf_counter()-start
    summary = dict(images=len(records), seconds=seconds, images_per_second=len(records)/seconds,
                   backends=dict(Counter(r['backend'] for r in records)),
                   fallback_images=sum(r['fallback'] for r in records),
                   timing_scope='folder read, inference and writes; excludes imports/runtime construction')
    report = destination.parent/(destination.name+'.report.json')
    report.write_text(json.dumps(dict(summary=summary, records=records), indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
