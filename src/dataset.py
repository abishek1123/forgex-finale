"""Datasets. Owner: Person A (Data).

RestoreDataset mixes KLA's real NoisyLR files with pairs we generate ourselves
via degrade(). p_real controls the blend. Synthetic pairs are re-rolled every
time an image is drawn, so there is no fixed noise pattern to memorise.
"""
import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset

from degrade import degrade, degrade_wide, TRAIN, WIDE


def list_ids(d):
    return sorted(os.path.splitext(os.path.basename(p))[0] for p in glob.glob(os.path.join(d, "*.npy")))


def make_block_split(gt_dir, lo, hi):
    """Hold out a CONTIGUOUS range of ids -- a genuine out-of-distribution test.

    The ids in semicon_train_data are ordered by content: median f90 (structure
    fineness) runs from 0.37 in the 478-955 block down to 0.12 in the 2868-3345
    block, a 3x range, while the degradation is uniform throughout. So a
    contiguous block is a distinct set of morphologies.

    make_split() draws uniformly at random, which puts a proportional slice of
    every content block on BOTH sides of the split -- the model has then seen
    examples of everything it is tested on, and the score measures interpolation
    rather than generalisation. This function does not, so the model is scored on
    structure it has never encountered. KLA state the hidden test set contains
    out-of-distribution samples from different sources; this is our only way to
    estimate what that costs us before it arrives.
    """
    ids = list_ids(gt_dir)
    val = [i for i in ids if lo <= int(i) <= hi]
    train = [i for i in ids if not (lo <= int(i) <= hi)]
    return train, val


def make_split(gt_dir, n_val=200, seed=0):
    """Deterministic train/val split. Everyone must use the same seed."""
    ids = list_ids(gt_dir)
    perm = np.random.default_rng(seed).permutation(len(ids))
    val = {ids[i] for i in perm[:n_val]}
    return [i for i in ids if i not in val], sorted(val)


def _dihedral(a, k):
    if k & 1:
        a = a[:, ::-1]
    if k & 2:
        a = a[::-1, :]
    if k & 4:
        a = a.T
    return np.ascontiguousarray(a)


class RestoreDataset(Dataset):
    """Yields (lr, hr) float32 tensors: (1, crop, crop) and (1, 2*crop, 2*crop)."""

    def __init__(self, gt_dir, lr_dir, ids, crop=64, p_real=0.3, cfg=TRAIN, seed=0,
                 length=None, wide_p=0.0, wide_cfg=WIDE, extra_gt=None, p_extra=0.0,
                 hard_w=0.0, stats_csv=None):
        self.gt_dir, self.lr_dir = gt_dir, lr_dir
        self.ids = list(ids)
        self.crop, self.p_real, self.cfg, self.seed = crop, p_real, cfg, seed
        # wide_p: of the SYNTHETIC draws, the fraction taken from the wide
        # out-of-distribution family instead of the calibrated one.
        self.wide_p, self.wide_cfg = float(wide_p), wide_cfg
        # extra_gt: clean images from ANOTHER source (NFFA-EUROPE), with no real
        # noisy counterpart. They exist to widen CONTENT, which is the one axis
        # the generator cannot reach -- it can re-damage an image endlessly but
        # it cannot invent a morphology. Always degraded synthetically, since
        # there is no real pair to load.
        self.extra_gt = extra_gt
        self.extra_ids = list_ids(extra_gt) if extra_gt else []
        self.p_extra = float(p_extra) if self.extra_ids else 0.0

        # CONTENT-BALANCED SAMPLING.
        # Content is the largest effect we have measured: coarse structure
        # scores +4.98 over bicubic, fine structure +2.42 -- a 2.56 dB spread on
        # identical noise. Uniform sampling therefore spends most of the budget
        # on the content we are already good at. hard_w tilts the sampler toward
        # FINE structure (high f90), which is where we are weak and where an
        # out-of-distribution test set is most likely to hurt.
        #   hard_w 0.0  uniform, as before
        #   hard_w 1.0  probability proportional to f90
        #   hard_w 2.0  proportional to f90 squared -- aggressive
        self.p_id = None
        if hard_w > 0 and stats_csv and os.path.isfile(stats_csv):
            import csv as _csv
            f90 = {}
            for r in _csv.DictReader(open(stats_csv)):
                try:
                    f90[r["id"]] = float(r["f90"])
                except Exception:
                    pass
            w = np.array([f90.get(i, 0.0) for i in self.ids], dtype=np.float64)
            if w.sum() > 0:
                w = np.maximum(w, 1e-6) ** float(hard_w)
                self.p_id = w / w.sum()
        self.length = int(length) if length else len(self.ids)

    def __len__(self):
        return self.length

    def _rng(self):
        if not hasattr(self, "_g"):
            info = torch.utils.data.get_worker_info()
            wid = 0 if info is None else info.id
            self._g = np.random.default_rng([self.seed, wid, os.getpid()])
        return self._g

    def __getitem__(self, idx):
        g = self._rng()
        from_extra = self.p_extra > 0 and g.random() < self.p_extra
        if from_extra:
            name = self.extra_ids[int(g.integers(len(self.extra_ids)))]
            gt = np.load(os.path.join(self.extra_gt, name + ".npy")).astype(np.float32)
        elif self.p_id is not None:
            name = self.ids[int(g.choice(len(self.ids), p=self.p_id))]
            gt = np.load(os.path.join(self.gt_dir, name + ".npy")).astype(np.float32)
        else:
            name = self.ids[idx % len(self.ids)]
            gt = np.load(os.path.join(self.gt_dir, name + ".npy")).astype(np.float32)
        if gt.ndim == 3:
            gt = gt[..., 0]
        c = self.crop
        H, W = gt.shape
        y = int(g.integers(0, H // 2 - c + 1))
        x = int(g.integers(0, W // 2 - c + 1))
        hr = gt[2 * y:2 * (y + c), 2 * x:2 * (x + c)]

        if (not from_extra) and self.lr_dir is not None and g.random() < self.p_real:
            lr = np.load(os.path.join(self.lr_dir, name + ".npy")).astype(np.float32)[y:y + c, x:x + c]
        elif self.wide_p and g.random() < self.wide_p:
            lr = degrade_wide(hr, g, self.wide_cfg)
        else:
            lr = degrade(hr, g, self.cfg)

        k = int(g.integers(8))
        lr, hr = _dihedral(lr, k), _dihedral(hr, k)
        return torch.from_numpy(lr)[None], torch.from_numpy(hr)[None]


class ValDataset(Dataset):
    """Full images, KLA's REAL NoisyLR only -- never synthetic. This is the
    number we trust, because it is measured on the actual degradation."""

    def __init__(self, gt_dir, lr_dir, ids):
        self.gt_dir, self.lr_dir, self.ids = gt_dir, lr_dir, list(ids)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        name = self.ids[i]
        lr = np.load(os.path.join(self.lr_dir, name + ".npy")).astype(np.float32)
        gt = np.load(os.path.join(self.gt_dir, name + ".npy")).astype(np.float32)
        return torch.from_numpy(lr)[None], torch.from_numpy(gt)[None], name
