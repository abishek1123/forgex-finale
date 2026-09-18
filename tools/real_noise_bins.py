#!/usr/bin/env python3
"""Which model handles varying noise better -- measured on REAL data only.

    python tools\real_noise_bins.py --data <root> --ckpt a.pt --ckpt b.pt

noise_sweep.py builds its inputs with OUR generator, which favours whichever
model trained on that generator. This does not: it fits the per-image noise
level of KLA's OWN NoisyLR files, splits the held-out validation set into
terciles by that measured sigma, and scores each checkpoint in each bin.
No synthetic data is involved anywhere, so neither model is on home ground.
"""
import argparse, os, sys
import numpy as np, torch, torch.nn.functional as F

HERE=os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0,os.path.join(HERE,"..")); sys.path.insert(0,os.path.join(HERE,"..","src"))
from src.model import Restorer
from src.degrade import box_down
from dataset import make_split
from metrics import ssim as m_ssim, lpips as m_lpips

def f90(img, q=0.90):
    """Radial frequency below which q of the non-DC energy sits. High = fine structure."""
    H, W = img.shape
    P = np.abs(np.fft.fftshift(np.fft.fft2(img - img.mean()))) ** 2
    yy, xx = np.mgrid[0:H, 0:W]
    r = np.sqrt(((yy - H/2)/H)**2 + ((xx - W/2)/W)**2)
    o = np.argsort(r.ravel()); c = np.cumsum(P.ravel()[o]); c /= c[-1]
    return float(r.ravel()[o][np.searchsorted(c, q)])


def fit_sigma(clean, resid, nb=18):
    c,r=clean.ravel(),resid.ravel()
    idx=np.digitize(c,np.linspace(0,1,nb+1))-1
    xs,ys=[],[]
    for b in range(nb):
        m=idx==b
        if m.sum()>=300: xs.append(c[m].mean()); ys.append(r[m].var())
    if len(xs)<5: return None
    xs,ys=np.array(xs),np.array(ys)
    s,*_=np.linalg.lstsq(np.vstack([np.ones_like(xs),xs**2]).T,ys,rcond=None)
    return float(np.sqrt(max(s[1],0)))          # sigma_mul

p=argparse.ArgumentParser()
p.add_argument("--data",required=True); p.add_argument("--ckpt",action="append",required=True)
p.add_argument("--bin-by",choices=["sigma","f90"],default="sigma",
               help="sigma = noise level (robustness to noise). f90 = structure fineness (robustness to content).")
p.add_argument("--n-val",type=int,default=200); p.add_argument("--seed",type=int,default=0)
p.add_argument("--device",default="cuda" if torch.cuda.is_available() else "cpu")
a=p.parse_args()
dev=torch.device(a.device)
G,L=os.path.join(a.data,"GT"),os.path.join(a.data,"NoisyLR")
_,val=make_split(G,n_val=a.n_val,seed=a.seed)

# measure each validation image's OWN noise level
rows=[]
for i in val:
    gt=np.load(os.path.join(G,i+".npy")).astype(np.float32)
    lr=np.load(os.path.join(L,i+".npy")).astype(np.float32)
    s = f90(gt) if a.bin_by == "f90" else fit_sigma(box_down(gt), lr-box_down(gt))
    if s is not None: rows.append((s,i))
rows.sort()
n=len(rows); k=n//3
bins={"low":rows[:k],"mid":rows[k:2*k],"high":rows[2*k:]}
lab = "structure fineness (f90)" if a.bin_by=="f90" else "noise level (sigma_mul)"
print(f"{n} validation images, binned by their OWN measured {lab}")
for b,v in bins.items():
    print(f"   {b:<5} n={len(v):<4} {v[0][0]:.4f} – {v[-1][0]:.4f}   (mean {np.mean([x[0] for x in v]):.4f})")
print()

def psnr(x,g): return 10*np.log10(1/max(float(torch.mean((x-g)**2)),1e-12))
hdr=f"{'checkpoint':<22}" + "".join(f"{b+' PSNR':>12}{b+' SSIM':>12}{b+' LPIPS':>13}" for b in bins)
print(hdr)
for ck_path in a.ckpt:
    ck=torch.load(ck_path,map_location="cpu")
    m=Restorer(**ck.get("config",{})).eval().to(dev); m.load_state_dict(ck.get("state_dict",ck))
    out=f"{os.path.basename(os.path.dirname(ck_path)):<22}"
    for b,v in bins.items():
        ps,ss,lp=[],[],[]
        for j,(_,i) in enumerate(v):
            gt=torch.from_numpy(np.load(os.path.join(G,i+".npy")).astype(np.float32))[None,None].to(dev)
            lr=torch.from_numpy(np.load(os.path.join(L,i+".npy")).astype(np.float32))[None,None].to(dev)
            with torch.inference_mode(): o=m(lr).clamp(0,1)
            ps.append(psnr(o,gt)); ss.append(float(m_ssim(o,gt)))
            if j<25: lp.append(m_lpips(o,gt,device=dev))
        out+=f"{np.mean(ps):>12.3f}{np.mean(ss):>12.4f}{np.nanmean(lp):>13.4f}"
    print(out)
# bicubic reference
out=f"{'bicubic':<22}"
for b,v in bins.items():
    ps,ss=[],[]
    for _,i in v:
        gt=torch.from_numpy(np.load(os.path.join(G,i+".npy")).astype(np.float32))[None,None].to(dev)
        lr=torch.from_numpy(np.load(os.path.join(L,i+".npy")).astype(np.float32))[None,None].to(dev)
        o=F.interpolate(lr,scale_factor=2,mode="bicubic",align_corners=False).clamp(0,1)
        ps.append(psnr(o,gt)); ss.append(float(m_ssim(o,gt)))
    out+=f"{np.mean(ps):>12.3f}{np.mean(ss):>12.4f}{'—':>13}"
print(out)
