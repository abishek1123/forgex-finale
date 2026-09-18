# Literature

Compiled 11 Sep 2026. Every link was verified at compile time, none written from
memory. This is the reading behind the project and the citations that defend it.

---

## Three findings that change what we say

### 1. Our capacity argument was stated backwards

An earlier version of the README and the engineering log said: *"the curve is a
straight line with no knee, therefore we are not capacity-limited."* The
scaling-law literature reads that signature the **opposite** way. A straight power
law means capacity is **still** buying quality at a constant rate. A **knee** is
what indicates saturation — Klug & Heckel's central result is that scaling *slows*
once the reconstruction floor dominates.

A judge who knows this literature takes that slide apart. The data still supports
the conclusion; the argument has to change.

- **Do not say:** "no knee, therefore not capacity-limited"
- **Say:** across a 78.4× parameter span we gain **0.39 dB** on the organisers'
  test set (0.29 dB on our held-out split) — about **0.05 dB per doubling**, an
  order of magnitude shallower than typical vision scaling exponents. The return
  on capacity is negligible at any budget we could afford, and inference time is
  scored.

That version argues a **flat slope**, not saturation, and R² = 0.992 over 78× is
exactly the evidence for it.

> Klug & Heckel, [Scaling Laws For Deep Learning Based Image Reconstruction](https://arxiv.org/abs/2209.13435), ICLR 2023 ·
> Bahri et al., [Explaining Neural Scaling Laws](https://arxiv.org/abs/2102.06701), PNAS 2024 ·
> Lin et al., [Limits of Learning-Based Superresolution](https://link.springer.com/article/10.1007/s11263-008-0148-2), IJCV 2008

### 2. The biggest threat to our evaluation section

Yang & Wang benchmark learned SR on line/space and contact-hole SEM at a
predeclared low false-positive rate. Their finding: the models attain the
**highest SSIM yet detect fewer defect pixels than plain bicubic upsampling.** On
semiconductor inspection imagery. Published July 2026.

If we report only PSNR / SSIM / LPIPS, this is the objection we will get, and it
is a good one — the point of restoring an inspection image is the inspection.

**The fix:** add one downstream metric. Defect recall at a fixed false-positive
rate, or edge-position error on a line/space pattern. Even a small version changes
the claim from *"our images look better"* to *"our images measure better."*

> Yang & Wang, [Does Super-Resolution Preserve Defect Evidence? A Low-False-Call Benchmark for Semiconductor Inspection](https://arxiv.org/abs/2607.17401), arXiv:2607.17401, 2026

### 3. Our variance-stabilising stem appears genuinely unpublished

Five independent counter-searches found nothing matching it. Every VST-plus-network
paper applies **one** transform **sequentially** — transform, denoise, inverse.
Herbreteau & Unser (ICCV 2025) learn a VST, but it is a *parameterised*
piecewise-linear function applied as a single view. VST-Net replaces
forward/denoise/inverse with three *learned* sub-networks — still serial, still
parameterised.

Both differentiators hold: **parameter-free**, and **parallel multi-view** rather
than serial single-view.

**Safe phrasing:** "a parameter-free multi-view variance-stabilising stem that
defers transform selection to the network, in contrast to serial single-transform
VST pipelines." Claim the specific combination — not "multi-channel input", which
is unremarkable.

---

## Eight papers to read before judging

**1. [AI-powered super-resolution and denoising of SEM images for high-throughput metrology and inspection](https://doi.org/10.1117/1.JMM.24.4.041407)** — Tiong, Kim, Choi et al. (Samsung), JM3 24(4) 041407, 2025
The nearest published thing to what we built, and the paper a knowledgeable judge
will name. Their MI-IES framework does SR and denoising as **two chained modules**;
we do one joint blind restoration. Know this difference cold — it is our
related-work anchor and our differentiation in the same sentence.

**2. [Does Super-Resolution Preserve Defect Evidence?](https://arxiv.org/abs/2607.17401)** — Yang & Wang, 2026
See finding 2. Have an answer ready even without time to run the experiment.

**3. [The Perception-Distortion Tradeoff](https://arxiv.org/abs/1711.06077)** — Blau & Michaeli, CVPR 2018
Converts our PSNR deficit from "a regression we have to excuse" into "a predicted,
theoretically necessary movement along a frontier". This is the answer to *"why is
your PSNR below the field leader?"* — we chose an operating point, and here is the
proof the frontier exists.

**4. [Distribution Matching Losses Can Hallucinate Features in Medical Image Translation](https://arxiv.org/abs/1805.08841)** — Cohen, Luck & Honari, MICCAI 2018
The strongest single citation for our no-GAN position. Distribution-matching losses
will **add or remove lesions** to match the target distribution's class balance.
Swap "lesion" for "defect" and it is our argument, peer-reviewed.

**5. [EstimateNoiseSEM: deep-learning noise estimation for SEM images](https://doi.org/10.1016/j.ultramic.2025.114192)** — Rahman, Salomon & Dembélé, Ultramicroscopy 276, 2025
Independent evidence that Poisson is the wrong default for SEM: noise follows
**Gaussian or Gamma depending on dwell time**, Gamma being multiplicative. Their
Gamma accuracy drops to ~80% and they explicitly call for better Gamma noise
estimation. Cite it, then present our fitted law as the answer they asked for.

**6. [Effect of Shot Noise and Secondary Emission Noise in SEM Images](https://www.ece.nus.edu.sg/stfpage/elettl/PDF%20files/E-publications/2004-Scanning-26-SimKS-Effect%20of%20shot%20noise%20and%20secondary%20emission%20noise%20in%20SEM%20images.pdf)** — Sim, Thong & Phang, Scanning 26:36–40, 2004
**Memorise this one.** It turns ΔBIC 1118 from an empirical curiosity into a
principled result: Poisson primary-beam shot noise compounded with binomial
secondary yield is **over-dispersed relative to Poisson** — variance grows faster
than the mean. That is the physical mechanism behind our σ_mul²·m² term, and the
best answer to "why speckle?"

**7. [Robust and Interpretable Blind Image Denoising via Bias-Free CNNs](https://arxiv.org/abs/1906.05478)** — Mohan, Kadkhodaie, Simoncelli & Fernandez-Granda, ICLR 2020
Denoising CNNs fail catastrophically outside their training noise range, and
removing additive bias terms restores generalisation. A concrete, cheap, testable
mechanism for our `p_real=1.0` collapse — and a possible architectural fix worth
one ablation. Code and models public.

**8. [The Tenth NTIRE 2025 Efficient Super-Resolution Challenge Report](https://arxiv.org/abs/2504.10686)** — Ren et al., CVPRW 2025
A published competition that scores inference time the way ours might: runtime
0.7 / FLOPs 0.15 / params 0.15 against a hard PSNR floor. If the organisers
confirm time is weighted, this is the precedent for how to trade.

---

## Baselines a reviewer will expect

"No comparison against published methods" is the most common desk-reject for an
applied restoration paper. These four have maintained public weights at ×2 — our
exact scale factor — so it is an afternoon of inference, not a research project.

| Paper | Why it matters |
|---|---|
| [BSRGAN](https://arxiv.org/abs/2103.14006) — K. Zhang et al., ICCV 2021 | The randomly-shuffled blur/downsample/noise pipeline is the direct ancestor of our wide degradation family. **Our most important baseline.** |
| [Real-ESRGAN](https://arxiv.org/abs/2107.10833) — Wang et al., ICCVW 2021 | Literally the paper a judge means by "why not use an off-the-shelf model". Its claim that *pure* synthetic data suffices makes our `p_real` result a domain-specific counter-example. |
| [SwinIR](https://arxiv.org/abs/2108.10257) — Liang et al., ICCVW 2021 | Answers "why not a transformer" and "why one joint model" in one experiment — it covers SR *and* denoising. Easiest strong transformer baseline to run. |
| [DASR](https://arxiv.org/abs/2203.14216) — Liang, Zeng & Zhang, ECCV 2022 | Splits the degradation space by difficulty and routes to lightweight experts — published precedent for our wide-vs-narrow split, light enough for our hardware. |

**Run them on both sets.** Benchmark all four on the visible test set *and* on the
strong-noise sweep. That produces the two-axis figure — in-distribution quality
against robustness — where our mixture beats both a pure-real and a pure-synthetic
model. We currently assert that tradeoff; this figure would prove it against named
published methods.

---

## Novelty claims, checked

| Claim | Verdict |
|---|---|
| Parameter-free multi-view VST stem | **Safe.** Nothing matching found; all prior VST work is serial and single-view. |
| A fitted, model-selected parametric degradation law for SEM | **Safe — our strongest.** EstimateNoiseSEM *classifies* Gaussian vs Gamma; Foi fits Poisson-Gaussian for camera raw; Sim et al. derive the physics analytically. **No paper found that fits a signal-dependent variance model to SEM data and does formal BIC model comparison between speckle and Poisson.** Make this claim explicitly. |
| Hallucination guard — energy above input Nyquist | **Safe.** Nearest neighbours (Deng et al. ICML 2024; Durall et al. CVPR 2020) *support* the construct rather than pre-empt it. |
| "Straight-line frontier proves we are not capacity-limited" | **Rephrase.** Inverted — see finding 1. The magnitude argument is correct and stronger. |
| Sweeping the real-vs-synthetic ratio to show a robustness crossover | **Nobody has done it.** TGSR (NeurIPS 2023) and Controlled Data Rebalancing (2025) are nearest but work *within* synthetic spaces. TGSR gives the vocabulary: multi-task learning with **task competition**. |

---

## The full map

### Degradation model and noise physics

- **[Practical Poissonian-Gaussian Noise Modeling and Fitting](https://webpages.tuni.fi/foi/papers/Foi-PoissonianGaussianClippedRaw-2007-IEEE_TIP.pdf)** — Foi et al., IEEE TIP 2008. The direct ancestor of our law, same local mean/variance fitting. We extend it from a linear to a quadratic mean term.
- **[A Variational Approach to Removing Multiplicative Noise](https://www.math.u-bordeaux.fr/~jaujol/PAPERS/uv.pdf)** — Aubert & Aujol, SIAM 2008. The canonical Gamma speckle model; our shifted-Gamma k=14 is the equivalent number of looks.
- **[Shot noise-mitigated secondary electron imaging](https://www.pnas.org/doi/10.1073/pnas.2401246121)** — Agarwal et al., PNAS 2024. Three-term signal-dependent variance model mapping almost one-to-one onto our three fitted terms.
- **[MuLoG](https://arxiv.org/abs/1704.05335)** — Deledalle et al., IEEE TIP 2017. Justifies the `log1p` branch: log is the variance-stabilising choice for multiplicative speckle.
- **[The Transformation of Poisson, Binomial and Negative-Binomial Data](https://academic.oup.com/biomet/article-abstract/35/3-4/246/280278)** — Anscombe, Biometrika 1948. Provenance for the `√x` branch.
- **Optimal inversion of the (generalized) Anscombe transformation** — Mäkitalo & Foi, IEEE TIP 2011/2013. Required the moment we mention inverting a VST — naive algebraic inversion is biased.
- **Noise Parameter Mismatch in Variance Stabilization** — Mäkitalo & Foi, IEEE TIP 2014. Quantifies how badly a VST degrades when its parameters are wrong — the strongest motivation for three views instead of one.
- **[Nanomanufacturing Concerns Part V: Dealing with Noise](https://www.nist.gov/publications/nanomanufacturing-concerns-about-measurements-made-sem-part-v-dealing-noise)** — Postek & Vladar, NIST 2016. SNR to dimensional measurement imprecision — our motivation paragraph, from the authoritative source.
- **[Self-Calibrated Variance-Stabilizing Transformations](https://arxiv.org/abs/2407.17399)** — Herbreteau & Unser, ICCV 2025. Nearest prior art on the stem — cite and distinguish.

### Blind / real-world SR and the mixture question

- **[Real-World Image SR as Multi-Task Learning (TGSR)](https://proceedings.neurips.cc/paper_files/paper/2023/hash/42806406dd99e30c3796bc98b2670fa2-Abstract-Conference.html)** — W. Zhang et al., NeurIPS 2023. Formalises our effect as multi-task learning with task competition.
- **[Evaluating the Generalization Ability of Super-Resolution Networks (SRGA)](https://arxiv.org/abs/2205.07019)** — Y. Liu et al., TPAMI 2023. A *published metric* for the robustness axis.
- **[Controlled Data Rebalancing in Multi-Task Learning for Real-World Image SR](https://arxiv.org/abs/2506.05607)** — Lin et al., 2025. Closest published work to our mixture ratio.
- **[Blind Image Super-Resolution: A Survey and Beyond](https://arxiv.org/abs/2107.03055)** — A. Liu et al., TPAMI 2022. The positioning citation; its explicit-vs-implicit taxonomy places our hybrid scheme on a named axis.
- **[RealSR](https://arxiv.org/abs/1904.00523)** — Cai et al., ICCV 2019. Canonical justification for collecting real LR/HR pairs.
- **[SRMD](https://arxiv.org/abs/1712.06116)** · **[IKC](https://arxiv.org/abs/1904.03377)** · **[KernelGAN](https://arxiv.org/abs/1909.06581)** · **[DASR-CVPR21](https://arxiv.org/abs/2104.00416)** — the kernel-estimation lineage; cite as a group.
- **[GenDeg](https://arxiv.org/abs/2411.17687)** — Rajagopalan et al., CVPR 2025. The modern counterpart to our synthetic engine.

### Architecture, efficiency and the capacity frontier

- **[ESPCN](https://arxiv.org/abs/1609.05158)** — Shi et al., CVPR 2016. The origin of our exact choice: all features in LR space, upsample once at the end.
- **[EDSR](https://arxiv.org/abs/1707.02921)** — Lim et al., CVPRW 2017. Our architecture family; canonical reference for removing BatchNorm and for residual scaling.
- **[IMDN](https://arxiv.org/abs/1909.11856)** — Hui et al., ACM MM 2019. **Figure 6 is a published PSNR-vs-parameters plot** across ten architectures from ~8K to >1.5M params. Overlay our curve on it.
- **[RLFN](https://arxiv.org/abs/2205.07514)** — Kong et al., CVPRW 2022. Argues parameter/FLOP-minimising structures are *not* what makes a model fast. Direct authority for "we optimised wall-clock, not parameter count".
- **[ShuffleNet V2](https://arxiv.org/abs/1807.11164)** — Ma et al., ECCV 2018. "FLOPs and params are indirect metrics; measure speed on the target device." Methodological cover for our end-to-end timing protocol.
- **[NTIRE 2022](https://arxiv.org/abs/2205.05675)** · **[AIM 2020](https://arxiv.org/abs/2009.06943)** · **[NTIRE 2024](https://arxiv.org/abs/2404.10343)** — the hold-quality/minimise-latency protocol with published frontiers.
- **[CARN](https://arxiv.org/abs/1803.08664)** · **[RFDN](https://arxiv.org/abs/2009.11551)** · **[SAFMN](https://arxiv.org/abs/2302.13800)** — the lightweight lineage. CARN-M has an explicit param-budget ablation.
- **[ASSL](https://proceedings.neurips.cc/paper/2021/hash/15de21c670ae7c3f6f3f1f37029303c9-Abstract.html)** — Y. Zhang et al., NeurIPS 2021. If asked "why not prune a big model instead", this is the state of the art to name.
- **[HAT](https://arxiv.org/abs/2205.04437)** — X. Chen et al., CVPR 2023. Current transformer SOTA; its analysis of how little input a transformer uses supports our "capacity is not the bottleneck" line.

### Loss, metrics and the hallucination argument

- **[Loss Functions for Image Restoration with Neural Networks](https://arxiv.org/abs/1511.08861)** — Zhao et al., IEEE TCI 2017. Precedent for a mixed loss and empirical weight tuning.
- **[LapSRN](https://arxiv.org/abs/1704.03915)** — Lai et al., CVPR 2017. Established Charbonnier as the standard robust reconstruction loss.
- **[Structure-Preserving SR with Gradient Guidance](https://arxiv.org/abs/2003.13081)** — Ma et al., CVPR 2020. Our citation for the gradient term.
- **[LPIPS](https://arxiv.org/abs/1801.03924)** — R. Zhang et al., CVPR 2018. **Calibration detail worth stating:** fit on human judgments over *natural photographs*. Its human-alignment guarantee does not transfer unexamined to SEM imagery. A judge may ask.
- **[SSIM](https://ece.uwaterloo.ca/~z70wang/publications/ssim.html)** — Wang et al., IEEE TIP 2004. Worth citing precisely because SSIM never claims to validate *synthesised* structure.
- **[Understanding SSIM](https://arxiv.org/abs/2006.13846)** — Nilsson & Akenine-Möller, 2020. Evidence that a ~0.01 SSIM delta sits inside the metric's implementation-sensitivity band.
- **[On instabilities of deep learning in image reconstruction](https://www.pnas.org/doi/abs/10.1073/pnas.1907377117)** — Antun et al., PNAS 2020. Networks failing to reconstruct small structural changes — a tumour, or a defect.
- **[The Troublesome Kernel](https://arxiv.org/abs/2001.01258)** — Gottschling et al., SIAM Review 2025. Proves hallucination is a structural consequence of the forward operator's null space, not a training bug. The theoretical backbone for our range/null-space decomposition.
- **[Hallucination Index](https://arxiv.org/abs/2407.12780)** — Tivnan et al., MICCAI 2024. **Validated on electron microscopy.** The closest published relative to our hallucination guard, in our modality.
- **[Exploring the Low-Pass Filtering Behavior in Image SR](https://arxiv.org/abs/2405.07919)** — Deng et al., ICML 2024. The "sinc phenomenon"; nearest published methodology to our above-Nyquist guard.
- **[Watch your Up-Convolution](https://arxiv.org/abs/2003.01826)** — Durall et al., CVPR 2020. Upsampling layers systematically produce wrong high-frequency spectra.
- **[The 2018 PIRM Challenge](https://arxiv.org/abs/1809.07517)** — Blau et al., ECCVW 2018. Precedent for reporting a *pair* of numbers and defending a chosen point on the curve.
- **[A Theory of the Distortion-Perception Tradeoff in Wasserstein Space](https://arxiv.org/abs/2107.02555)** — Freirich et al., NeurIPS 2021. Justifies our *small* LPIPS weight as principled interpolation rather than a guessed hyperparameter.
- **[Image-to-Image Regression with Distribution-Free Uncertainty Quantification](https://arxiv.org/abs/2202.05265)** — Angelopoulos et al., ICML 2022. The answer to "how would you *bound* the risk of a fabricated feature?" — not implemented, but knowing it exists is the difference between a shrug and an answer.

### SEM and electron microscopy — the domain

- **[Resolution enhancement in SEM using deep learning](https://arxiv.org/abs/1901.11094)** — de Haan et al., Sci. Rep. 2019. The canonical SEM SR paper. Mandatory citation — and their adversarial loss is exactly the hallucination risk the defect benchmark quantifies.
- **[ReNIn: Efficient and Robust SEM Image Denoising for Wafer Defect Inspection](https://academic.oup.com/mam/article/31/5/ozaf084/8300402)** — Bae et al., Microsc. Microanal. 2025. Validates with PSNR/SSIM **plus a downstream detection-failure rate** — precedent for adding that metric.
- **[Deep learning denoising of SEM images towards noise-reduced LER measurements](https://doi.org/10.1016/j.mee.2019.111051)** — Microelectronic Engineering 2019. Trained on *synthesised* noisy SEM data for line-edge-roughness metrology — our methodological precedent. *(Verify the author list from the publisher before citing.)*
- **[Towards SEM image denoising: overview, benchmark, taxonomies](https://link.springer.com/article/10.1007/s00138-024-01573-9)** — Rahman et al., MVA 2024. The only SEM-denoising survey with a real benchmark; releases a 500-image dataset.
- **[Deep learning denoising enables rapid SEM imaging under charging conditions](https://www.nature.com/articles/s41598-025-33273-3)** — Park et al., Sci. Rep. 2025. Restormer/NAFNet/HINet on 2-frame inputs with 32-frame averaged references. The frame-averaging protocol is how clean SEM targets are made.
- **[Deep Denoising for Scientific Discovery: A Case Study in Electron Microscopy](https://arxiv.org/abs/2010.12970)** — Mohan et al., IEEE TCI 2022. Supports "generic natural-image priors are the wrong prior for electron imaging".
- **[Sparsity-Based Super Resolution for SEM Images](https://arxiv.org/abs/1709.02235)** — Tsiper et al., Nano Letters 2017. Pre-deep-learning dictionary learning on *microelectronic chip* SEM. Our classical baseline.
- **[SR for SEM images based on pixelwise weighted loss](https://academic.oup.com/jmicro/article/72/5/408/7026008)** — Ito et al. (Hitachi), Microscopy 2023. Content-weighted loss for semiconductor defect detection.
- **[High throughput CD-SEM metrology using deep-learning denoising](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/12955/129551Z/High-throughput-CD-SEM-metrology-using-image-denoising-based-on/10.1117/12.3006709.short)** — Okuda et al., SPIE 2024. The industrial throughput argument — our "why this matters economically" paragraph.
- **[Deep learning-based defect classification and detection in SEM images](https://arxiv.org/abs/2206.13505)** — Dey et al. (imec), SPIE 2022. The downstream task our restoration is supposed to serve.
- **[Zero-Shot Image Denoising for High-Resolution Electron Microscopy](https://arxiv.org/abs/2406.14264)** — Tian et al., IEEE TCI 2024. The "you have no clean data at all" alternative.
- **[Topaz-Denoise](https://www.nature.com/articles/s41467-020-18952-1)** · **[Unsupervised deep denoising for 4D-STEM](https://www.nature.com/articles/s41524-024-01428-x)** — the 4D-STEM paper uses *capacity-constrained* networks that learn structure without memorising noise; useful for defending a deliberately small model.
- **[The first annotated set of SEM images for nanoscience](https://pmc.ncbi.nlm.nih.gov/articles/PMC6111892)** — Aversa et al., Sci. Data 2018. ~22k annotated SEM images, the largest public corpus.

### Self-supervised denoising — "where did your targets come from?"

- **[Noise2Noise](https://arxiv.org/abs/1803.04189)** — Lehtinen et al., ICML 2018. Origin of the clean-data-free line.
- **[Noise2Void](https://arxiv.org/abs/1811.10980)** — Krull et al., CVPR 2019. Worth a sentence: its assumption of pixel-wise independent zero-mean noise is **violated** by our multiplicative term. Saying so shows we read it properly.
- **[Noise2Self](https://arxiv.org/abs/1901.11365)** — Batson & Royer, ICML 2019. The J-invariance theory.
- **[Neighbor2Neighbor](https://arxiv.org/abs/2101.02824)** — Huang et al., CVPR 2021. Its sub-sampler is a **2×2 construction**, directly relevant to our 2×2 block statistics.
- **[Speckle2Void](https://arxiv.org/abs/2007.02075)** — Bordone Molini et al., IEEE TGRS 2022. Blind-spot denoising on an explicit **Gamma speckle likelihood** — the closest existing combination of our two core commitments.
- **[FFDNet](https://arxiv.org/abs/1710.04026)** · **[CBDNet](https://arxiv.org/abs/1807.04686)** — the main design alternative: inject an explicit noise-level map rather than let a stem discover the representation.

---

## Citation hygiene

- arXiv:2407.17399 carries different titles across versions ("Self-Calibrated
  Variance-Stabilizing Transformations" vs the repo name "Noise2VST") — check
  which one the venue used.
- The Microelectronic Engineering 2019 LER paper needs its author list verified
  from the publisher before citing.
- Do not cite anything in this file from memory. Open the link first.
