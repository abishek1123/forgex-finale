# ForgeX — KLA PS01 Round 2
## Demo video script · target 3:30

**Setup before recording**

- Terminal at `C:\Users\abishek sr\Desktop\kla\kla-restore`, font large enough to read at 1080p
- A second window with `outputs\preview.png` ready to show
- Delete `..\_probe_out` first so the run is clean
- Record at 1080p. Speak at a normal pace — the script is written for ~150 wpm

---

### 0:00 – 0:25 · What the problem is

> "Every wafer is imaged at every process step. Imaging fast gives you a noisy,
> half-resolution frame; imaging slowly gives you a clean one but costs throughput.
> A defect hidden under speckle is a defect that ships.
>
> So the task is: take the fast frame, remove the noise, and put the resolution back.
> One two-hundred-and-fifty-six-pixel image from one twenty-eight, denoised, in
> milliseconds."

**On screen:** slide 2 of the deck, or the preview image with the degraded column visible.

---

### 0:25 – 1:05 · What we actually did — the one idea

> "We did not start from an architecture. We started by measuring the damage.
>
> Fitting residual variance against the clean image across KLA's own pairs recovers
> their forward model: an additive floor, a term that grows with the square of
> brightness, and a term that grows with local detail. Adjusted R-squared of point
> nine nine eight seven.
>
> That square-of-brightness term matters. Poisson shot noise would be linear in
> brightness; multiplicative speckle is quadratic. We fitted both and compared them
> by BIC. The difference is eleven hundred, which is decisive — so the generator
> multiplies, and it draws from a Gamma distribution, because the real residual is
> skewed and a Gaussian would have matched the variance and missed the shape."

**On screen:** slide 3, then slide 4 (Synthetic Degradation Generation).

---

### 1:05 – 1:35 · The result that reversed us

> "Here is the part we got wrong first.
>
> Training on a hundred percent real pairs beat every synthetic mixture on our
> held-out split. We shipped that for a week. Then we swept the noise level instead
> of holding it fixed — and at the level KLA's data actually sits, every model we
> trained is within point three of a decibel of every other. We had been resolving
> a tenth of a decibel at the one point where there is nothing to see.
>
> Three steps outside that range, the real-only model is four decibels behind, and
> it turns over — its gain peaks and then falls. That is the signature of a model
> that memorised a noise amplitude instead of learning the inverse problem.
>
> So we train on a degradation family that is deliberately wider than reality. On the
> organisers' own test set that costs us about a hundredth of a decibel. Three sigma
> steps out it buys three point seven."

**On screen:** slide 7 — the in-distribution / out-of-distribution chart. Let it sit.

---

### 1:35 – 2:35 · Run it — the live demo

> "Here is the submitted script running on the two hundred and ninety-seven image
> test set the organisers gave us."

```
python run.py ..\semicon_test_data ..\demo_out
```

> "Note the first line. We handed it the dataset folder, not the images folder — it
> found the degraded inputs one level down and said so, rather than exiting with
> nothing. Two hundred and ninety-seven images, end to end, in under six seconds.
> Roughly seventy percent of that is Python importing torch and CUDA starting up;
> the network itself is about two milliseconds an image."

> "And here is the verification harness."

```
python tools\package_check.py --data ..\semicon_test_data
```

> "It restores every image, checks dtype, finiteness, range and shape on all two
> hundred and ninety-seven, and then copies only the four required files into an
> empty directory and runs from there. Bit-identical output. That is what PASS means."

**On screen:** the terminal, full width. Do not talk over the output — let both runs
finish visibly.

---

### 2:35 – 3:05 · The numbers

> "On the organisers' test set: twenty-three point six three decibels against bicubic's
> twenty point four six. Plus three point one eight, and fifty-nine percent lower
> perceptual distance, from one point three seven million parameters.
>
> The same weights, on things they have not seen: plus three point six five on a
> content block we withheld entirely. Plus four point five two at five-twelve
> resolution, which the model was never trained at. And positive on all ten labelled
> morphology classes from a different image corpus — mean plus four point zero one.
>
> The gain is largest where the model has seen least. That is what we optimised for."

**On screen:** slide 9 (Results) — the table and the before/after panel.

---

### 3:05 – 3:30 · Why this model, and close

> "We trained eleven checkpoints and scored all of them in one pass. We do not ship
> the one with the best PSNR — we are ninth of eleven on it.
>
> Across the field, the worst model still keeps ninety-one percent of the best PSNR
> gain. On perceptual distance, the worst keeps thirty-two percent. The downside is
> seven and a half times larger on the perceptual axis, so we shipped the model whose
> weakest axis is strongest, and we wrote down why.
>
> Everything is in the repository — including eight claims we published during this
> round and had to withdraw, and three optimisations we wrote, benchmarked, and
> rejected. We stopped tuning the model and started measuring the data, and every
> result that beat our noise floor came from that decision."

**On screen:** slide 8 (Model Selection), then the final slide.

---

## Recording notes

- **Total ≈ 3:30.** If you need 3:00, cut the BIC sentences at 0:45 and the
  package_check demo at 2:15 — keep the `run.py` run, it is the required component.
- **Do not rush the terminal.** Dead air while a command runs reads as confidence.
- The single most important 20 seconds is 1:05–1:35 — the reversal. Everything else
  is evidence for it. Slow down there.
- Have `outputs\preview.png` open as a fallback if a command misbehaves on the day.
- Say "we measured" rather than "we think" throughout. It is accurate here, and it is
  the whole difference between this submission and a tuned baseline.
