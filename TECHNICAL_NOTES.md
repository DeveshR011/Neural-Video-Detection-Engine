# Technical Notes

Engineering notes from rebuilding the detection core. Three of these are
silent-correctness defects: bugs that produce plausible-looking output while
being wrong. They are recorded here because each was found by a different
technique, and each is now pinned by a regression test.

---

## 1. A loss function that could not train

**Symptom.** Training ran, loss decreased, checkpoints were written. There was
no error at any point. But no evaluation harness existed, so nobody had ever
measured whether the model detected anything.

**Diagnosis.** The loss needed to convert flat `[N, 4]` box predictions into
decoded boxes for IoU matching. To do that it had to know the feature-map
geometry — but only received a flat tensor. It tried to *infer* the grid shape
from the prediction count:

```python
cells_total = n_pred // 3
base = int(round((cells_total * 16.0 / 21.0) ** 0.5))
h1 = max(base, 4); h2 = max(h1 // 2, 2); h3 = max(h1 // 4, 1)

counts = [3*h1*w1, 3*h2*w2, 3*h3*w3]
if sum(counts) != n_pred:
    # silently give up and return something shaped correctly
    xy = torch.sigmoid(pred_boxes_raw[:, :2])
    wh = torch.sigmoid(pred_boxes_raw[:, 2:4])
    return torch.cat([xy, wh], dim=-1)
```

The constant `16/21` encodes an assumption that the three pyramid levels
contribute cells in the ratio 16:4:1. When the arithmetic failed to round-trip
— which happens for most input sizes — the function fell back to
`sigmoid(raw)`. That fallback discards the grid entirely: every predicted box
collapses into the unit square regardless of which cell produced it.

Target assignment then ran global IoU matching over those boxes, inside a
Python loop over the batch.

**Why it stayed hidden.** The fallback returns a correctly *shaped* tensor.
Shapes flow, gradients flow, loss decreases — the model learns to place all
boxes near the dataset's mean box. Only an actual detection metric reveals
that the output is meaningless, and there was no metric.

**Fix.** Anchor-free head where each prediction is bound to its grid cell by
construction. The head emits `(anchor_points, stride_tensor)` alongside the
predictions, so geometry is *passed*, never *inferred*. Assignment is handled
by a Task-Aligned Assigner scoring pairs by `s^alpha * u^beta`.

**Regression test.** `test_overfit_single_batch` — a correct detector must
drive loss down on a fixed batch and recover the target boxes. The old
implementation could not satisfy this.

**Lesson.** A function that guesses its inputs and silently degrades on failure
is worse than one that crashes. If `_decode_flat_boxes` had raised on the
count mismatch, this would have been found on day one.

---

## 2. A loss value that exceeded its mathematical maximum

**Symptom.** After the rewrite, first training step reported:

```
box: 14.446   cls: 11.988   dfl: 2.860
```

**Diagnosis.** The box term is `1 - CIoU`. CIoU is bounded below by roughly
-1.5: IoU is at least 0, the distance penalty `rho2/c2` is at most 1, and the
aspect-ratio term `v * alpha` is at most 0.5 since `alpha = v/(v - iou + 1)`
is at most 0.5 when IoU is 0. So `1 - CIoU` **cannot exceed about 2.5**.

A value of 14.4 was not a bad number. It was an impossible one.

The cause was tensor broadcasting:

```python
weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)   # [N, 1]
iou    = bbox_iou(pred[fg_mask], target[fg_mask], ...)  # [N]
loss   = ((1.0 - iou) * weight).sum() / target_scores_sum
```

`[N] * [N, 1]` broadcasts to `[N, N]` — an outer product. Every prediction was
multiplied by every other prediction's weight, and the sum ran over N² terms
instead of N. The loss was inflated by roughly a factor of N.

**Why it stayed hidden.** It is silent by construction. No error, no NaN, and
the loss still decreases — the gradient direction is not random, merely
mis-weighted. Training would have produced a mediocre model that looked like
it simply needed more epochs.

**Fix.** Keep both operands one-dimensional, with the shape contract stated at
the call site:

```python
weight = target_scores.sum(-1)[fg_mask]   # [N]
iou    = bbox_iou(...)                    # [N]
```

Box loss dropped from 14.447 to 0.430 — inside the valid range.

**Regression test.** `test_bbox_iou_shape_matches_weight` asserts
`bbox_iou(...).shape == (n,)`, and `test_ciou_bounded` asserts `1 - CIoU` stays
within `[0, 2.5]` for maximally distant boxes.

**Lesson.** Know the bounds of your own loss terms. The bug was invisible in
the code and obvious in the output — but only to someone who knew that 14.4
was impossible. Broadcasting is the most common source of silent numerical
error in PyTorch, and the defense is asserting shapes, not reading code
harder.

---

## 3. A bfloat16 rounding error that overflowed a distribution

**Symptom.** Training on real COCO images crashed inside the DFL term:

```
IndexError: Target 16 is out of bounds.
...
torch.AcceleratorError: CUDA error: device-side assert triggered
```

It did **not** reproduce in the single-batch test, on synthetic data, or in
float32. Only with AMP enabled, on real images.

**Diagnosis.** DFL represents each box side as a distribution over `reg_max`
bins (here 16, indices 0–15). A continuous target distance is supervised by
cross-entropy against the two bins bracketing it:

```python
tl = target.long()   # lower bin
tr = tl + 1          # upper bin
```

For `tr` to be valid, the target must stay strictly below `reg_max - 1 = 15`.
The encoder clamped accordingly:

```python
return torch.cat((lt, rb), -1).clamp_(0, reg_max - 1.01)   # 14.99
```

That is correct in float32. Under bfloat16 autocast it is not.

bfloat16 keeps 8 bits of mantissa. Near 15 the representable spacing is
`2^3 * 2^-8 = 0.0625`. The nearest representable values around 14.99 are
14.9375 and **15.0**, so `14.99` rounds to exactly `15.0`. Then `tl = 15`,
`tr = 16`, and cross-entropy indexes past the end of a 16-bin distribution.

The reason it needed real data: only a box whose edge is far from its anchor
point produces a distance near the clamp ceiling. Synthetic boxes were too
small to reach it.

**Fix.** Enforce the range in float32 at the point of use, rather than relying
on a clamp that must survive an unknown autocast dtype:

```python
n_bins = pred_dist.shape[1]
target = target.float().clamp_(0, n_bins - 1 - 1e-3)
tl = target.long()
tr = (tl + 1).clamp_(max=n_bins - 1)
```

**Regression test.** `test_dfl_bin_range_under_bf16` constructs the target in
bfloat16 and first asserts the precondition — `float(target[0]) == 15.0` —
so the test documents *why* it exists rather than merely checking the fix.

**Lesson.** A numerical guard must hold in the dtype it will actually execute
in. Reduced-precision training silently changes the semantics of any epsilon
or near-boundary clamp. A margin of 0.01 is meaningless in a format whose
resolution at that magnitude is 0.0625.

---

## 4. Security features were in the accuracy path

Not a bug — a design problem worth recording.

The original forward pass applied a frozen random projection to the P3/P4/P5
feature maps and called `.zero_()` on intermediate activations during eval:

```python
p3 = self.random_projection(p3)
...
if not self.training:
    c3.zero_(); del c3
```

Both are costly and neither is effective where it sits:

- **Random projection** on feature maps directly degrades detection accuracy.
  It is only meaningful if an untrusted party can observe intermediate
  activations — a threat model that does not apply to a local process.
- **Zeroing activations** mutates tensors that may still be referenced by
  autograd, and the memory was about to be freed regardless. It buys nothing
  and risks corrupting the graph.

**Resolution.** The projection became an opt-in `--secure-features` flag,
default off, documented as reducing mAP. The `.zero_()` calls were removed
from the forward pass. Model encryption, the hash-chain audit log, the
integrity registry, and device-bound keys are unchanged — they operate at
checkpoint load/save and at API boundaries, where they are actually effective.

**All reported benchmark numbers were produced with `secure_features=False`.**
The security layer costs zero mAP because it no longer sits in the inference
path.

---

## Verification approach

Detection pipelines fail silently. These checks were built to force failures
into the open:

| Check | What it catches |
|---|---|
| **Overfit a single batch** | Any defect that makes the loss untrainable. Loss must collapse and boxes must be recovered to sub-pixel accuracy. |
| **Memorize a small real set** | Defects appearing only with real data. With augmentation off, the model reached 0.798 mAP@[.5:.95] on 24 real COCO images, confirming model, loss, assigner, dataset, NMS, and scorer are mutually consistent. |
| **Letterbox round-trip** | Coordinate-transform errors. Forward then inverse must recover the original boxes; measured error 1.5e-05 px. A mistake here shifts every box and depresses mAP with no visible failure. |
| **Bounds assertions** | Silent broadcasting and normalization errors, by asserting each loss term stays within its mathematically valid range. |
| **bf16 step test** | Precision-dependent defects that float32 testing cannot reach. |

The single-batch overfit test is the highest-value check in the suite: it is
cheap, it runs in seconds, and it is the one test the original implementation
could not have passed.
