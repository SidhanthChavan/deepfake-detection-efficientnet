# Frame-level evaluation evidence

Reviewed against the original notebook and modular dataset code on 27 October
2026. This describes preserved outputs, not a fresh training run.

The documented video subset has 1,363 videos: 954 train, 204 validation and 205
test. The project documents sampling ten frames per video before face detection.
The notebook loads processed JPGs from separate train, validation and test
directories. The final balanced dataset implementation randomly samples the
smaller class count from each class and shuffles the result without a fixed seed.

The saved classification report contains 547 real and 547 fake test frames,
1,094 total. Its confusion matrix is:

```text
                 Predicted real  Predicted fake
Actual real             444            103
Actual fake              45            502
```

Thus `(444 + 502) / 1094 = 0.8647166362`, approximately 86.5% frame-level
accuracy. The report formats this as 0.86. Frames are not videos or independent
people, and this is not evaluation on the complete AV-Deepfake1M dataset.

Predictions are obtained with `model.eval()`, `torch.no_grad()` and class-logit
argmax, then passed to scikit-learn classification metrics. The saved ROC AUC
uses hard predicted classes rather than scores; it should not be presented as
a robust probability-based ROC evaluation.

## Reproducibility limits

- Original video archives, processed frames, trained weights and sampled-frame
  manifests are not distributed here; evaluation cannot be reproduced from
  this repository alone.
- The split-generation history and source-video/identity disjointness are not
  independently established by the saved outputs. Do not claim leakage-free,
  identity-independent or video-independent performance.
- The dataset supplies the same RGB image to both branches; a true frequency
  transform is not implemented in the modular pipeline.
- Class balancing and training lack a fully preserved seeded execution path.

Before a new benchmark, save source-video/identity groups, split manifests,
sampling seeds, face-extraction failures, preprocessing configuration and model
weights. Evaluate per-frame probabilities and aggregated per-video predictions
on a fixed untouched test set, reporting the two separately.
