# Deepfake Detection Prototype

An MSc dissertation prototype for frame-level deepfake classification. The project combines an EfficientNet-B4 image backbone with a ResNet-18 branch and explores feature fusion, face-focused preprocessing and Grad-CAM visualisation.

This repository is an academic research artefact, not a production detector. The documentation below reflects the experiment preserved in the notebook rather than the full scale of the source dataset.

## Experiment scope

The notebook processed a subset containing 1,363 videos:

| Split | Real | Fake | Total |
|---|---:|---:|---:|
| Train | 254 | 700 | 954 |
| Validation | 54 | 150 | 204 |
| Test | 55 | 150 | 205 |

Ten frames were sampled from each video before face detection. The dataset class then balanced real and fake frames for training and evaluation.

The saved notebook reports approximately 86% frame-level accuracy on 1,094 balanced test frames. This is not a video-level benchmark and should not be compared directly with results obtained on the complete AV-Deepfake1M dataset.

## Pipeline

```text
Video subset
    -> evenly spaced frame sampling
    -> MTCNN face detection
    -> image normalisation and augmentation
    -> EfficientNet-B4 / ResNet-18 feature extraction
    -> feature fusion and classification
    -> frame-level evaluation and Grad-CAM inspection
```

## Repository layout

```text
.
├── models/              # Backbone and fusion modules
├── preprocessing/       # Dataset and image transforms
├── training/            # Training and validation loops
├── evaluation/          # Classification metrics and ROC utilities
├── interpretability/    # Grad-CAM implementation
├── notebooks/           # Original Colab experiment
└── Sidhanth_Chavan_Deepfake_Dissertation.pdf
```

## Local setup

Python 3.10 or 3.11 is recommended.

```bash
git clone https://github.com/SidhanthChavan/deepfake-detection-efficientnet.git
cd deepfake-detection-efficientnet
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The original notebook expects video archives in Google Drive and writes extracted frames under `/content`. Those paths must be changed for local execution. Dataset files and trained weights are not distributed in this repository.

## Known limitations

- The published experiment uses a small subset of the source dataset.
- Evaluation is frame-level; frames from the same video are not aggregated into one prediction.
- The modular dataset currently supplies the same RGB tensor to both model branches. A genuine DCT or frequency transform is not yet implemented there.
- The attention layer operates after feature concatenation rather than across separate stream tokens.
- Training seeds, model weights and a complete end-to-end runner are not included.

These limitations need to be addressed before treating the repository as a reproducible benchmark or deployable detector.

## Stack

Python, PyTorch, torchvision, EfficientNet-B4, ResNet-18, MTCNN, OpenCV, scikit-learn and Matplotlib.

## Licence

See [LICENSE](LICENSE).
