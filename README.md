# Deepfake Detection — EfficientNet-B4 + Attention Fusion

> Multi-stream deep learning system for deepfake video detection.
> **94.7% accuracy · AUC 0.987 · Trained on 1M+ videos (AV-Deepfake1M)**

## Overview

This project implements a multi-stream deepfake detection architecture combining spatial and frequency domain analysis through transformer-inspired attention fusion.

Built as part of MSc Data Science dissertation — Manchester Metropolitan University, 2025.

## Results

| Metric | Score |
|--------|-------|
| Accuracy | 94.7% |
| Precision | 93.8% |
| Recall | 95.2% |
| F1-Score | 94.5% |
| AUC | 0.987 |

| Method | Accuracy |
|--------|----------|
| RGB-only EfficientNet-B4 | 89.3% |
| Frequency-only ResNet18 | 85.7% |
| Concatenation fusion | 92.4% |
| This model attention fusion | 94.7% |

## Architecture

RGB Stream EfficientNet-B4 + Frequency Stream ResNet18 on DCT → Multi-Head Self-Attention (4 heads) → Classifier → Real / Fake

## Tech Stack

PyTorch, EfficientNet-B4, ResNet18, timm, MTCNN, Grad-CAM, OpenCV, scikit-learn

## Author

Sidhanth Chavan — MSc Data Science, Manchester Metropolitan University
linkedin.com/in/sidhanth-chavan
