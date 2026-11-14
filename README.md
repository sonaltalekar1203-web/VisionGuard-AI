# VisionGuard AI

AI-powered industrial defect detection and predictive quality intelligence using PatchCore and computer vision.

## Overview

VisionGuard AI is a prototype industrial visual-inspection system that identifies anomalous products, highlights suspicious regions, records inspection history, and generates PDF inspection reports.

## Features

- PatchCore-based anomaly detection
- Defect heatmaps and localization overlays
- FastAPI prediction backend
- Streamlit inspection dashboard
- SQLite inspection history
- PDF inspection reports
- Quality analytics and CSV export

## Technology Stack

- Python
- PyTorch
- Anomalib
- PatchCore
- FastAPI
- Streamlit
- OpenCV
- SQLite
- Plotly
- ReportLab

## Dataset

This project uses the MVTec Anomaly Detection (MVTec AD) dataset.

The current trained model focuses on the `metal_nut` category.

## Preliminary Results

| Metric | Result |
|---|---:|
| Image-level ROC-AUC | 0.998 |
| Image-level Average Precision | 0.9996 |
| Pixel-level ROC-AUC | 0.9862 |
| Pixel-level Average Precision | 0.8859 |

These are development results on the dataset used during experimentation. They should not be interpreted as independent validation or production performance.

## Important Limitations

- Current implementation is trained on the metal_nut category.
- The decision threshold is provisional.
- Independent validation is required before industrial deployment.
- The trained model checkpoint and original dataset are not included in this repository.

## Project Structure

```text
VisionGuard-AI/
├── backend/
├── frontend/
├── models/
├── uploads/
├── outputs/
├── reports/
├── database/
├── tests/
└── README.md
