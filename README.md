# Custard Apple Disease Detection Web Application

This project implements a deep learning-based web application for identifying diseases in custard apple fruit and leaves. The application allows users to upload images and receive top-3 predicted disease classes with confidence scores.

## Features

- Upload image of custard apple fruit or leaf
- Select between DenseNet121 and Xception models
- Real-time disease prediction
- Display top-3 predicted classes with confidence scores
- Unified display for Cylindrocladium leaf spot (fruit and leaf)
- Responsive web interface (Streamlit-based)

## Dataset

The dataset used in this project is from:

**R. Rane et al., "A comprehensive image dataset for identifying diseases in custard apple (Annona squamosa) fruits and leaves," Data in Brief, vol. 54, 109308, 2024.**

Dataset contains 8,226 images across 6 classes.

## Installation

```bash
pip install -r requirements.txt
