# SentraAI

## Overview

SentraAI is a computer vision proof-of-concept designed for security camera environments.

Given two images captured by the same static camera a few seconds apart, the system determines whether the detected difference is caused by actual object movement or by changes in lighting conditions such as shadows, reflections, or sunlight variations.

The system analyzes structural, geometric, and photometric characteristics of the scene, using techniques such as SSIM, edge analysis, photometric normalization, and texture evaluation to distinguish between real scene changes and illumination effects.

## Features

* Motion vs. lighting change classification
* Photometric compensation for illumination variations
* Edge-based structural analysis
* SSIM-based change detection
* Confidence scoring
* Human-readable decision explanations
* Automatic visualization generation
* Support for single-image and batch processing workflows

## Technologies

* Python
* OpenCV
* NumPy
* Scikit-Image
* SciPy
* Matplotlib

## Installation

Clone the repository:

```bash
git clone https://github.com/sariezen/SentraAI.git
cd SentraAI/SentraAI_project
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run the detector on two images:

```bash
python main.py image1.jpg image2.jpg
```

Save the generated visualization:

```bash
python main.py image1.jpg image2.jpg --save result.png
```

Run without opening the visualization window:

```bash
python main.py image1.jpg image2.jpg --no-show
```
