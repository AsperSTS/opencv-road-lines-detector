# Highway Lane Detection with Image Processing Operators

This project implements a Python program to detect highway lane lines in videos using custom image processing operators. The focus is on using point operators (such as gamma correction, histogram equalization, and color masks) without relying on high-level OpenCV functions.

## Features
- **Gamma Correction:** Enhance the brightness and contrast of frames.
- **Histogram Equalization:** Improve the visibility of lane lines.
- **Color Masking:** Detect white and yellow lane lines based on defined color ranges.
- **Polygon Mask:** Define a region of interest (ROI) to focus on lane lines.
- **Frame Binarization:** Convert frames to a binary format for easy processing.

## Dependencies
The project requires the following libraries:
- `opencv-python`
- `numpy`

You can install them using `pip`:
```bash
pip install opencv-python numpy
