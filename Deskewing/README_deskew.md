
# Deskew Starter (OpenCV)

This starter shows two practical ways to estimate and correct rotation (deskew) on scanned/photographed sheet music.

## Install
```bash
pip install opencv-python numpy
```

## Run
```bash
python deskew_starter.py --input path/to/your/score.png --outdir outputs/
```
Outputs:
- `0_gray.png` – grayscale
- `1_binary.png` – thresholded image
- `2_deskewed.png` – rotated (corrected) image
- `3_comparison.png` – before vs after

## How it works

1. **Binarization**: Otsu and Adaptive Threshold are tried; the one with more black pixels (likely staff lines) is chosen.
2. **Angle (Hough)**: Emphasizes horizontal lines with morphology, runs Hough Lines, turns `theta` into a near-horizontal angle, and uses the median.
3. **Angle (minAreaRect)**: As a fallback, finds connected components and uses the orientation of each blob; again, median for robustness.
4. **Rotation**: Uses `cv2.getRotationMatrix2D` + `cv2.warpAffine` and computes a larger output canvas to avoid cropped edges.

## Tips for Music Scores

- If Hough fails, try changing the morphological kernel height (e.g., `(61, 3)` or `(41, 5)`).
- If the image is low-contrast, consider CLAHE before thresholding:
  ```python
  clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
  gray = clahe.apply(gray)
  ```
- Preserve DPI/scale metadata if you plan downstream measurements.
- After deskew, you can proceed with staff line detection (Hough or morphological projection) and then symbol detection.

> **Tip:** to launch the whole Secundo app (backend + frontend) in one step, double-click `launch.bat` (Windows) or `launch.command` (macOS) in the project root. See the main README.
