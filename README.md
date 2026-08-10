# Industrial Vision Inspection System

A real-time industrial vision inspection system using **YOLO11n**, **OpenCV**, and multiple image preprocessing techniques.

This project implements a modular computer vision pipeline and quantitatively evaluates how different preprocessing methods affect **object detection performance** and **real-time processing speed**.

---

## Project Overview

The objective of this project is to investigate the impact of image preprocessing on YOLO11n object detection performance under consistent video conditions.

The system supports video input, preprocessing, object detection, performance measurement, CSV logging, and automated visualization.

Four preprocessing modes were compared:

* Original
* Gaussian Blur
* CLAHE
* Histogram Equalization

Performance was evaluated using:

* Detection Count
* Average Confidence
* Average FPS

---

## System Architecture

```text
Video Input
    ↓
Frame Acquisition
    ↓
Preprocessing
    ↓
YOLO11n Detection
    ↓
Performance Measurement
    ↓
CSV Logging
    ↓
Performance Visualization
```

The pipeline separates input handling, preprocessing, detection, performance measurement, and result analysis into independent modules.

This modular structure makes individual components easier to modify, test, and maintain.

---

## Project Structure

```text
IndustrialVision/
│
├── assets/
│   ├── figure1_detection_performance.png
│   ├── figure2_fps.png
│   └── figure3_tradeoff.png
│
├── core/
│   ├── camera.py
│   ├── detector.py
│   └── preprocessor.py
│
├── inputs/
│   ├── images/
│   └── videos/
│
├── models/
│   └── yolo11n.pt
│
├── outputs/
│   ├── images/
│   └── metrics/
│       └── performance.csv
│
├── scripts/
│   └── graphs.py
│
├── utils/
│   ├── csv_writer.py
│   ├── fps.py
│   └── metrics.py
│
├── config.py
├── main.py
├── README.md
└── .gitignore
```

---

## Implementation

### YOLO11n Inference

YOLO11n is integrated into the video processing pipeline for frame-level object detection.

The system extracts:

* Bounding boxes
* Class labels
* Confidence scores

### Video Processing

OpenCV is used for:

* Video frame acquisition
* Image preprocessing
* Detection visualization
* Result image handling

### Image Preprocessing

The pipeline supports four preprocessing modes:

**Original**

Uses the input frame without additional preprocessing.

**Gaussian Blur**

Applies smoothing to reduce image noise.

**CLAHE**

Applies Contrast Limited Adaptive Histogram Equalization to locally enhance image contrast.

**Histogram Equalization**

Applies global histogram equalization to modify image contrast.

### Performance Measurement

The system measures:

* Detection Count
* Average Confidence
* Average FPS

Experiment results are aggregated and stored in:

```text
outputs/metrics/performance.csv
```

### Automated Visualization

Experimental results are converted into comparative performance graphs using **Matplotlib**.

The visualization script is located at:

```text
scripts/graphs.py
```

---

## Experiment Design

All preprocessing methods were evaluated under consistent conditions.

### Test Condition

* Same input video
* Same YOLO11n model
* Same detection pipeline

### Preprocessing Comparison

```text
Original
Gaussian Blur
CLAHE
Histogram Equalization
```

### Evaluation Metrics

| Metric             | Purpose                                   |
| ------------------ | ----------------------------------------- |
| Detection Count    | Compare the number of detected objects    |
| Average Confidence | Compare detector confidence               |
| Average FPS        | Evaluate real-time processing performance |

### Experiment Flow

```text
Same Input Video
       ↓
Apply Preprocessing
       ↓
YOLO11n Detection
       ↓
Measure Performance
       ↓
Compare Results
```

---

## Test Video

The experiments were conducted using the same video across all preprocessing modes to maintain consistent evaluation conditions.

**Source:** [Pexels - Video #10472351](https://www.pexels.com/ko-kr/video/10472351/)

---

## Experimental Results

### Quantitative Comparison

| Preprocessing Method   | Detection Count | Average Confidence | Average FPS | Remark             |
| ---------------------- | --------------: | -----------------: | ----------: | ------------------ |
| Original               |             243 |              0.635 |   **25.01** | Fastest            |
| Gaussian Blur          |             230 |              0.634 |       23.61 | Slight smoothing   |
| **CLAHE**              |         **256** |          **0.679** |       19.94 | **Best Detection** |
| Histogram Equalization |              79 |              0.640 |       24.43 | Lowest Detection   |

---

### Detection Performance

![Detection Performance](assets/figure1_detection_performance.png)

CLAHE achieved the highest detection count (**256**) and average confidence (**0.679**) under the tested video condition.

Compared with the Original mode, CLAHE increased the detection count from **243 to 256** while also producing a higher average confidence.

---

### Real-time Performance

![Real-time Performance](assets/figure2_fps.png)

The Original mode achieved the highest processing speed at **25.01 FPS**.

CLAHE achieved **19.94 FPS**, showing that additional preprocessing introduced computational overhead.

---

### Detection Performance-Speed Trade-off

![Detection Performance-Speed Trade-off](assets/figure3_tradeoff.png)

The experiment demonstrates a trade-off between detection performance and processing speed.

CLAHE produced the strongest detection results under the tested condition, while the Original mode maintained the highest real-time processing speed.

This indicates that preprocessing methods should be selected according to the performance requirements of the target application.

---

## Graph Generation

The figures are generated from:

```text
outputs/metrics/performance.csv
```

using the Matplotlib visualization script:

```text
scripts/graphs.py
```

Run:

```bash
python scripts/graphs.py
```

The following figures are generated automatically:

```text
assets/
├── figure1_detection_performance.png
├── figure2_fps.png
└── figure3_tradeoff.png
```

---

## Key Findings

* **CLAHE** achieved the highest detection count and average confidence.
* **Original** achieved the highest real-time processing speed.
* Gaussian Blur showed similar detection confidence to the Original mode with slightly lower processing speed.
* Histogram Equalization significantly reduced the detection count under the tested condition.
* The experiment showed a measurable trade-off between detection performance and processing speed.

---

## What I Learned

Through this project, I gained practical experience in building an end-to-end real-time computer vision pipeline integrating:

* YOLO11n object detection
* OpenCV video processing
* Image preprocessing
* Performance measurement
* CSV-based experiment logging
* Matplotlib data visualization

The project also demonstrated that preprocessing should be evaluated from both **detection performance** and **computational efficiency**, rather than detection results alone.

---

## Future Work

Future improvements include:

* Training and evaluating the system on a custom industrial defect dataset
* Extending evaluation with Precision, Recall, and mAP using ground-truth annotations
* Optimizing inference performance for edge environments
* Extending the pipeline toward ROI-based industrial inspection

---

## Tech Stack

**Language**

* Python

**Computer Vision / AI**

* YOLO11n
* Ultralytics
* OpenCV

**Data Analysis & Visualization**

* NumPy
* Pandas
* Matplotlib

**Development**

* Git
* GitHub
* VS Code

---

## How to Run

### 1. Create a virtual environment

```bash
python -m venv cv_env
```

### 2. Activate the environment

Windows:

```bash
cv_env\Scripts\activate
```

### 3. Install required packages

```bash
pip install ultralytics opencv-python numpy pandas matplotlib
```

### 4. Run the vision pipeline

```bash
python main.py
```

### 5. Generate performance graphs

After generating `performance.csv`:

```bash
python scripts/graphs.py
```

The generated figures will be saved in the `assets/` directory.

---

## Notes

YOLO model weights and input video files may be excluded from the repository through `.gitignore` because of their file size.

The experiment results presented in this repository were obtained from the test video and experimental conditions described above.
