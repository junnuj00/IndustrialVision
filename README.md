# Experimental Results

The preprocessing performance was evaluated using the same input video and the YOLO11n model. Detection accuracy and real-time performance were compared across four preprocessing methods.

## Table 1. Quantitative Comparison

| Preprocessing Method | Detection Count | Average Confidence | Average FPS | Remark |
|----------------------|----------------:|-------------------:|------------:|--------|
| Original | 243 | 0.635 | **25.01** | Fastest |
| Gaussian Blur | 230 | 0.634 | 23.61 | Slight smoothing |
| **CLAHE** | **256** | **0.679** | 19.94 | **Best Detection** |
| Histogram Equalization | 79 | 0.640 | 24.43 | Lowest Detection |

---

## Figure 1. Detection Performance

![Detection Performance](assets/figure1_detection_performance.png)

CLAHE achieved the highest number of detected objects and the highest average confidence, indicating improved detection performance through local contrast enhancement.

---

## Figure 2. Real-time Performance

![Real-time Performance](assets/figure2_fps.png)

The Original input achieved the highest processing speed, while CLAHE required additional computation, resulting in lower FPS.

---

## Figure 3. Accuracy–Speed Trade-off

![Trade-off](assets/figure3_tradeoff.png)

The trade-off analysis illustrates that CLAHE provides superior detection performance at the expense of inference speed, whereas the Original input maintains the highest real-time efficiency.

---

## Visualization

The experimental graphs were automatically generated from the recorded metrics using **Matplotlib**.

```bash
python scripts/graphs.py
```

The script reads the experimental results stored in `outputs/metrics/performance.csv` and generates all figures included in this README.