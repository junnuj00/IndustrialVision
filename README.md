## Experimental Results

### Quantitative Comparison

| Preprocessing Method | Detection Count | Average Confidence | Average FPS | Remark |
|----------------------|----------------:|-------------------:|------------:|--------|
| Original | 243 | 0.635 | **25.01** | Fastest |
| Gaussian Blur | 230 | 0.634 | 23.61 | Slight smoothing |
| **CLAHE** | **256** | **0.679** | 19.94 | **Best Detection** |
| Histogram Equalization | 79 | 0.640 | 24.43 | Lowest Detection |

### Detection Performance

![Detection Performance](assets/figure1_detection_performance.png)

CLAHE achieved the highest detection count and average confidence.

### Real-time Performance

![Real-time Performance](assets/figure2_fps.png)

The original input achieved the highest average FPS.

### Accuracy-Speed Trade-off

![Accuracy-Speed Trade-off](assets/figure3_tradeoff.png)

CLAHE improved detection performance at the cost of processing speed.

### Graph Generation

The figures were generated from `outputs/metrics/performance.csv` using Matplotlib.

```bash
python scripts/graphs.py