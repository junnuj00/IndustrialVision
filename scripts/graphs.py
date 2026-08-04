import os

import pandas as pd
import matplotlib.pyplot as plt


# =========================
# Path
# =========================

CSV_PATH = "outputs/metrics/performance.csv"

OUTPUT_DIR = "assets"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# =========================
# Load CSV
# =========================

df = pd.read_csv(
    CSV_PATH
)


methods = df["Mode"]

detection = df["Detection Count"]

confidence = df["Average Confidence"]

fps = df["Average FPS"]


# =========================
# Figure 1
# Detection Performance
# =========================

fig, axes = plt.subplots(
    1,
    2,
    figsize=(12, 5)
)


bars = axes[0].bar(
    methods,
    detection
)

axes[0].set_title(
    "Detection Count"
)

axes[0].set_ylabel(
    "Count"
)

axes[0].grid(
    axis="y",
    alpha=0.3
)


for bar in bars:

    height = bar.get_height()

    axes[0].text(

        bar.get_x() + bar.get_width()/2,

        height,

        f"{int(height)}",

        ha="center",

        va="bottom"

    )


bars = axes[1].bar(
    methods,
    confidence
)

axes[1].set_title(
    "Average Confidence"
)

axes[1].set_ylabel(
    "Confidence"
)

axes[1].grid(
    axis="y",
    alpha=0.3
)


for bar in bars:

    height = bar.get_height()

    axes[1].text(

        bar.get_x() + bar.get_width()/2,

        height,

        f"{height:.3f}",

        ha="center",

        va="bottom"

    )


plt.suptitle(
    "Detection Performance Comparison",
    fontsize=16,
    fontweight="bold"
)

plt.tight_layout()

plt.savefig(

    os.path.join(

        OUTPUT_DIR,

        "figure1_detection_performance.png"

    ),

    dpi=300

)

plt.close()


# =========================
# Figure 2
# FPS
# =========================

plt.figure(
    figsize=(7,5)
)


bars = plt.bar(
    methods,
    fps
)

plt.title(
    "Real-time Performance"
)

plt.ylabel(
    "Average FPS"
)

plt.grid(
    axis="y",
    alpha=0.3
)


for bar in bars:

    height = bar.get_height()

    plt.text(

        bar.get_x()+bar.get_width()/2,

        height,

        f"{height:.2f}",

        ha="center",

        va="bottom"

    )


plt.tight_layout()

plt.savefig(

    os.path.join(

        OUTPUT_DIR,

        "figure2_fps.png"

    ),

    dpi=300

)

plt.close()


# =========================
# Figure 3
# Trade-off
# =========================

plt.figure(
    figsize=(6,5)
)


plt.scatter(
    fps,
    detection,
    s=80
)


for i in range(
    len(methods)
):

    plt.annotate(

        methods[i],

        (
            fps[i],
            detection[i]
        ),

        xytext=(5,5),

        textcoords="offset points"

    )


plt.title(
    "Accuracy-Speed Trade-off"
)

plt.xlabel(
    "Average FPS"
)

plt.ylabel(
    "Detection Count"
)

plt.grid(
    alpha=0.3
)

plt.tight_layout()

plt.savefig(

    os.path.join(

        OUTPUT_DIR,

        "figure3_tradeoff.png"

    ),

    dpi=300

)

plt.close()


print(
    "Graphs generated successfully."
)