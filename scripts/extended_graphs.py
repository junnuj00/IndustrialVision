from pathlib import Path

import matplotlib.pyplot as plt


# ============================================
# Paths
# ============================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

ASSETS_DIR = (
    PROJECT_ROOT
    / "assets"
)

ASSETS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================
# Phase 2
# Detection Robustness Graph
# ============================================

def create_robustness_graph():

    conditions = [
        "Original",
        "Low-light",
        "Low-light\n+ CLAHE",
        "Low-light\n+ Gamma"
    ]

    map50_95 = [
        0.5074,
        0.4490,
        0.4454,
        0.4647
    ]


    fig, ax = plt.subplots(
        figsize=(9, 5.5)
    )


    bars = ax.bar(
        conditions,
        map50_95
    )


    ax.set_title(
        "YOLO11n Detection Robustness under Low-Light Conditions"
    )

    ax.set_ylabel(
        "mAP50-95"
    )

    ax.set_ylim(
        0,
        0.6
    )


    # ----------------------------------------
    # Value Labels
    # ----------------------------------------

    for bar, value in zip(
        bars,
        map50_95
    ):

        ax.text(
            bar.get_x()
            + bar.get_width() / 2,

            value + 0.01,

            f"{value:.4f}",

            ha="center",
            va="bottom"
        )


    ax.grid(
        axis="y",
        alpha=0.25
    )


    fig.tight_layout()


    output_path = (
        ASSETS_DIR
        / "figure4_lowlight_robustness.png"
    )


    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )


    plt.close(
        fig
    )


    print(
        f"Saved: {output_path}"
    )


# ============================================
# Phase 3
# VLM Evaluation Graph
# ============================================

def create_vlm_graph():

    metrics = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1-score"
    ]

    values = [
        0.9786,
        0.9231,
        0.8571,
        0.8889
    ]


    fig, ax = plt.subplots(
        figsize=(9, 5.5)
    )


    bars = ax.bar(
        metrics,
        values
    )


    ax.set_title(
        "Qwen2.5-VL Ground-Truth Evaluation"
    )

    ax.set_ylabel(
        "Score"
    )

    ax.set_ylim(
        0,
        1.05
    )


    # ----------------------------------------
    # Value Labels
    # ----------------------------------------

    for bar, value in zip(
        bars,
        values
    ):

        ax.text(
            bar.get_x()
            + bar.get_width() / 2,

            value + 0.015,

            f"{value:.4f}",

            ha="center",
            va="bottom"
        )


    ax.grid(
        axis="y",
        alpha=0.25
    )


    fig.tight_layout()


    output_path = (
        ASSETS_DIR
        / "figure5_vlm_evaluation.png"
    )


    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )


    plt.close(
        fig
    )


    print(
        f"Saved: {output_path}"
    )


# ============================================
# Main
# ============================================

def main():

    print("=" * 60)
    print("Generating Extended Evaluation Figures")
    print("=" * 60)


    create_robustness_graph()

    create_vlm_graph()


    print()
    print("=" * 60)
    print("Graph Generation Complete")
    print("=" * 60)


if __name__ == "__main__":

    main()