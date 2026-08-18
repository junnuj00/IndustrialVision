from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# ============================================
# Paths
# ============================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CSV_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "metrics"
    / "performance.csv"
)

ASSETS_DIR = (
    PROJECT_ROOT
    / "assets"
)

ASSETS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================
# Mode Display Names
# ============================================

MODE_NAMES = {
    "original": "Original",
    "gaussian": "Gaussian Blur",
    "clahe": "CLAHE",
    "histogram": "Histogram Eq."
}


# ============================================
# Load Data
# ============================================

def load_data():

    if not CSV_PATH.exists():

        raise FileNotFoundError(
            f"CSV file not found: {CSV_PATH}"
        )


    df = pd.read_csv(
        CSV_PATH
    )


    required_columns = {
        "Mode",
        "Detection Count",
        "Average Confidence",
        "Average FPS"
    }


    missing_columns = (
        required_columns
        - set(df.columns)
    )


    if missing_columns:

        raise ValueError(
            "Missing required CSV columns: "
            + ", ".join(
                sorted(missing_columns)
            )
        )


    return df


# ============================================
# Mode Labels
# ============================================

def get_mode_labels(df):

    return [
        MODE_NAMES.get(
            mode,
            mode
        )
        for mode in df["Mode"]
    ]


# ============================================
# Figure 1
# Detection Count + Confidence
# ============================================

def create_detection_performance_graph(
    df
):

    modes = get_mode_labels(
        df
    )

    detection_counts = (
        df["Detection Count"]
    )

    confidences = (
        df["Average Confidence"]
    )


    fig, ax1 = plt.subplots(
        figsize=(10, 6.5)
    )


    # ----------------------------------------
    # Detection Count
    # ----------------------------------------

    bars = ax1.bar(
        modes,
        detection_counts,
        alpha=0.8,
        label="Detection Count"
    )


    ax1.set_title(
        "Detection Performance by Preprocessing Method",
        pad=45
    )

    ax1.set_xlabel(
        "Preprocessing Method"
    )

    ax1.set_ylabel(
        "Detection Count"
    )


    # ----------------------------------------
    # Detection Count Y-axis Margin
    # ----------------------------------------

    ax1.set_ylim(
        0,
        detection_counts.max() * 1.22
    )


    ax1.grid(
        axis="y",
        alpha=0.25
    )


    # ----------------------------------------
    # Highest Detection Count
    # ----------------------------------------

    max_detection_position = (
        detection_counts
        .reset_index(
            drop=True
        )
        .idxmax()
    )


    # ----------------------------------------
    # Detection Count Labels
    # ----------------------------------------

    for position, (
        bar,
        value
    ) in enumerate(
        zip(
            bars,
            detection_counts
        )
    ):

        label = (
            f"{int(value):,}"
        )


        if (
            position
            == max_detection_position
        ):

            label += (
                "\nHighest Count"
            )


        ax1.text(
            bar.get_x()
            + bar.get_width() / 2,

            value + 18,

            label,

            ha="center",
            va="bottom",
            fontsize=9
        )


    # ----------------------------------------
    # Average Confidence Axis
    # ----------------------------------------

    ax2 = ax1.twinx()


    ax2.plot(
        modes,
        confidences,
        marker="o",
        linewidth=2,
        label="Average Confidence"
    )


    ax2.set_ylabel(
        "Average Confidence"
    )


    confidence_min = (
        max(
            0.0,
            confidences.min()
            - 0.05
        )
    )

    confidence_max = (
        min(
            1.0,
            confidences.max()
            + 0.07
        )
    )


    ax2.set_ylim(
        confidence_min,
        confidence_max
    )


    # ----------------------------------------
    # Highest Confidence
    # ----------------------------------------

    max_confidence_position = (
        confidences
        .reset_index(
            drop=True
        )
        .idxmax()
    )


    # ----------------------------------------
    # Confidence Labels
    # All labels below the points
    # ----------------------------------------

    for position, value in enumerate(
        confidences
    ):

        label = (
            f"{value:.3f}"
        )


        if (
            position
            == max_confidence_position
        ):

            label += (
                "\nHighest Confidence"
            )


        ax2.annotate(
            label,

            (
                position,
                value
            ),

            textcoords="offset points",

            xytext=(0, -12),

            ha="center",
            va="top",

            fontsize=9
        )


    # ----------------------------------------
    # Legend
    # ----------------------------------------

    handles1, labels1 = (
        ax1.get_legend_handles_labels()
    )

    handles2, labels2 = (
        ax2.get_legend_handles_labels()
    )


    ax1.legend(
        handles1 + handles2,
        labels1 + labels2,

        loc="upper center",

        bbox_to_anchor=(
            0.5,
            1.10
        ),

        ncol=2,

        frameon=True
    )


    fig.tight_layout()


    output_path = (
        ASSETS_DIR
        / "figure1_detection_performance.png"
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
# Figure 2
# FPS Performance
# ============================================

def create_fps_graph(
    df
):

    modes = get_mode_labels(
        df
    )

    fps_values = (
        df["Average FPS"]
    )


    fig, ax = plt.subplots(
        figsize=(9, 5.5)
    )


    bars = ax.bar(
        modes,
        fps_values,
        alpha=0.8
    )


    ax.set_title(
        "Real-Time Processing Performance"
    )

    ax.set_xlabel(
        "Preprocessing Method"
    )

    ax.set_ylabel(
        "Average FPS"
    )


    # ----------------------------------------
    # Y-axis Margin
    # ----------------------------------------

    ax.set_ylim(
        0,
        fps_values.max() * 1.20
    )


    ax.grid(
        axis="y",
        alpha=0.25
    )


    # ----------------------------------------
    # Highest FPS
    # ----------------------------------------

    max_fps_position = (
        fps_values
        .reset_index(
            drop=True
        )
        .idxmax()
    )


    # ----------------------------------------
    # FPS Labels
    # ----------------------------------------

    for position, (
        bar,
        value
    ) in enumerate(
        zip(
            bars,
            fps_values
        )
    ):

        label = (
            f"{value:.2f}"
        )


        if (
            position
            == max_fps_position
        ):

            label += (
                "\nHighest FPS"
            )


        ax.text(
            bar.get_x()
            + bar.get_width() / 2,

            value + 0.25,

            label,

            ha="center",
            va="bottom",
            fontsize=9
        )


    fig.tight_layout()


    output_path = (
        ASSETS_DIR
        / "figure2_fps.png"
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
# Figure 3
# Detection / Speed Trade-off
# ============================================

def create_tradeoff_graph(
    df
):

    modes = get_mode_labels(
        df
    )

    detection_counts = (
        df["Detection Count"]
    )

    fps_values = (
        df["Average FPS"]
    )

    confidences = (
        df["Average Confidence"]
    )


    fig, ax = plt.subplots(
        figsize=(9, 6)
    )


    # ----------------------------------------
    # Scatter
    # ----------------------------------------

    ax.scatter(
        fps_values,
        detection_counts,
        s=120
    )


    # ----------------------------------------
    # Point Labels
    # ----------------------------------------

    for (
        mode,
        fps,
        count,
        confidence
    ) in zip(
        modes,
        fps_values,
        detection_counts,
        confidences
    ):

        label = (
            f"{mode}\n"
            f"Conf: {confidence:.3f}"
        )


        ax.annotate(
            label,

            (
                fps,
                count
            ),

            textcoords="offset points",

            xytext=(8, 8),

            fontsize=9
        )


    # ----------------------------------------
    # Axis
    # ----------------------------------------

    ax.set_title(
        "Detection Performance-Speed Trade-off"
    )

    ax.set_xlabel(
        "Average FPS"
    )

    ax.set_ylabel(
        "Detection Count"
    )


    # ----------------------------------------
    # Axis Margins
    # ----------------------------------------

    fps_margin = (
        (
            fps_values.max()
            - fps_values.min()
        )
        * 0.15
    )

    detection_margin = (
        (
            detection_counts.max()
            - detection_counts.min()
        )
        * 0.25
    )


    ax.set_xlim(
        fps_values.min()
        - fps_margin,

        fps_values.max()
        + fps_margin
    )


    ax.set_ylim(
        detection_counts.min()
        - detection_margin,

        detection_counts.max()
        + detection_margin
    )


    ax.grid(
        alpha=0.25
    )


    # ----------------------------------------
    # Summary
    # ----------------------------------------

    best_detection_position = (
        detection_counts
        .reset_index(
            drop=True
        )
        .idxmax()
    )

    best_fps_position = (
        fps_values
        .reset_index(
            drop=True
        )
        .idxmax()
    )

    best_confidence_position = (
        confidences
        .reset_index(
            drop=True
        )
        .idxmax()
    )


    best_detection_mode = (
        modes[
            best_detection_position
        ]
    )

    best_fps_mode = (
        modes[
            best_fps_position
        ]
    )

    best_confidence_mode = (
        modes[
            best_confidence_position
        ]
    )


    summary = (
        f"Highest Detection Count: "
        f"{best_detection_mode}\n"
        f"Highest FPS: "
        f"{best_fps_mode}\n"
        f"Highest Confidence: "
        f"{best_confidence_mode}"
    )


    ax.text(
        0.02,
        0.98,

        summary,

        transform=ax.transAxes,

        ha="left",
        va="top",

        fontsize=9,

        bbox={
            "boxstyle": "round",
            "alpha": 0.1
        }
    )


    fig.tight_layout()


    output_path = (
        ASSETS_DIR
        / "figure3_tradeoff.png"
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

    print(
        "=" * 48
    )

    print(
        " Industrial Vision Graph Generator"
    )

    print(
        "=" * 48
    )


    try:

        df = load_data()


        create_detection_performance_graph(
            df
        )

        create_fps_graph(
            df
        )

        create_tradeoff_graph(
            df
        )


        print()

        print(
            "=" * 48
        )

        print(
            " Graph Generation Complete"
        )

        print(
            "=" * 48
        )


    except Exception as error:

        print(
            f"[ERROR] {error}"
        )


# ============================================
# Entry Point
# ============================================

if __name__ == "__main__":

    main()