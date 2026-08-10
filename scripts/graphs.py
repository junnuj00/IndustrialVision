from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure


BASE_DIR = Path(__file__).resolve().parent.parent
CSV_PATH = BASE_DIR / "outputs" / "metrics" / "performance.csv"
OUTPUT_DIR = BASE_DIR / "assets"

MAIN_COLOR = "#4C78A8"
SECONDARY_COLOR = "#C9CED6"
GRID_COLOR = "#E5E7EB"
TEXT_COLOR = "#333333"
ANNOTATION_COLOR = "#666666"

DPI = 300
TITLE_SIZE = 16
FIGURE_TITLE_SIZE = 18
LABEL_SIZE = 12
VALUE_SIZE = 10

REQUIRED_COLUMNS = {
    "Mode",
    "Detection Count",
    "Average Confidence",
    "Average FPS",
}


plt.rcParams.update(
    {
        "font.size": 11,
        "axes.titlesize": TITLE_SIZE,
        "axes.titleweight": "bold",
        "axes.labelsize": LABEL_SIZE,
        "axes.labelcolor": TEXT_COLOR,
        "axes.edgecolor": TEXT_COLOR,
        "xtick.color": TEXT_COLOR,
        "ytick.color": TEXT_COLOR,
        "text.color": TEXT_COLOR,
        "axes.facecolor": "white",
        "figure.facecolor": "white",
    }
)


def load_data() -> tuple[list[str], list[int], list[float], list[float]]:
    """Load performance metrics from CSV."""

    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"Performance CSV not found: {CSV_PATH}\n"
            "Run main.py and create performance.csv first."
        )

    methods: list[str] = []
    detection: list[int] = []
    confidence: list[float] = []
    fps: list[float] = []

    with CSV_PATH.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        if reader.fieldnames is None:
            raise ValueError("performance.csv has no header.")

        missing_columns = REQUIRED_COLUMNS - set(reader.fieldnames)

        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"Missing required CSV columns: {missing}")

        for row_number, row in enumerate(reader, start=2):
            try:
                methods.append(row["Mode"].strip().lower())
                detection.append(int(row["Detection Count"]))
                confidence.append(float(row["Average Confidence"]))
                fps.append(float(row["Average FPS"]))

            except (TypeError, ValueError, KeyError) as error:
                raise ValueError(
                    f"Invalid data in performance.csv at row {row_number}."
                ) from error

    if not methods:
        raise ValueError(
            "performance.csv contains no experiment results."
        )

    return methods, detection, confidence, fps


def get_colors(
    methods: Sequence[str],
    highlight: str,
) -> list[str]:
    """Return colors with one highlighted preprocessing method."""

    return [
        MAIN_COLOR
        if method.lower() == highlight.lower()
        else SECONDARY_COLOR
        for method in methods
    ]


def display_names(
    methods: Iterable[str],
) -> list[str]:
    """Convert internal mode names into readable graph labels."""

    name_map = {
        "original": "Original",
        "gaussian": "Gaussian",
        "clahe": "CLAHE",
        "histogram": "Histogram EQ",
    }

    return [
        name_map.get(method.lower(), method.title())
        for method in methods
    ]


def style_axis(
    ax: Axes,
    grid_axis: str = "y",
) -> None:
    """Apply a clean GitHub technical-blog style to an axis."""

    ax.grid(
        axis=grid_axis,
        linestyle="--",
        linewidth=0.8,
        color=GRID_COLOR,
        alpha=0.8,
        zorder=0,
    )

    ax.set_axisbelow(True)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def add_value_labels(
    ax: Axes,
    bars,
    value_format: str,
    offset_ratio: float = 0.015,
) -> None:
    """Display numeric values above bars."""

    heights = [
        bar.get_height()
        for bar in bars
    ]

    max_height = max(heights) if heights else 0
    offset = max_height * offset_ratio

    for bar in bars:
        height = bar.get_height()

        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + offset,
            format(height, value_format),
            ha="center",
            va="bottom",
            fontsize=VALUE_SIZE,
            fontweight="bold",
            color=TEXT_COLOR,
        )


def draw_bar_chart(
    ax: Axes,
    methods: Sequence[str],
    values: Sequence[float],
    title: str,
    ylabel: str,
    highlight: str,
    value_format: str,
) -> None:
    """Draw a styled bar chart."""

    labels = display_names(methods)
    colors = get_colors(
        methods,
        highlight,
    )

    bars = ax.bar(
        labels,
        values,
        color=colors,
        width=0.68,
        edgecolor="white",
        linewidth=0.8,
        zorder=2,
    )

    ax.set_title(
        title,
        pad=12,
    )

    ax.set_ylabel(
        ylabel,
    )

    style_axis(ax)

    add_value_labels(
        ax,
        bars,
        value_format,
    )

    max_value = max(values)

    ax.set_ylim(
        0,
        max_value * 1.18,
    )


def save_figure(
    fig: Figure,
    filename: str,
) -> None:
    """Save and close a Matplotlib figure."""

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = OUTPUT_DIR / filename

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=DPI,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close(fig)


def add_best_annotation(
    ax: Axes,
    index: int,
    value: float,
    label: str,
    offset_ratio: float,
) -> None:
    """Add a small highlight label above the best bar."""

    ax.text(
        index,
        value * (1 + offset_ratio),
        label,
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold",
        color=MAIN_COLOR,
    )


def create_detection_figure(
    methods: Sequence[str],
    detection: Sequence[int],
    confidence: Sequence[float],
) -> None:
    """Create detection count and average confidence comparison."""

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 5),
    )

    draw_bar_chart(
        ax=axes[0],
        methods=methods,
        values=detection,
        title="Detection Count",
        ylabel="Detected Objects",
        highlight="clahe",
        value_format=".0f",
    )

    best_detection_index = detection.index(
        max(detection)
    )

    add_best_annotation(
        axes[0],
        best_detection_index,
        detection[best_detection_index],
        "Best",
        0.10,
    )

    draw_bar_chart(
        ax=axes[1],
        methods=methods,
        values=confidence,
        title="Average Confidence",
        ylabel="Confidence Score",
        highlight="clahe",
        value_format=".3f",
    )

    best_confidence_index = confidence.index(
        max(confidence)
    )

    add_best_annotation(
        axes[1],
        best_confidence_index,
        confidence[best_confidence_index],
        "Highest",
        0.10,
    )

    fig.suptitle(
        "Figure 1. Detection Performance Comparison",
        fontsize=FIGURE_TITLE_SIZE,
        fontweight="bold",
        y=1.02,
    )

    save_figure(
        fig,
        "figure1_detection_performance.png",
    )


def create_fps_figure(
    methods: Sequence[str],
    fps: Sequence[float],
) -> None:
    """Create average FPS comparison."""

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    draw_bar_chart(
        ax=ax,
        methods=methods,
        values=fps,
        title="Average Processing Speed",
        ylabel="Frames Per Second",
        highlight="original",
        value_format=".2f",
    )

    fastest_index = fps.index(
        max(fps)
    )

    add_best_annotation(
        ax,
        fastest_index,
        fps[fastest_index],
        "Fastest",
        0.10,
    )

    fig.suptitle(
        "Figure 2. Real-time Performance",
        fontsize=FIGURE_TITLE_SIZE,
        fontweight="bold",
        y=1.02,
    )

    save_figure(
        fig,
        "figure2_fps.png",
    )


def create_tradeoff_figure(
    methods: Sequence[str],
    detection: Sequence[int],
    fps: Sequence[float],
) -> None:
    """Create a detection performance-speed trade-off scatter plot."""

    fig, ax = plt.subplots(
        figsize=(8, 6)
    )

    labels = display_names(
        methods
    )

    colors = get_colors(
        methods,
        "clahe",
    )

    ax.scatter(
        fps,
        detection,
        s=135,
        c=colors,
        edgecolors="white",
        linewidths=1.5,
        zorder=3,
    )

    for index, label in enumerate(labels):
        ax.annotate(
            label,
            (
                fps[index],
                detection[index],
            ),
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=10,
            color=TEXT_COLOR,
        )

    if "clahe" in methods:
        clahe_index = methods.index(
            "clahe"
        )

        ax.annotate(
            "Detection priority",
            (
                fps[clahe_index],
                detection[clahe_index],
            ),
            xytext=(-80, 45),
            textcoords="offset points",
            arrowprops={
                "arrowstyle": "->",
                "color": MAIN_COLOR,
                "linewidth": 1.5,
            },
            fontsize=10,
            fontweight="bold",
            color=MAIN_COLOR,
        )

    if "original" in methods:
        original_index = methods.index(
            "original"
        )

        ax.annotate(
            "Speed priority",
            (
                fps[original_index],
                detection[original_index],
            ),
            xytext=(-80, -42),
            textcoords="offset points",
            arrowprops={
                "arrowstyle": "->",
                "color": ANNOTATION_COLOR,
                "linewidth": 1.5,
            },
            fontsize=10,
            fontweight="bold",
            color=ANNOTATION_COLOR,
        )

    ax.set_xlabel(
        "Average FPS"
    )

    ax.set_ylabel(
        "Detection Count"
    )

    ax.set_title(
        "Detection Performance vs. Processing Speed",
        pad=12,
    )

    style_axis(
        ax,
        grid_axis="both",
    )

    x_margin = max(fps) - min(fps)
    y_margin = max(detection) - min(detection)

    ax.set_xlim(
        min(fps) - max(
            x_margin * 0.15,
            0.5,
        ),
        max(fps) + max(
            x_margin * 0.15,
            0.5,
        ),
    )

    ax.set_ylim(
        min(detection) - max(
            y_margin * 0.12,
            10,
        ),
        max(detection) + max(
            y_margin * 0.18,
            10,
        ),
    )

    fig.suptitle(
        "Figure 3. Detection Performance-Speed Trade-off",
        fontsize=FIGURE_TITLE_SIZE,
        fontweight="bold",
        y=1.02,
    )

    save_figure(
        fig,
        "figure3_tradeoff.png",
    )


def main() -> None:
    """Generate all README figures."""

    print("=" * 48)
    print(" Industrial Vision Graph Generator")
    print("=" * 48)

    try:
        methods, detection, confidence, fps = load_data()

        print(
            f"[INFO] Loaded metrics from: {CSV_PATH}"
        )

        create_detection_figure(
            methods,
            detection,
            confidence,
        )

        print(
            "[INFO] Figure 1 created."
        )

        create_fps_figure(
            methods,
            fps,
        )

        print(
            "[INFO] Figure 2 created."
        )

        create_tradeoff_figure(
            methods,
            detection,
            fps,
        )

        print(
            "[INFO] Figure 3 created."
        )

    except (
        FileNotFoundError,
        ValueError,
    ) as error:
        print(
            f"[ERROR] {error}"
        )

        return

    print("=" * 48)
    print(" Graph generation complete")
    print("=" * 48)

    print(
        f"Saved location: {OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()