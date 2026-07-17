
import io

import matplotlib
matplotlib.use("Agg")  # non-interactive backend, required for server-side rendering
import matplotlib.pyplot as plt

from app.core.logging_config import get_logger

logger = get_logger(__name__)

SUPPORTED_CHART_TYPES = {"bar", "line", "pie"}


class ChartGenerationError(Exception):
    pass


def generate_chart(spec: dict) -> bytes:
    """
    spec expected shape:
    {
        "type": "bar" | "line" | "pie",
        "title": str,
        "labels": [str, ...],
        "values": [float, ...],
        "x_label": str (optional),
        "y_label": str (optional)
    }
    Returns PNG image bytes.
    """
    chart_type = spec.get("type", "bar").lower()
    if chart_type not in SUPPORTED_CHART_TYPES:
        raise ChartGenerationError(f"Unsupported chart type: {chart_type}")

    labels = spec.get("labels", [])
    values = spec.get("values", [])
    if not labels or not values or len(labels) != len(values):
        raise ChartGenerationError("labels and values must be non-empty and equal length")

    fig, ax = plt.subplots(figsize=(6, 4), dpi=120)

    try:
        if chart_type == "bar":
            ax.bar(labels, values, color="#4C72B0")
            ax.set_xlabel(spec.get("x_label", ""))
            ax.set_ylabel(spec.get("y_label", ""))
        elif chart_type == "line":
            ax.plot(labels, values, marker="o", color="#4C72B0")
            ax.set_xlabel(spec.get("x_label", ""))
            ax.set_ylabel(spec.get("y_label", ""))
        elif chart_type == "pie":
            ax.pie(values, labels=labels, autopct="%1.1f%%")

        ax.set_title(spec.get("title", ""))
        fig.tight_layout()

        buffer = io.BytesIO()
        fig.savefig(buffer, format="png")
        buffer.seek(0)
        return buffer.read()

    finally:
        plt.close(fig)  # always release the figure — leaks memory otherwise across many requests