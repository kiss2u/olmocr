"""
Plot for OCR performance vs cost Pareto frontier figure for NeurIPS paper.

Invocation:
    python scripts/pareto_plot.py .
"""

import argparse
import os
from dataclasses import dataclass
from typing import Literal

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
from matplotlib import font_manager

# Parse arguments
ap = argparse.ArgumentParser()
ap.add_argument("output_dir", type=str, help="Path to the output directory")
ap.add_argument(
    "--font-path",
    type=str,
    help="Path to the font file",
    default=None,
)
args = ap.parse_args()

# Add custom font if provided
if args.font_path:
    font_manager.fontManager.addfont(args.font_path)
    plt.rcParams["font.family"] = "Manrope"
    plt.rcParams["font.weight"] = "medium"

# Ensure output directory exists
os.makedirs(args.output_dir, exist_ok=True)
OUTPUT_PATHS = [f"{args.output_dir}/ocr_pareto.pdf", f"{args.output_dir}/ocr_pareto.png"]
# Define column names
MODEL_COLUMN_NAME = "Model"
CATEGORY_COLUMN_NAME = "Category"
COST_COLUMN_NAME = "Cost_Per_Million"
PERF_COLUMN_NAME = "Performance"
COLOR_COLUMN_NAME = "Color"
OFFSET_COLUMN_NAME = "Label_Offset"
MARKER_COLUMN_NAME = "Marker"
# Define colors
DARK_BLUE = "#093235"
DARK_GREEN = "#255457"
LIGHT_GREEN = "#6FE0BA"
LIGHT_PINK = "#F697C4"
DARK_PINK = "#F0529C"
YELLOW = "#fff500"
ORANGE = "#f65834"
DARK_TEAL = "#0a3235"
OFF_WHITE = "#faf2e9"
TEAL = "#105257"
PURPLE = "#b11be8"
GREEN = "#0fcb8c"


# Dataclass for model data
@dataclass(frozen=True)
class ModelData:
    name: str
    cost_per_million: float
    performance: float
    category: str
    label_offset: tuple[float, float]


def cost_per_million_by_token(gpu: Literal["a100", "h100", "l40s"], tokens_sec: float, tokens_per_page: float = 750) -> float:
    """
    Calculate cost per million pages based on GPU type and token throughput.

    Args:
        gpu: GPU type ("a100", "h100", or "l40s")
        tokens_sec: Number of tokens processed per second
        tokens_per_page: Average number of tokens per page (default: 750)

    Returns:
        Cost per million pages in USD
    """
    # GPU hourly costs in USD from https://www.runpod.io/pricing Nov 3 2025
    gpu_costs = {
        "a100": 1.39,
        "h100": 2.69,
        "l40s": 0.79,
    }

    cost_per_hour = gpu_costs[gpu]

    # Calculate pages per hour
    # tokens per hour = tokens_sec * 3600
    # pages per hour = tokens per hour / tokens_per_page
    tokens_per_hour = tokens_sec * 3600
    pages_per_hour = tokens_per_hour / tokens_per_page

    # Calculate cost per million pages
    cost_per_million = (cost_per_hour / pages_per_hour) * 1_000_000

    return cost_per_million


def cost_per_million_by_page(gpu: Literal["a100", "h100", "l40s"], pages_sec: float) -> float:
    """
    Calculate cost per million pages based on GPU type and page throughput.

    Args:
        gpu: GPU type ("a100", "h100", or "l40s")
        pages_sec: Number of pages processed per second

    Returns:
        Cost per million pages in USD
    """
    # GPU hourly costs in USD from https://www.runpod.io/pricing Nov 3 2025
    gpu_costs = {
        "a100": 1.39,
        "h100": 2.69,
        "l40s": 0.79,
    }

    cost_per_hour = gpu_costs[gpu]

    # Calculate pages per hour
    pages_per_hour = pages_sec * 3600

    # Calculate cost per million pages
    cost_per_million = (cost_per_hour / pages_per_hour) * 1_000_000

    return cost_per_million


# All model data in one place for easy editing
MODEL_DATA = [
    # Perf data from historical API pricing
    ModelData(name="GPT-4o", cost_per_million=12480, performance=69.9, category="Commercial VLM", label_offset=(-35, 10)),
    ModelData(name="GPT-4o (Batch)", cost_per_million=6240, performance=69.9, category="Commercial VLM", label_offset=(-50, 10)),
    ModelData(name="Mistral OCR", cost_per_million=1000, performance=72.0, category="Commercial API Tool", label_offset=(-20, 10)),
    ModelData(name="Gemini Flash 2", cost_per_million=499, performance=63.8, category="Commercial VLM", label_offset=(-10, 10)),
    ModelData(name="Gemini Flash 2 (Batch)", cost_per_million=249, performance=63.8, category="Commercial VLM", label_offset=(-50, -20)),
    # Perf data from paper https://arxiv.org/pdf/2509.22186
    ModelData(
        name="MinerU 2.5.4", cost_per_million=cost_per_million_by_page("a100", 2.12), performance=75.2, category="Open Source Tool", label_offset=(10, -5)
    ),
    # Perf data is hard to measure, using previously calculated value, using more generous number from v.1.7.5
    ModelData(name="Marker v1.10.1", cost_per_million=1492, performance=76.1, category="Open Source Tool", label_offset=(-25, 10)),
    # Using cost per million pages from original olmocr paper
    ModelData(name="Qwen 2 VL", cost_per_million=178, performance=31.5, category="Open VLM", label_offset=(-35, 10)),
    ModelData(name="Qwen 2.5 VL", cost_per_million=178, performance=65.5, category="Open VLM", label_offset=(-35, 10)),
    # Perf data from https://arxiv.org/pdf/2509.22186
    ModelData(name="Nanonets-OCR2-3B", cost_per_million=cost_per_million_by_page("a100", 0.55), performance=69.5, category="Open VLM", label_offset=(-85, 10)),
    # Pricing from this tweet: https://x.com/VikParuchuri/status/1980725223616876704
    # You'd get better pricing running locally, but I couldn't get a number
    ModelData(name="Chandra OCR API", cost_per_million=4000, performance=83.1, category="Commercial VLM", label_offset=(-85, 10)),
    # Going off of 200k pages per day per A100
    ModelData(
        name="DeepSeek-OCR",
        cost_per_million=cost_per_million_by_page("a100", pages_sec=200_000 / (24 * 3600)),
        performance=75.7,
        category="Open VLM",
        label_offset=(-20, 10),
    ),
    # Perf data from paper pg 18 https://arxiv.org/pdf/2510.14528
    ModelData(name="PaddleOCR-VL", cost_per_million=cost_per_million_by_page("a100", 1.2241), performance=80.0, category="Open VLM", label_offset=(-35, 10)),
    # Perf data is here: https://beaker.allen.ai/orgs/ai2/workspaces/olmocr/work/01K8V42ERGBHAZ2KKDBKXKZHPJ?taskId=01K8V42ERJ9S82C06CSWQT7RR6&jobId=01K8VH0Y9J47ZXMCCWG97J7P54
    ModelData(
        name="Ours", cost_per_million=cost_per_million_by_page("h100", 10000 / (36 * 60 + 47)), performance=82.3, category="Ours", label_offset=(-20, 10)
    ),
]

# Create dataframe from the aggregated data
df = pd.DataFrame(
    [
        {
            MODEL_COLUMN_NAME: m.name,
            COST_COLUMN_NAME: m.cost_per_million,
            PERF_COLUMN_NAME: m.performance,
            CATEGORY_COLUMN_NAME: m.category,
            OFFSET_COLUMN_NAME: list(m.label_offset),
        }
        for m in MODEL_DATA
    ]
)

# Category colors
category_colors = {"Commercial API Tool": DARK_GREEN, "Commercial VLM": DARK_GREEN, "Open Source Tool": PURPLE, "Ours": DARK_PINK, "Open VLM": PURPLE}

df[COLOR_COLUMN_NAME] = df[CATEGORY_COLUMN_NAME].map(category_colors)

# Define marker types
category_markers = {"Commercial API Tool": "o", "Commercial VLM": "^", "Open Source Tool": "o", "Ours": "*", "Open VLM": "^"}

df[MARKER_COLUMN_NAME] = df[CATEGORY_COLUMN_NAME].map(category_markers)

# Define marker sizes - increased sizes
category_marker_sizes = {"Commercial API Tool": 120, "Commercial VLM": 120, "Open Source Tool": 140, "Ours": 300, "Open VLM": 140}

# Define text colors
category_text_colors = {
    "Commercial API Tool": DARK_GREEN,
    "Commercial VLM": DARK_GREEN,
    "Open Source Tool": PURPLE,  # darker purple
    "Ours": DARK_PINK,  # darker pink
    "Open VLM": PURPLE,  # darker purple
}

# Create the plot
plt.figure(figsize=(10, 6))

# Plot each category
categories = df[CATEGORY_COLUMN_NAME].unique()
for category in categories:
    mask = df[CATEGORY_COLUMN_NAME] == category
    data = df[mask]
    plt.scatter(
        data[COST_COLUMN_NAME],
        data[PERF_COLUMN_NAME],
        label=category,
        c=data[COLOR_COLUMN_NAME],
        marker=category_markers[category],
        alpha=1.0,
        s=category_marker_sizes[category],
    )

# Add labels for each point with increased font size
FONTSIZE = 12  # Increased from 9
for idx, row in df.iterrows():
    plt.annotate(
        row[MODEL_COLUMN_NAME],
        (row[COST_COLUMN_NAME], row[PERF_COLUMN_NAME]),
        xytext=row[OFFSET_COLUMN_NAME],
        textcoords="offset points",
        fontsize=FONTSIZE,
        alpha=1.0,
        weight="medium",
        color=category_text_colors[row[CATEGORY_COLUMN_NAME]],
    )

# Set up axes
plt.ylim(25, 85)  # Set y-axis limits from 25 to 85 to include Qwen2VL
plt.xlim(100, 15000)
plt.xscale("log")  # Use log scale for cost
plt.grid(True, which="both", ls=":", color=TEAL, alpha=0.2)


# Format y-axis to show percentages without scientific notation
def percent_formatter(y, pos):
    return f"{y:.1f}%"


plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(percent_formatter))


# Format x-axis to show dollar amounts
def dollar_formatter(x, pos):
    return f"${x:,.0f}"


# Set specific x-axis ticks with increased font size
plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(dollar_formatter))
plt.gca().set_xticks([100, 200, 300, 500, 1000, 2000, 3000, 5000, 10000])
plt.xticks(fontsize=12)  # Increased tick font size
plt.yticks(fontsize=12)  # Increased tick font size

# Add labels and title with increased font size
plt.xlabel("Cost per Million Pages (USD, log scale)", fontsize=16, weight="medium")
plt.ylabel("Overall Performance (Pass Rate %)", fontsize=16, weight="medium")
# plt.title("OCR Engines: Performance vs. Cost", fontsize=12, weight="medium")

# Remove spines
plt.gca().spines["top"].set_visible(False)
plt.gca().spines["right"].set_visible(False)

# Add the legend with custom ordering and increased font size
handles, labels = plt.gca().get_legend_handles_labels()
desired_order = ["Ours", "Open Source Tool", "Open VLM", "Commercial API Tool", "Commercial VLM"]
label_to_handle = dict(zip(labels, handles))
ordered_handles = [label_to_handle[label] for label in desired_order if label in label_to_handle]
ordered_labels = [label for label in desired_order if label in labels]

plt.legend(
    ordered_handles, ordered_labels, loc="lower right", fontsize=12, frameon=True, framealpha=0.9, edgecolor=TEAL, facecolor="white"  # Increased from 10
)

# Adjust layout
plt.tight_layout()

# Save the figure
for output_path in OUTPUT_PATHS:
    plt.savefig(output_path, dpi=300, bbox_inches="tight")

print(f"Plot saved to {', '.join(OUTPUT_PATHS)}")
