import datetime as dt
import os
import textwrap

import matplotlib.dates as mdates
import matplotlib.pyplot as plt

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

# Model categories
model_categories = {
    "GOT OCR": "Open Source Tool",
    "Marker v1.7.5": "Open Source Tool",
    "Marker v1.10.1": "Open Source Tool",
    "MinerU v1.3.10": "Open Source Tool",
    "MinerU v2.5.4": "Open Source Tool",
    "Nanonets OCR S": "Commercial API Tool",
    "MonkeyOCR Pro 3B": "Open VLM",
    "MinerU2.5": "Open VLM",
    "dots.ocr": "Open VLM",
    "PaddleOCR-VL": "Open VLM",
    "Mistral OCR API": "Commercial API Tool",
    "GPT-4o": "Commercial VLM",
    "Gemini Flash 2": "Commercial VLM",
    "Qwen 2.5 VL": "Open VLM",
    "olmOCR v0.1.58": "Ours",
    "olmOCR v0.1.60": "Ours",
    "olmOCR v0.1.68": "Ours",
    "olmOCR v0.2.0": "Ours",
    "olmOCR v0.3.0": "Ours",
    "olmOCR v0.4grporl": "Ours",
}

# Category colors
category_colors = {"Commercial API Tool": DARK_GREEN, "Commercial VLM": DARK_GREEN, "Open Source Tool": PURPLE, "Ours": DARK_PINK, "Open VLM": PURPLE}

# Define marker types
category_markers = {"Commercial API Tool": "o", "Commercial VLM": "^", "Open Source Tool": "o", "Ours": "*", "Open VLM": "^"}

# Define marker sizes
category_marker_sizes = {"Commercial API Tool": 100, "Commercial VLM": 100, "Open Source Tool": 120, "Ours": 300, "Open VLM": 120}

# Define text colors
category_text_colors = {"Commercial API Tool": DARK_GREEN, "Commercial VLM": DARK_GREEN, "Open Source Tool": PURPLE, "Ours": DARK_PINK, "Open VLM": PURPLE}

# Data
ocr_overall_scores = {
    "GOT OCR": {
        "name": "GOT OCR",
        "descriptor": "",
        "Overall": "48.3 ± 1.1",
        "date": "Sep 4, 2024",
        "paper": "https://arxiv.org/abs/2409.01704",
    },
    # "Marker v1.4.0": {"Overall": "70.1 ± 1.1", "date": "Feb 11, 2024", "paper": "https://github.com/datalab-to/marker/tree/v1.4.0"},
    "Marker v1.7.5": {
        "name": "Marker v1.7.5",
        "descriptor": "force ocr",
        "Overall": "70.1 ± 1.1",
        "date": "Jun 11, 2025",
        "paper": "https://github.com/datalab-to/marker/tree/v1.7.5",
    },
    "Marker v1.10.1": {
        "name": "Marker v1.10.1",
        "descriptor": "",
        "Overall": "76.1 ± 1.1",
        "date": "Sep 30, 2025",
        "paper": "https://github.com/datalab-to/marker/releases/tag/v1.10.1",
    },
    "MinerU v1.3.10": {
        "name": "MinerU v1.3.10",
        "descriptor": "",
        "Overall": "61.5 ± 1.1",
        "date": "Apr 29, 2025",
        "paper": "https://github.com/opendatalab/MinerU/tree/magic_pdf-1.3.10-released",
    },
    "MinerU v2.5.4": {
        "name": "MinerU v2.5.4",
        "descriptor": "",
        "Overall": "62.9 ± 1.1",
        "date": "Sep 25, 2025",
        "paper": "https://github.com/opendatalab/MinerU/releases/tag/mineru-2.5.4-released",
    },
    "Nanonets OCR S": {
        "name": "Nanonets OCR S",
        "descriptor": "",
        "Overall": "64.5 ± 1.1",
        "date": "Jun 12, 2025",
        "paper": "https://nanonets.com/research/nanonets-ocr-s/",
    },
    "MonkeyOCR Pro 3B": {
        "name": "MonkeyOCR Pro 3B",
        "descriptor": "",
        "Overall": "75.8 ± 1.0",
        "date": "Jun 5, 2025",
        "paper": "https://arxiv.org/abs/2506.05218",
    },
    "MinerU2.5": {
        "name": "MinerU2.5",
        "descriptor": "",
        "Overall": "77.5 ± 1.0",
        "date": "Sep 26, 2025",
        "paper": "https://arxiv.org/abs/2509.22186",
    },
    "dots.ocr": {
        "name": "dots.ocr",
        "descriptor": "",
        "Overall": "79.1 ± 1.0",
        "date": "Jul 30, 2025",
        "paper": "https://github.com/rednote-hilab/dots.ocr",
    },
    "PaddleOCR-VL": {
        "name": "PaddleOCR-VL",
        "descriptor": "",
        "Overall": "80.0 ± 1.0",
        "date": "Oct 15, 2025",
        "paper": "https://arxiv.org/abs/2510.14528",
    },
    "Mistral OCR API": {
        "name": "Mistral OCR API",
        "descriptor": "",
        "Overall": "72.0 ± 1.1",
        "date": "Mar 6, 2025",
        "paper": "https://mistral.ai/fr/news/mistral-ocr",
    },
    "GPT-4o": {
        "name": "GPT-4o",
        "descriptor": "No Anchor",
        "Overall": "68.9 ± 1.1",
        "date": "May 13, 2024",
        "paper": "https://openai.com/index/hello-gpt-4o/",
    },
    # "GPT-4o (Anchored)": {
    #     "name": "GPT-4o",
    #     "descriptor": "Anchored",
    #     "Overall": "69.9 ± 1.1",
    #     "date": "May 13, 2024",
    #     "paper": "https://openai.com/index/hello-gpt-4o/",
    # },
    "Gemini Flash 2": {
        "name": "Gemini Flash 2",
        "descriptor": "No Anchor",
        "Overall": "57.8 ± 1.1",
        "date": "Dec 11, 2024",
        "paper": "https://blog.google/technology/google-deepmind/google-gemini-ai-update-december-2024/",
    },
    # "Gemini Flash 2 (Anchored)": {
    #     "name": "Gemini Flash 2",
    #     "descriptor": "Anchored",
    #     "Overall": "63.8 ± 1.2",
    #     "date": "Dec 11, 2024",
    #     "paper": "https://blog.google/technology/google-deepmind/google-gemini-ai-update-december-2024/",
    # },
    "Qwen 2 VL": {
        "name": "Qwen 2 VL",
        "descriptor": "No Anchor",
        "Overall": "31.5 ± 0.9",
        "date": "Aug 29, 2024",
        "paper": "https://arxiv.org/abs/2409.12191v1",
    },
    "Qwen 2.5 VL": {
        "name": "Qwen 2.5 VL",
        "descriptor": "No Anchor",
        "Overall": "65.5 ± 1.2",
        "date": "Feb 19, 2025",
        "paper": "https://arxiv.org/abs/2502.13923",
    },
    "olmOCR v0.1.58": {
        "name": "olmOCR v0.1.58",
        "descriptor": "Initial release",
        "Overall": "68.2 ± 1.1",
        "date": "Feb 14, 2025",
        "paper": "https://github.com/allenai/olmocr/releases/tag/v0.1.58",
    },
    # "olmOCR v0.1.60": {
    #     "name": "olmOCR v0.1.60",
    #     "descriptor": "Temperature from 0.8 fixed to dynamic",
    #     "Overall": "71.4 ± 1.1",
    #     "date": "Mar 17, 2025",
    #     "paper": "https://github.com/allenai/olmocr/releases/tag/v0.1.60",
    # },
    "olmOCR v0.1.68": {
        "name": "olmOCR v0.1.68",
        "descriptor": "Base VLM, prompts, dynamic temp",
        "Overall": "75.8 ± 1.1",
        "date": "May 19, 2025",
        "paper": "https://github.com/allenai/olmocr/releases/tag/v0.1.68",
    },
    "olmOCR v0.2.0": {
        "name": "olmOCR v0.2.0",
        "descriptor": "Trainer, YAML, image size",
        "Overall": "78.5 ± 1.1",
        "date": "Jul 23, 2025",
        "paper": "https://github.com/allenai/olmocr/releases/tag/v0.2.0",
    },
    # "olmOCR v0.3.0": {
    #     "name": "olmOCR v0.3.0",
    #     "descriptor": "Fixing bug with blank page hallucinations",
    #     "Overall": "78.5 ± 1.1",
    #     "date": "Aug 13, 2025",
    #     "paper": "https://github.com/allenai/olmocr/releases/tag/v0.3.0",
    # },
    "olmOCR v0.4grporl": {
        "name": "olmOCR v0.4grporl",
        "descriptor": "RLVR",
        "Overall": "82.6 ± 1.1",
        "date": "Oct 21, 2025",
        "paper": "https://github.com/allenai/olmocr/releases/tag/v0.4.0",
    },
}


# Convert string dates and overall means
olm_data = [(k, v) for k, v in ocr_overall_scores.items() if k.startswith("olmOCR")]
marker_data = [(k, v) for k, v in ocr_overall_scores.items() if k.startswith("Marker")]
mineru_data = [(k, v) for k, v in ocr_overall_scores.items() if k.startswith("MinerU")]
other_data = [(k, v) for k, v in ocr_overall_scores.items() if not k.startswith("olmOCR") and not k.startswith("Marker") and not k.startswith("MinerU")]


def parse_mean(value):
    return float(value.split("±")[0].strip())


def wrap_text(text, max_chars=20):
    """Wrap text to fit within max_chars per line."""
    if len(text) <= max_chars:
        return text
    return "\n".join(textwrap.wrap(text, width=max_chars))


# Label position offsets (x_offset in days, y_offset in score units)
# Adjust these to manually tune label positions relative to their data points
label_offsets = {
    "olmOCR v0.1.58": (-20, -0.5),
    "olmOCR v0.1.60": (0, 1.5),
    "olmOCR v0.1.68": (-10, 1.5),
    "olmOCR v0.2.0": (0, 1.5),
    "olmOCR v0.4grporl": (0, 1.5),
    "Marker v1.7.5": (0, 1.5),
    "Marker v1.10.1": (0, -5.5),
    "MinerU v1.3.10": (0, -5.5),
    "MinerU v2.5.4": (0, -5.5),
    "Nanonets OCR S": (0, 1.5),
    "MonkeyOCR Pro 3B": (0, 1.5),
    "MinerU2.5": (0, 1.5),
    "dots.ocr": (0, 1.5),
    "PaddleOCR-VL": (0, 1.5),
    "GOT OCR": (0, 1.5),
    "Mistral OCR API": (0, 1.5),
    "GPT-4o": (0, 1.5),
    "Gemini Flash 2": (0, -5.5),
    "Qwen 2.5 VL": (0, -6),
}

# Floating label offsets for line curves (x_offset in days, y_offset in score units)
floating_label_offsets = {
    "olmOCR": (-40, 8.5),
    "Marker": (-40, -7.5),
    "MinerU": (-70, -4.5),
}

# Sort entries by date (if filled)
olm_data_sorted = sorted(
    [(name, dt.datetime.strptime(v["date"], "%b %d, %Y"), parse_mean(v["Overall"]), v["name"], v["descriptor"]) for name, v in olm_data],
    key=lambda x: x[1],
)
marker_data_sorted = sorted(
    [(name, dt.datetime.strptime(v["date"], "%b %d, %Y"), parse_mean(v["Overall"]), v["name"]) for name, v in marker_data],
    key=lambda x: x[1],
)
mineru_data_sorted = sorted(
    [(name, dt.datetime.strptime(v["date"], "%b %d, %Y"), parse_mean(v["Overall"]), v["name"]) for name, v in mineru_data],
    key=lambda x: x[1],
)

# Plot
plt.figure(figsize=(8, 5))

# olmOCR line
dates = [d for _, d, _, _, _ in olm_data_sorted]
scores = [s for _, _, s, _, _ in olm_data_sorted]
category = model_categories.get("olmOCR v0.1.58", "Ours")
color = category_colors[category]
marker_star = category_markers[category]  # star marker
marker_triangle = "^"  # VLM marker shape
marker_size = category_marker_sizes[category]

plt.plot(dates, scores, color=color, linewidth=2)
for idx, (name, date, score, display_name, descriptor) in enumerate(olm_data_sorted):
    # Only the last point gets a star, others get triangles
    marker = marker_star if idx == len(olm_data_sorted) - 1 else marker_triangle
    size = marker_size if idx == len(olm_data_sorted) - 1 else 100
    plt.scatter(date, score, color=color, edgecolor="none", s=size, marker=marker, zorder=3)

# Add descriptor labels above olmOCR circles (black text)
for name, date, score, display_name, descriptor in olm_data_sorted:
    x_off, y_off = label_offsets.get(name, (0, 1.5))
    label_date = date + dt.timedelta(days=x_off)
    wrapped_descriptor = wrap_text(descriptor, max_chars=20)
    plt.text(label_date, score + y_off, f"{wrapped_descriptor}\n{score:.1f}", ha="center", va="bottom", fontsize=7, fontweight="bold", color="black")

# Add floating label for olmOCR line (above the line, no border)
if olm_data_sorted:
    mid_idx = len(olm_data_sorted) // 2
    mid_date = olm_data_sorted[mid_idx][1]
    mid_score = olm_data_sorted[mid_idx][2]
    x_offset_days, y_offset = floating_label_offsets.get("olmOCR", (0, -3))
    label_date = mid_date + dt.timedelta(days=x_offset_days)
    plt.text(
        label_date,
        mid_score + y_offset,
        "olmOCR",
        fontsize=10,
        fontweight="bold",
        color=color,
        ha="center",
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="none", alpha=0.8),
    )

# Marker line
if marker_data_sorted:
    dates = [d for _, d, _, _ in marker_data_sorted]
    scores = [s for _, _, s, _ in marker_data_sorted]

    # Get category info for first Marker model
    first_name = marker_data_sorted[0][0]
    category = model_categories.get(first_name, "Open Source Tool")
    color = category_colors[category]
    marker = category_markers[category]
    marker_size = category_marker_sizes[category]
    text_color = category_text_colors[category]

    if len(marker_data_sorted) > 1:
        # Multiple points: draw line, use version labels, and add floating label
        plt.plot(dates, scores, color=color, linewidth=2)
        for name, date, score, display_name in marker_data_sorted:
            plt.scatter(date, score, color=color, edgecolor="none", s=marker_size, marker=marker, zorder=3)

        # Add version labels above Marker circles
        for name, date, score, display_name in marker_data_sorted:
            version = display_name.replace("Marker ", "")
            x_off, y_off = label_offsets.get(name, (0, 1.5))
            label_date = date + dt.timedelta(days=x_off)
            plt.text(label_date, score + y_off, f"{version}\n{score:.1f}", ha="center", va="bottom", fontsize=7, fontweight="bold", color="black")

        # Add floating label for Marker line
        mid_idx = len(marker_data_sorted) // 2
        mid_date = marker_data_sorted[mid_idx][1]
        mid_score = marker_data_sorted[mid_idx][2]
        x_offset_days, y_offset = floating_label_offsets.get("Marker", (0, -3))
        label_date = mid_date + dt.timedelta(days=x_offset_days)
        plt.text(
            label_date,
            mid_score + y_offset,
            "Marker",
            fontsize=10,
            fontweight="bold",
            color=color,
            ha="center",
            va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="none", alpha=0.8),
        )
    else:
        # Single point: suppress version, just show "Marker"
        for name, date, score, display_name in marker_data_sorted:
            plt.scatter(date, score, color=color, edgecolor="none", s=marker_size, marker=marker, zorder=2)
            x_off, y_off = label_offsets.get(name, (0, 1.5))
            label_date = date + dt.timedelta(days=x_off)
            plt.text(label_date, score + y_off, f"Marker\n{score:.1f}", ha="center", va="bottom", fontsize=7, fontweight="bold", color="black")

# MinerU line
if mineru_data_sorted:
    dates = [d for _, d, _, _ in mineru_data_sorted]
    scores = [s for _, _, s, _ in mineru_data_sorted]

    # Get category info for first MinerU model
    first_name = mineru_data_sorted[0][0]
    category = model_categories.get(first_name, "Open Source Tool")
    color = category_colors[category]
    marker = category_markers[category]
    marker_size = category_marker_sizes[category]
    text_color = category_text_colors[category]

    if len(mineru_data_sorted) > 1:
        # Multiple points: draw line, use version labels, and add floating label
        plt.plot(dates, scores, color=color, linewidth=2)
        for name, date, score, display_name in mineru_data_sorted:
            plt.scatter(date, score, color=color, edgecolor="none", s=marker_size, marker=marker, zorder=3)

        # Add version labels above MinerU circles
        for name, date, score, display_name in mineru_data_sorted:
            version = display_name.replace("MinerU ", "")
            x_off, y_off = label_offsets.get(name, (0, 1.5))
            label_date = date + dt.timedelta(days=x_off)
            plt.text(label_date, score + y_off, f"{version}\n{score:.1f}", ha="center", va="bottom", fontsize=7, fontweight="bold", color="black")

        # Add floating label for MinerU line
        mid_idx = len(mineru_data_sorted) // 2
        mid_date = mineru_data_sorted[mid_idx][1]
        mid_score = mineru_data_sorted[mid_idx][2]
        x_offset_days, y_offset = floating_label_offsets.get("MinerU", (0, -3))
        label_date = mid_date + dt.timedelta(days=x_offset_days)
        plt.text(
            label_date,
            mid_score + y_offset,
            "MinerU",
            fontsize=10,
            fontweight="bold",
            color=color,
            ha="center",
            va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="none", alpha=0.8),
        )
    else:
        # Single point: suppress version, just show "MinerU"
        for name, date, score, display_name in mineru_data_sorted:
            plt.scatter(date, score, color=color, edgecolor="none", s=marker_size, marker=marker, zorder=2)
            x_off, y_off = label_offsets.get(name, (0, 1.5))
            label_date = date + dt.timedelta(days=x_off)
            plt.text(label_date, score + y_off, f"MinerU\n{score:.1f}", ha="center", va="bottom", fontsize=7, fontweight="bold", color="black")

# Other models
for name, v in other_data:
    if v["date"]:
        d = dt.datetime.strptime(v["date"], "%b %d, %Y")
        s = parse_mean(v["Overall"])

        # Get category info
        category = model_categories.get(name, "Open VLM")
        color = category_colors[category]
        marker = category_markers[category]
        marker_size = category_marker_sizes[category]
        text_color = category_text_colors[category]

        plt.scatter(d, s, color=color, edgecolor="none", s=marker_size, marker=marker, zorder=2)

        # Add label above circle
        x_off, y_off = label_offsets.get(name, (0, 1.5))
        label_date = d + dt.timedelta(days=x_off)
        wrapped_name = wrap_text(name, max_chars=20)
        plt.text(label_date, s + y_off, f"{wrapped_name}\n{s:.1f}", ha="center", va="bottom", fontsize=7, fontweight="bold", color="black")

# Labels and style
plt.xlabel("Date")
plt.ylabel("Overall Performance")

# Format x-axis with dates
ax = plt.gca()
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.gcf().autofmt_xdate()  # Rotate date labels for better readability

# Increase y-axis limits to prevent label bleeding
current_ylim = ax.get_ylim()
ax.set_ylim(current_ylim[0] - 5, current_ylim[1] + 10)
plt.grid(alpha=0.3, linestyle="--")

plt.tight_layout()

# Save the plot first before showing
save_path = os.path.join(os.path.dirname(__file__), "olmocr2_timeline.png")
plt.savefig(save_path, dpi=300, bbox_inches="tight")
print(f"Saved plot to {save_path}")

plt.show()
