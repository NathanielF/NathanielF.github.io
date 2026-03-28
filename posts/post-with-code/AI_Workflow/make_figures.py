"""
Generate static figures for the AI Workflow meta-post.
Reads git history of attention.qmd and produces three PNGs.

Run from the repo root:
    python posts/post-with-code/AI_Workflow/make_figures.py
"""

import subprocess
import re
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ---- Palette (matches attention.qmd) ----
PAL = {
    "grey1":    "#a8a0a0",
    "grey2":    "#7a7170",
    "taupe":    "#b5a99a",
    "brown":    "#8c7a6b",
    "focal1":   "#c25a3c",
    "focal2":   "#4a6d8c",
    "focal3":   "#6b8f6b",
    "ref_line": "#3d3535",
    "ref_dash": "#6b6060",
}

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor":   "white",
    "axes.edgecolor":   PAL["ref_dash"],
    "axes.labelcolor":  PAL["ref_line"],
    "text.color":       PAL["ref_line"],
    "xtick.color":      PAL["ref_dash"],
    "ytick.color":      PAL["ref_dash"],
    "axes.grid":        False,
    "font.size":        10,
})

import os
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ---- Parse git log ----
TARGET = "posts/post-with-code/Consideration_Sets_Attention/attention.qmd"

raw = subprocess.check_output(
    ["git", "log", "--format=%H|%ai|%s", "--numstat", "--follow", "--", TARGET],
    text=True,
)

commits = []
lines = raw.strip().split("\n")
i = 0
while i < len(lines):
    line = lines[i]
    if "|" in line and len(line.split("|")) >= 3:
        parts = line.split("|", 2)
        h = parts[0].strip()
        date_str = parts[1].strip()
        msg = parts[2].strip()
        dt = datetime.strptime(date_str[:19], "%Y-%m-%d %H:%M:%S")
        adds, dels = 0, 0
        i += 1
        # skip blank lines
        while i < len(lines) and lines[i].strip() == "":
            i += 1
        # read numstat lines
        while i < len(lines) and lines[i].strip() != "" and "|" not in lines[i]:
            parts2 = lines[i].strip().split()
            if len(parts2) >= 2:
                try:
                    adds += int(parts2[0])
                    dels += int(parts2[1])
                except ValueError:
                    pass
            i += 1
        commits.append({
            "hash": h[:7],
            "date": dt,
            "message": msg,
            "additions": adds,
            "deletions": dels,
        })
    else:
        i += 1

commits.reverse()  # chronological order

# Compute cumulative line count
total = 0
for c in commits:
    total += c["additions"] - c["deletions"]
    c["total_lines"] = total

# ---- Phase assignments ----
PHASES = {
    "Phase 1: Initial Draft":       ("2026-03-13", "2026-03-15"),
    "Phase 2: Consideration Model": ("2026-03-15", "2026-03-19"),
    "Phase 3: McCloskey & Streetlight": ("2026-03-23", "2026-03-23 14:00"),
    "Phase 4: Swiss Metro":         ("2026-03-23 14:00", "2026-03-23 21:00"),
    "Phase 5: Accountability Sinks": ("2026-03-23 21:00", "2026-03-24 09:00"),
    "Phase 6: Exclusion Restriction Lands": ("2026-03-24 09:00", "2026-03-27"),
}

phase_colors = [
    PAL["grey1"], PAL["taupe"], PAL["focal2"],
    PAL["focal3"], PAL["focal1"], PAL["brown"],
]

def get_phase(dt):
    for idx, (name, (start, end)) in enumerate(PHASES.items()):
        s = datetime.strptime(start, "%Y-%m-%d %H:%M" if " " in start else "%Y-%m-%d")
        e = datetime.strptime(end, "%Y-%m-%d %H:%M" if " " in end else "%Y-%m-%d")
        if s <= dt < e:
            return idx, name
    return len(PHASES) - 1, list(PHASES.keys())[-1]


for c in commits:
    c["phase_idx"], c["phase_name"] = get_phase(c["date"])

# ================================================================
# Figure 1: Line count evolution
# ================================================================
fig, ax = plt.subplots(figsize=(16, 7))

dates = [c["date"] for c in commits]
totals = [c["total_lines"] for c in commits]
colors = [phase_colors[c["phase_idx"]] for c in commits]

ax.step(dates, totals, where="post", color=PAL["ref_dash"], linewidth=1.5, alpha=0.6)
for d, t, col, c in zip(dates, totals, colors, commits):
    ax.scatter(d, t, color=col, s=60, zorder=5, edgecolors="white", linewidth=0.8)

# Annotate key commits
annotations = {
    "update transformer draft": ("Initial draft", (-30, 15)),
    "Update with swiss train model": ("Swiss Metro\nadded", (-60, 20)),
    "tightening": ("Tightening\n-140 lines", (-50, -30)),
    "Update accountability sink edit.": ("Accountability\nsinks", (10, 20)),
    "removing summary tables": ("Tables cut", (10, -25)),
    "Update attention article, more pointed,.": ("Final\nsharpening", (-60, 20)),
}

for c in commits:
    for key, (label, offset) in annotations.items():
        if key in c["message"]:
            ax.annotate(
                label, (c["date"], c["total_lines"]),
                xytext=offset, textcoords="offset points",
                fontsize=7.5, color=PAL["ref_line"],
                arrowprops=dict(arrowstyle="-", color=PAL["ref_dash"], lw=0.7),
                ha="center",
            )

# Phase legend
patches = [mpatches.Patch(color=phase_colors[i], label=name, alpha=0.85)
           for i, name in enumerate(PHASES.keys())]
ax.legend(handles=patches, fontsize=7.5, loc="upper left", framealpha=0.9)

ax.set_ylabel("Total lines in file")
ax.set_xlabel("")
ax.set_title("Evolution of attention.qmd: 18 commits over 13 days")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/line_evolution.png", dpi=300, bbox_inches="tight")
plt.close()
print("Saved line_evolution.png")

# ================================================================
# Figure 2: Additions vs deletions per commit
# ================================================================
fig, ax = plt.subplots(figsize=(14, 10))

y_pos = np.arange(len(commits))
labels = [f"{c['hash']} {c['message'][:40]}" for c in commits]
adds = [c["additions"] for c in commits]
dels = [-c["deletions"] for c in commits]
bar_colors = [phase_colors[c["phase_idx"]] for c in commits]

ax.barh(y_pos, adds, color=PAL["focal3"], alpha=0.75, label="Additions", height=0.7)
ax.barh(y_pos, dels, color=PAL["focal1"], alpha=0.75, label="Deletions", height=0.7)
ax.set_yticks(y_pos)
ax.set_yticklabels(labels, fontsize=7)
ax.axvline(0, color=PAL["ref_line"], linewidth=0.8)
ax.set_xlabel("Lines changed")
ax.set_title("Per-commit additions and deletions")
ax.legend(fontsize=9)
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/additions_deletions.png", dpi=300, bbox_inches="tight")
plt.close()
print("Saved additions_deletions.png")

# ================================================================
# Figure 3: Argument arc
# ================================================================
fig, ax = plt.subplots(figsize=(18, 6))

nodes = [
    ("Technical\nanalogy", "Core Q/K/V\n\u2194 consideration"),
    ("Methodological\nclaim", "Named parameters\nvs anonymous heads"),
    ("Rhetorical\nclarity", "McCloskey pass\nstreetlight thread"),
    ("Empirical\ngrounding", "Swiss Metro\naliasing effect"),
    ("Political\nargument", "Accountability sinks\nvendor APIs"),
    ("Exclusion restriction\nlands", "Clean vs impure\ninstruments"),
]

x_positions = np.linspace(0.08, 0.92, len(nodes))
y_node = 0.65
y_label = 0.25

for i, ((title, subtitle), x) in enumerate(zip(nodes, x_positions)):
    color = phase_colors[i]
    circle = plt.Circle((x, y_node), 0.045, transform=ax.transAxes,
                        color=color, alpha=0.85, zorder=5)
    ax.add_patch(circle)
    ax.text(x, y_node, str(i + 1), transform=ax.transAxes,
            ha="center", va="center", fontsize=11, fontweight="bold", color="white")
    ax.text(x, y_node + 0.12, title, transform=ax.transAxes,
            ha="center", va="bottom", fontsize=8.5, fontweight="bold",
            color=PAL["ref_line"])
    ax.text(x, y_label, subtitle, transform=ax.transAxes,
            ha="center", va="top", fontsize=7.5, color=PAL["grey2"],
            style="italic")

# Connecting arrows
for i in range(len(nodes) - 1):
    ax.annotate(
        "", xy=(x_positions[i + 1] - 0.05, y_node),
        xytext=(x_positions[i] + 0.05, y_node),
        xycoords="axes fraction", textcoords="axes fraction",
        arrowprops=dict(arrowstyle="->", color=PAL["ref_dash"], lw=1.5),
    )

# Dashed "latent → explicit" arrow from node 4 to node 6
ax.annotate(
    "latent \u2192 explicit",
    xy=(x_positions[5], y_node - 0.08),
    xytext=(x_positions[3], y_node - 0.08),
    xycoords="axes fraction", textcoords="axes fraction",
    fontsize=7.5, color=PAL["brown"], ha="center", va="top",
    arrowprops=dict(arrowstyle="->", color=PAL["brown"], lw=1.2,
                    linestyle="dashed"),
)

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")
ax.set_title("How the argument evolved: from technical analogy to accountability infrastructure",
             fontsize=11, pad=10)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/argument_arc.png", dpi=300, bbox_inches="tight")
plt.close()
print("Saved argument_arc.png")

print("\nDone. All figures saved to", OUT_DIR)
