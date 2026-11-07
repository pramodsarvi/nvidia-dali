"""Plot decode_all_frames.py results.

  python plot_decode.py decode.csv ../images
Writes decode_compare.png (methods x resolution) and decode_dali_seq.png (DALI vs sequence length).
"""
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

csv_path = sys.argv[1] if len(sys.argv) > 1 else "decode.csv"
out_dir = sys.argv[2] if len(sys.argv) > 2 else "."

# Fixed colour per method (never by rank): blue, orange, aqua from a validated categorical palette.
COLORS = {"CPU FFmpeg": "#2a78d6", "FFmpeg + NVDEC": "#eb6834", "DALI": "#1baf7a"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 11, "axes.edgecolor": GRID, "axes.labelcolor": MUTED,
                     "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
                     "axes.spines.top": False, "axes.spines.right": False})

df = pd.read_csv(csv_path)
res_order = [r for r in ["1080p", "720p"] if r in set(df.resolution)]
best_seq = 256

def fps(res, method, seq=None):
    d = df[(df.resolution == res) & (df.method == method)]
    if seq is not None:
        d = d[d.seq_len == seq]
    return float(d.frames_per_s.iloc[0])

# ---- Chart 1: grouped bars --------------------------------------------------
methods = [("cpu", "CPU FFmpeg"), ("ffmpeg-gpu", "FFmpeg + NVDEC"), ("dali", "DALI")]
fig, ax = plt.subplots(figsize=(8, 4.6))
w, gap = 0.26, 0.02
for j, (key, label) in enumerate(methods):
    xs, vals = [], []
    for i, res in enumerate(res_order):
        xs.append(i + (j - 1) * (w + gap))
        vals.append(fps(res, key, best_seq if key == "dali" else None))
    bars = ax.bar(xs, vals, w, color=COLORS[label], label=label, zorder=3)
    for x, v in zip(xs, vals):
        ax.text(x, v + 40, f"{v:,.0f}", ha="center", va="bottom", fontsize=10, color=INK)
ax.set_xticks(range(len(res_order)))
ax.set_xticklabels(res_order)
ax.set_ylabel("frames decoded per second")
fig.suptitle("Decoding every frame (L40S, 8 CPU cores, H.264)", x=0.01, ha="left", fontsize=12)
ax.grid(axis="y", color=GRID, zorder=0)
ax.set_axisbelow(True)
ax.legend(frameon=False, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 1.0))
ax.set_ylim(0, ax.get_ylim()[1] * 1.05)
fig.text(0.01, 0.01, f"DALI uses {best_seq}-frame sequences. Median of repeats; higher is better.",
         fontsize=9, color=MUTED)
fig.tight_layout(rect=(0, 0.03, 1, 0.94))
fig.savefig(f"{out_dir}/decode_compare.png", dpi=160)

# ---- Chart 2: DALI vs sequence length, one panel per resolution ------------------
fig, axes = plt.subplots(1, len(res_order), figsize=(9, 4.2), sharey=True)
for ax, res in zip(axes, res_order):
    d = df[(df.resolution == res) & (df.method == "dali")].sort_values("seq_len")
    ax.plot(d.seq_len, d.frames_per_s, marker="o", ms=7, lw=2, color=COLORS["DALI"], label="DALI")
    ax.axhline(fps(res, "cpu"), color=COLORS["CPU FFmpeg"], lw=2, ls="--", label="CPU FFmpeg")
    ax.axhline(fps(res, "ffmpeg-gpu"), color=COLORS["FFmpeg + NVDEC"], lw=2, ls="--", label="FFmpeg + NVDEC")
    for x, y in zip(d.seq_len, d.frames_per_s):
        ax.annotate(f"{y:,.0f}", (x, y), textcoords="offset points", xytext=(0, -16),
                    ha="center", fontsize=9, color=INK)
    ax.set_xscale("log", base=2)
    ax.set_xticks(list(d.seq_len)); ax.set_xticklabels([str(int(v)) for v in d.seq_len])
    ax.set_xlabel("DALI sequence length (frames)")
    ax.set_title(res, loc="left", fontsize=12)
    ax.grid(axis="y", color=GRID); ax.set_axisbelow(True)
axes[0].set_ylabel("frames decoded per second")
axes[0].set_ylim(0, None)
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.0))
fig.tight_layout(rect=(0, 0, 1, 0.92))
fig.savefig(f"{out_dir}/decode_dali_seq.png", dpi=160)
print("wrote", f"{out_dir}/decode_compare.png", f"{out_dir}/decode_dali_seq.png")
