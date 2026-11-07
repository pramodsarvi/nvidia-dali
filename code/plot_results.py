# plot_results.py
import pandas as pd, matplotlib.pyplot as plt

df = pd.read_csv("results.csv")
for (modality, mode), g in df.groupby(["modality", "mode"]):
    pivot = g.pivot_table(index="workers", columns="loader", values="samples_per_s")
    ax = pivot.plot.bar(figsize=(7, 4), rot=0)
    ax.set_ylabel("clips / s" if modality == "video" else "images / s")
    ax.set_title(f"{modality} – {mode}  ({g['gpu'].iloc[0]})")
    plt.tight_layout()
    plt.savefig(f"bench_{modality}_{mode}.png", dpi=150)
    print("wrote", f"bench_{modality}_{mode}.png")
