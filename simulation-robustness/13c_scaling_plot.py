#!/usr/bin/env python3
"""
13c_scaling_plot.py
===================
Plot sec/epoch vs D_actual for K in {10, 20, 30}, using
results_simulation/scaling_summary.csv. Saves PDF + PNG to
results_simulation/scaling_plot.{pdf,png}.

Only fully-completed configs (200/200 epochs, done=True) are used for
the marker; the partial Dx3.0_K10 row is shown as an open marker for
context.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(REPO, "results_simulation")

df = pd.read_csv(os.path.join(RES, "scaling_summary.csv"))
# Keep only fully-completed configs with D_factor <= 2.0
df = df[(df["done"] == True) & (df["D_factor"] <= 2.0)].copy()

colors = {10: "#1f77b4", 20: "#ff7f0e", 30: "#2ca02c"}
markers = {10: "o", 20: "s", 30: "^"}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

for K in sorted(df["K"].unique()):
    sub_done = df[df["K"] == K].sort_values("D_actual")
    ax1.errorbar(sub_done["D_actual"], sub_done["sec_per_epoch_mean"],
                 yerr=sub_done["sec_per_epoch_std"], fmt=markers[K] + "-",
                 color=colors[K], label=f"K={K}", capsize=3, lw=1.5, ms=7)

ax1.set_xlabel(r"$D$ (number of documents)")
ax1.set_ylabel("seconds per epoch")
ax1.set_title("(a) CAVI wall-clock per epoch")
ax1.legend(loc="upper left", fontsize=9)
ax1.grid(alpha=0.3)

# log-log
for K in sorted(df["K"].unique()):
    sub_done = df[df["K"] == K].sort_values("D_actual")
    ax2.loglog(sub_done["D_actual"], sub_done["sec_per_epoch_mean"],
               markers[K] + "-", color=colors[K], label=f"K={K}",
               lw=1.5, ms=7)

# Reference: linear scaling reference line, anchored on smallest run
ref_x = df["D_actual"].min()
ref_y = df[(df["K"] == 10) & (df["D_actual"] == ref_x)]["sec_per_epoch_mean"].iloc[0]
xs = np.array([ref_x, df["D_actual"].max()])
ax2.loglog(xs, ref_y * xs / ref_x, "--", color="gray", lw=1,
           label="linear ref. (slope 1)")

ax2.set_xlabel(r"$D$ (log scale)")
ax2.set_ylabel("seconds per epoch (log scale)")
ax2.set_title("(b) Same data, log--log axes")
ax2.legend(loc="upper left", fontsize=9)
ax2.grid(which="both", alpha=0.3)

plt.tight_layout()

out_pdf = os.path.join(RES, "scaling_plot.pdf")
out_png = os.path.join(RES, "scaling_plot.png")
plt.savefig(out_pdf, bbox_inches="tight")
plt.savefig(out_png, dpi=150, bbox_inches="tight")
print(f"wrote {out_pdf}")
print(f"wrote {out_png}")
