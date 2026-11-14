import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Read the actual Vanda experimental data
df = pd.read_csv("serial_scaling_clean.csv")

# Calculate mean and standard deviation
stats = (
    df.groupby("n")["elapsed_s"]
    .agg(["mean", "std"])
    .reset_index()
    .sort_values("n")
)

print(stats.to_string(index=False))

# Plot runtime scaling
fig, ax = plt.subplots(figsize=(7, 5))

ax.errorbar(
    stats["n"],
    stats["mean"],
    yerr=stats["std"],
    fmt="o-",
    capsize=4,
    linewidth=1.8,
    markersize=6,
    label="Serial (mean ± SD)"
)

ax.set_xscale("log", base=2)
ax.set_yscale("log")

ax.set_xlabel("Grid Size N (N × N)")
ax.set_ylabel("Runtime (seconds)")
ax.set_title("Serial Runtime Scaling on NUS Vanda")

ax.set_xticks(stats["n"])
ax.set_xticklabels(stats["n"].astype(str))

ax.grid(True, which="both", linestyle="--", alpha=0.4)
ax.legend()

fig.tight_layout()

fig.savefig("serial_runtime_scaling.png", dpi=300)
fig.savefig("serial_runtime_scaling.pdf")

print("Plots saved successfully.")
