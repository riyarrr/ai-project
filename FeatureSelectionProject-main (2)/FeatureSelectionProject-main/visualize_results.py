
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("final_comparison.csv")

# 1. Accuracy comparison
for dataset in df["Dataset"].unique():
    subset = df[df["Dataset"] == dataset]

    plt.figure(figsize=(9, 5))
    plt.bar(subset["Method"], subset["Accuracy"])
    plt.title(f"Accuracy Comparison - {dataset}")
    plt.xlabel("Feature Selection Method")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(f"{dataset.lower().replace(' ', '_')}_accuracy.png")
    plt.show()

# 2. Feature reduction comparison
df["Features Removed (%)"] = (
    (df["Total Features"] - df["Selected Features"])
    / df["Total Features"] * 100
)

for dataset in df["Dataset"].unique():
    subset = df[df["Dataset"] == dataset]

    plt.figure(figsize=(9, 5))
    plt.bar(subset["Method"], subset["Features Removed (%)"])
    plt.title(f"Feature Reduction - {dataset}")
    plt.xlabel("Feature Selection Method")
    plt.ylabel("Features Removed (%)")
    plt.tight_layout()
    plt.savefig(f"{dataset.lower().replace(' ', '_')}_reduction.png")
    plt.show()

# 3. Runtime comparison
for dataset in df["Dataset"].unique():
    subset = df[df["Dataset"] == dataset]

    plt.figure(figsize=(9, 5))
    plt.bar(subset["Method"], subset["Runtime (s)"])
    plt.title(f"Runtime Comparison - {dataset}")
    plt.xlabel("Feature Selection Method")
    plt.ylabel("Runtime (seconds)")
    plt.tight_layout()
    plt.savefig(f"{dataset.lower().replace(' ', '_')}_runtime.png")
    plt.show()

print("All comparison graphs saved successfully.")
