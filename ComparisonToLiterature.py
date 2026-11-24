
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# 1. LITERATURE DATA
# ==========================================

literature = pd.DataFrame({
    "T": [100, 95.5, 89, 86.7, 85.3, 84.1, 82.7,
          82.3, 80.7, 79.7, 79.3, 78.74, 78.41,
          78.15, 78.3],

    "x": [0, 0.019, 0.0721, 0.099, 0.1238, 0.1661,
          0.2337, 0.2608, 0.3965, 0.5198, 0.5732,
          0.6763, 0.7472, 0.8943, 1],

    "y": [0, 0.17, 0.3891, 0.4375, 0.4704, 0.5089,
          0.5445, 0.558, 0.6122, 0.6599, 0.6841,
          0.7385, 0.7815, 0.8943, 1]
})


# ==========================================
# 2. W16 EXPERIMENTAL DATA
# ==========================================

W16 = pd.DataFrame({
    "T": [81.2, 81.3, 81, 81.4, 82, 82.6, 90.3, 85.5],

    "x": [0.883, 0.889, 0.924, 0.953,
          0.982, 0.992, -0.056, 0.157],

    "y": [0.8292, 0.8903, 0.9514, 0.9764,
          0.9917, 0.9958, 0.8583, 0.9444]
})


# ==========================================
# 3. W5 EXPERIMENTAL DATA
# ==========================================

W5 = pd.DataFrame({
    "T": [80, 79.9, 79.9, 79.9, 80.4, 82.1],

    # Only vapour/distillate composition provided
    "y": [1.0316481, 1.016512126, 1.006975616,
          1.009041102, 0.97588875, 0.979658967]
})


# ==========================================
# 4. W6 EXPERIMENTAL DATA
# ==========================================

W6 = pd.DataFrame({
    "T": [111.9, 124.5, 77.4, 85.4, 85.8, 83],

    "x": [-0.044633503, -0.027343198, -0.027343198,
          0.202794111, 0.196006499, 0.496820406],

    "y": [-0.04349336, 0.020919561, 0.00608715,
          0.249410209, 0.431665528, 0.528804996]
})


# ==========================================
# 5. CHECK AND FILTER INVALID MOLE FRACTIONS
# ==========================================

datasets = {
    "Literature": literature,
    "W16": W16,
    "W5": W5,
    "W6": W6
}

for name, df in datasets.items():

    for col in ["x", "y"]:

        if col in df.columns:

            invalid = df[(df[col] < 0) | (df[col] > 1)]

            if not invalid.empty:
                print(f"\nInvalid {col} values in {name}:")
                print(invalid[["T", col]])

            # Remove physically invalid mole fractions
            df.loc[(df[col] < 0) | (df[col] > 1), col] = np.nan


# ==========================================
# 6. CREATE T-x-y PLOT
# ==========================================

plt.figure(figsize=(10, 7))

# ----- Literature bubble-point curve -----

lit_x = literature.dropna(subset=["x"]).sort_values("x")

plt.plot(
    lit_x["x"],
    lit_x["T"],
    color="black",
    linewidth=2,
    marker="o",
    markersize=5,
    label="Literature Bubble Point"
)


# ----- Literature dew-point curve -----

lit_y = literature.dropna(subset=["y"]).sort_values("y")

plt.plot(
    lit_y["y"],
    lit_y["T"],
    color="black",
    linewidth=2,
    linestyle="--",
    marker="s",
    markersize=5,
    label="Literature Dew Point"
)


# ==========================================
# 7. EXPERIMENTAL DATA
# ==========================================

# W16
plt.scatter(
    W16["x"], W16["T"],
    marker="o",
    color="red",
    s=65,
    label="W16 Bubble Point"
)

plt.scatter(
    W16["y"], W16["T"],
    marker="s",
    color="red",
    s=65,
    facecolors="none",
    label="W16 Dew Point"
)


# W5 (only vapour composition available)
plt.scatter(
    W5["y"], W5["T"],
    marker="^",
    color="blue",
    s=75,
    label="W5 Dew Point"
)


# W6
plt.scatter(
    W6["x"], W6["T"],
    marker="o",
    color="green",
    s=65,
    label="W6 Bubble Point"
)

plt.scatter(
    W6["y"], W6["T"],
    marker="s",
    color="green",
    s=65,
    facecolors="none",
    label="W6 Dew Point"
)


# ==========================================
# 8. GRAPH FORMATTING
# ==========================================

plt.xlabel(
    "Mole Fraction of Ethanol",
    fontsize=13
)

plt.ylabel(
    "Temperature (°C)",
    fontsize=13
)

plt.title(
    "T-x-y Diagram for Ethanol-Water System",
    fontsize=15
)

plt.xlim(0, 1)
plt.ylim(75, 105)

plt.xticks(np.arange(0, 1.01, 0.1))
plt.grid(True, linestyle=":", alpha=0.6)

plt.legend(
    loc="best",
    fontsize=9,
    frameon=True
)

plt.tight_layout()

# Save high-resolution figure
plt.savefig(
    "Ethanol_Water_Txy_Diagram.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
