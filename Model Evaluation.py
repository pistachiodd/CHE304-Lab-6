
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# 1. IMPORT DATA
# ==========================================

df = pd.read_excel("Txy_Data.xlsx")

models = ["NRTL", "WILSON", "UNIQUAC", "IDEAL"]

# Extract literature data
lit = df[["Temp Literature",
          "x Literature",
          "y Literature"]].dropna()

lit = lit.sort_values("x Literature")

T_lit = lit["Temp Literature"].to_numpy()
x_lit = lit["x Literature"].to_numpy()
y_lit = lit["y Literature"].to_numpy()

# ==========================================
# 2. PLOT T-x-y DIAGRAM
# ==========================================

colors = {
    "NRTL": "red",
    "WILSON": "blue",
    "UNIQUAC": "green",
    "IDEAL": "purple"
}

fig, ax = plt.subplots(figsize=(11, 7))

for model in models:

    data = df[
        [f"Temp {model}", f"x {model}", f"y {model}"]
    ].dropna()

    # Sort by liquid composition
    data = data.sort_values(f"x {model}")

    T = data[f"Temp {model}"].to_numpy()
    x = data[f"x {model}"].to_numpy()
    y = data[f"y {model}"].to_numpy()

    # Bubble point
    ax.plot(
        x, T,
        color=colors[model],
        linewidth=2,
        label=f"{model} Bubble Point"
    )

    # Dew point
    ax.plot(
        y, T,
        color=colors[model],
        linewidth=2,
        linestyle="--",
        label=f"{model} Dew Point"
    )

# Literature bubble point
ax.scatter(
    x_lit, T_lit,
    color="black",
    marker="o",
    s=65,
    zorder=5,
    label="Literature Bubble Point"
)

# Literature dew point
ax.scatter(
    y_lit, T_lit,
    color="black",
    marker="s",
    s=65,
    zorder=5,
    label="Literature Dew Point"
)

ax.set_xlabel("Mole Fraction of Ethanol", fontsize=13)
ax.set_ylabel("Temperature (°C)", fontsize=13)

ax.set_title(
    "T-x-y Diagram: Ethanol-Water System",
    fontsize=15
)

ax.set_xlim(0, 1)
ax.set_ylim(75, 102)

ax.grid(True, linestyle=":", alpha=0.5)

ax.legend(
    fontsize=9,
    loc="best",
    ncol=2
)

plt.tight_layout()
plt.savefig("Txy_Model_Comparison.png", dpi=300)
plt.show()


# ==========================================
# 3. CALCULATE MODEL ERRORS
# ==========================================

results = []

for model in models:

    data = df[
        [f"Temp {model}", f"x {model}", f"y {model}"]
    ].dropna()

    data = data.sort_values(f"x {model}")

    x_model = data[f"x {model}"].to_numpy()
    T_model = data[f"Temp {model}"].to_numpy()
    y_model = data[f"y {model}"].to_numpy()

    # Interpolate model predictions at literature x values
    T_pred = np.interp(x_lit, x_model, T_model)
    y_pred = np.interp(x_lit, x_model, y_model)

    # Temperature errors
    T_error = T_pred - T_lit

    # Vapour composition errors
    y_error = y_pred - y_lit

    # RMSE
    RMSE_T = np.sqrt(np.mean(T_error**2))
    RMSE_y = np.sqrt(np.mean(y_error**2))

    # MAE
    MAE_T = np.mean(np.abs(T_error))
    MAE_y = np.mean(np.abs(y_error))

    # R-squared
    R2_T = 1 - (
        np.sum(T_error**2) /
        np.sum((T_lit - np.mean(T_lit))**2)
    )

    R2_y = 1 - (
        np.sum(y_error**2) /
        np.sum((y_lit - np.mean(y_lit))**2)
    )

    # Normalized RMSE
    NRMSE_T = RMSE_T / np.ptp(T_lit)
    NRMSE_y = RMSE_y / np.ptp(y_lit)

    # Combined normalized error
    Combined_NRMSE = np.sqrt(
        (NRMSE_T**2 + NRMSE_y**2) / 2
    )

    results.append({
        "Model": model,
        "RMSE Temperature": RMSE_T,
        "RMSE y": RMSE_y,
        "MAE Temperature": MAE_T,
        "MAE y": MAE_y,
        "R2 Temperature": R2_T,
        "R2 y": R2_y,
        "Combined NRMSE": Combined_NRMSE
    })


# ==========================================
# 4. RANK MODELS
# ==========================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="Combined NRMSE"
).reset_index(drop=True)

results_df.index += 1

print("\nMODEL COMPARISON RESULTS")
print("========================")

print(results_df.round(5).to_string())

best_model = results_df.iloc[0]["Model"]

print("\nBEST FITTING MODEL:")
print(best_model)

# Export comparison results
results_df.to_excel(
    "Model_Comparison_Results.xlsx",
    index=True
)


# ==========================================
# 5. PLOT MODEL ACCURACY
# ==========================================

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Temperature RMSE
axes[0].bar(
    results_df["Model"],
    results_df["RMSE Temperature"],
    color=[colors[m] for m in results_df["Model"]]
)

axes[0].set_title("Temperature RMSE")
axes[0].set_ylabel("RMSE (°C)")
axes[0].grid(axis="y", alpha=0.3)

# Vapour composition RMSE
axes[1].bar(
    results_df["Model"],
    results_df["RMSE y"],
    color=[colors[m] for m in results_df["Model"]]
)

axes[1].set_title("Vapour Composition RMSE")
axes[1].set_ylabel("RMSE (mole fraction)")
axes[1].grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("Model_RMSE_Comparison.png", dpi=300)
plt.show()


# ==========================================
# 6. PLOT COMBINED MODEL RANKING
# ==========================================

plt.figure(figsize=(8, 5))

plt.bar(
    results_df["Model"],
    results_df["Combined NRMSE"],
    color=[colors[m] for m in results_df["Model"]]
)

plt.ylabel("Combined Normalized RMSE")
plt.title("Overall Thermodynamic Model Accuracy")
plt.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("Best_Model_Ranking.png", dpi=300)
plt.show()
