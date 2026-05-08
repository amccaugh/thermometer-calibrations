#%%
file = r"C:\Users\anm16\Documents\temp\2026-05-04 15-38-15 thermometer calibration FAA2.csv"

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv(file)

name = df["name"].iloc[0]
file_dir = os.path.dirname(file)
interpolated_file = os.path.join(file_dir, f"{name}.csv")
raw_file = os.path.join(file_dir, f"{name}-raw-data.csv")

required_columns = {"temperature", "voltage", "current", "time"}
missing = required_columns - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns: {sorted(missing)}")

temperature = df["temperature"].to_numpy()
time_data = df["time"]
if np.issubdtype(time_data.dtype, np.number):
    x_for_rate = time_data.to_numpy(dtype=float)
else:
    parsed_time = pd.to_datetime(time_data, errors="coerce")
    x_for_rate = (parsed_time - parsed_time.iloc[0]).dt.total_seconds().to_numpy()

sweep_rate = np.gradient(temperature, x_for_rate) * 60
smooth_window = max(7, (len(sweep_rate) // 100) * 2 + 1)
kernel = np.ones(smooth_window, dtype=float) / smooth_window
sweep_rate_smooth = np.convolve(sweep_rate, kernel, mode="same")

fig, (ax1, ax2, ax3_first) = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

ax1.plot(df["temperature"], df["voltage"], label="Voltage", linewidth=2, color="tab:blue")
ax1.set_ylabel("Voltage")
ax1.set_title(f"({name}) Voltage vs Temperature")
ax1.grid(True, alpha=0.3)
ax1.legend(loc="best")

ax2.plot(
    df["temperature"], df["current"] * 1e6, label="Current", linewidth=2, color="tab:red"
)
ax2.set_xlabel("Temperature")
ax2.set_ylabel("Current (uA)")
ax2.set_title(f"({name}) Current vs Temperature")
ax2.grid(True, alpha=0.3)
ax2.legend(loc="best")

ax3_first.plot(
    df["temperature"],
    sweep_rate_smooth,
    label="Sweep rate (K/min), smoothed",
    linewidth=2,
    color="tab:orange",
)
ax3_first.set_xlabel("Temperature")
ax3_first.set_ylabel("Sweep rate (K/min)")
ax3_first.set_title(f"({name}) Estimated Sweep Rate vs Temperature")
ax3_first.grid(True, alpha=0.3)
ax3_first.legend(loc="best")

fig.tight_layout()
fig.savefig(os.path.join(file_dir, f"{name}-raw.png"), dpi=150)
plt.show()

# Build a single-valued T(V) calibration curve by averaging across hysteretic branches.
cal = df[["voltage", "temperature"]].dropna().sort_values("voltage").copy()
n_bins = min(600, max(40, len(cal) // 5))
bin_edges = np.linspace(cal["voltage"].min(), cal["voltage"].max(), n_bins + 1)
cal["voltage_bin"] = pd.cut(cal["voltage"], bins=bin_edges, include_lowest=True)

avg_curve = (
    cal.groupby("voltage_bin", observed=False)
    .agg(
        voltage=("voltage", "mean"),
        temperature=("temperature", "mean"),
        temp_min=("temperature", "min"),
        temp_max=("temperature", "max"),
        samples=("temperature", "size"),
    )
    .dropna()
    .sort_values("voltage")
)
interpolated_df = (
    avg_curve[["temperature", "voltage"]]
)
df.to_csv(raw_file, index=False)
interpolated_df.to_csv(interpolated_file, index=False, header=False)

interp_temperature = np.interp(
    cal["voltage"].to_numpy(),
    avg_curve["voltage"].to_numpy(),
    avg_curve["temperature"].to_numpy(),
)
temp_residual = cal["temperature"].to_numpy() - interp_temperature
temp_distance = np.abs(temp_residual)
temp_distance = np.clip(temp_distance, np.finfo(float).eps, None)

fig2, (ax3, ax4) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
ax3.scatter(
    cal["voltage"],
    cal["temperature"],
    s=8,
    alpha=0.25,
    color="gray",
    label="Raw hysteretic data",
)
ax3.plot(
    avg_curve["voltage"],
    avg_curve["temperature"],
    color="tab:green",
    linewidth=2.5,
    label="Averaged/interpolated T(V)",
)
ax3.set_ylabel("Temperature")
ax3.set_title(f"({name}) Temperature vs Voltage with Branch-Averaged Calibration")
ax3.set_yscale("log")
ax3.set_yticks([40, 20, 10, 4])
ax3.set_yticklabels(["40 K", "20 K", "10 K", "4 K"])
ax3.grid(True, alpha=0.3)
ax3.grid(axis="y", which="major", alpha=0.6)
ax3.legend(loc="best")

ax4.scatter(
    cal["voltage"],
    temp_distance,
    s=8,
    alpha=0.35,
    color="tab:purple",
    label="|Temperature - T_interp|",
)
ax4.set_xlabel("Voltage")
ax4.set_ylabel("|Temperature - T_interp|")
ax4.set_title(f"({name}) Distance from Interpolated T(V) Curve")
ax4.set_yscale("log")
ax4.grid(True, alpha=0.3)
ax4.legend(loc="best")

fig2.tight_layout()
fig2.savefig(os.path.join(file_dir, f"{name}-calibration.png"), dpi=150)
plt.show()
