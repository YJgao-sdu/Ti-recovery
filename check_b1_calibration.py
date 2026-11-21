
import os
import numpy as np

DATA_PATH = "logs/b1_calibration/teacher_physics_data.npz"

data = np.load(DATA_PATH)

print("========== B1 DATA AUDIT ==========")

print("Available keys:", data.files)

teacher = data["teacher_obs"].astype(np.float64)

assert teacher.ndim == 2
assert teacher.shape[1] == 20

print("Teacher observation shape:", teacher.shape)
print("Total samples:", len(teacher))

# Check finite values
print("\n--- Finite check ---")
print("NaN:", np.isnan(teacher).sum())
print("Inf:", np.isinf(teacher).sum())

assert np.isfinite(teacher).all()

# Physical task labels
velocity = teacher[:, 0:3]
contact = teacher[:, 10:15]
slip = teacher[:, 18:20]

def describe(name, x):
    print(f"\n--- {name} ---")
    print("Mean:", np.mean(x, axis=0))
    print("Std:", np.std(x, axis=0))
    print("Min:", np.min(x, axis=0))
    print("Max:", np.max(x, axis=0))
    print("P01:", np.percentile(x, 1, axis=0))
    print("P50:", np.percentile(x, 50, axis=0))
    print("P99:", np.percentile(x, 99, axis=0))

describe("Base linear velocity", velocity)
describe("Wheel slip velocity", slip)
describe("Contact labels", contact)

# Contact label audit
print("\n--- Contact label audit ---")

for j in range(5):
    values, counts = np.unique(
        contact[:, j], return_counts=True
    )

    print(
        f"Contact {j}:",
        dict(zip(values.tolist(), counts.tolist()))
    )

# Check auxiliary trajectory information
if "base_height" in data.files:
    height = data["base_height"].reshape(-1)
    print("\nBase height range:", height.min(), height.max())

if "commands" in data.files:
    commands = data["commands"]
    print("Command shape:", commands.shape)
    print("Command min:", commands.min(axis=0))
    print("Command max:", commands.max(axis=0))

if "dones" in data.files:
    dones = data["dones"].reshape(-1)
    print("Episode termination count:", dones.sum())

print("\n========== AUDIT FINISHED ==========")
