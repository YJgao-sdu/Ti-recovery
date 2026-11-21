
import os
import json
import numpy as np

DATA_PATH = "logs/b1_calibration/teacher_physics_data.npz"

SAVE_DIR = "logs/b1_calibration"
SAVE_PATH = os.path.join(
    SAVE_DIR,
    "b1_normalization.npz"
)

data = np.load(DATA_PATH)

teacher = data["teacher_obs"].astype(np.float64)

assert teacher.ndim == 2
assert teacher.shape[1] == 20
assert np.isfinite(teacher).all()

# Extract physical labels
velocity = teacher[:, 0:3]
contact = teacher[:, 10:15]
slip = teacher[:, 18:20]

# Velocity: all valid samples
velocity_mean = velocity.mean(axis=0)
velocity_std = velocity.std(axis=0, ddof=0)

# Slip: calculate statistics separately
# using each wheel's contact samples

left_mask = contact[:, 3] > 0.5
right_mask = contact[:, 4] > 0.5

slip_mean = np.array([
    slip[left_mask, 0].mean(),
    slip[right_mask, 1].mean()
])

slip_std = np.array([
    slip[left_mask, 0].std(ddof=0),
    slip[right_mask, 1].std(ddof=0)
])

eps = 1e-6

velocity_std = np.maximum(velocity_std, eps)
slip_std = np.maximum(slip_std, eps)

# Calculate normalized labels for verification
velocity_norm = (
    velocity - velocity_mean
) / velocity_std

# Use the same contact masks as normalization

left_mask = contact[:, 3] > 0.5
right_mask = contact[:, 4] > 0.5

left_slip_norm = (
    slip[left_mask, 0] - slip_mean[0]
) / slip_std[0]

right_slip_norm = (
    slip[right_mask, 1] - slip_mean[1]
) / slip_std[1]

# Print results
print("\n========== NORMALIZATION RESULTS ==========")

print("\nVelocity mean:")
print(velocity_mean.tolist())

print("\nVelocity std:")
print(velocity_std.tolist())

print("\nSlip mean:")
print(slip_mean.tolist())

print("\nSlip std:")
print(slip_std.tolist())

print("\nNormalized velocity mean:")
print(velocity_norm.mean(axis=0))

print("\nNormalized velocity std:")
print(velocity_norm.std(axis=0))

print("\nNormalized slip mean:")
print([
    left_slip_norm.mean(),
    right_slip_norm.mean()
])

print("\nNormalized slip std:")
print([
    left_slip_norm.std(),
    right_slip_norm.std()
])

# Save parameters
os.makedirs(SAVE_DIR, exist_ok=True)

np.savez_compressed(
    SAVE_PATH,
    velocity_mean=velocity_mean.astype(np.float32),
    velocity_std=velocity_std.astype(np.float32),
    slip_mean=slip_mean.astype(np.float32),
    slip_std=slip_std.astype(np.float32),
)

print("\nSaved normalization parameters:")
print(SAVE_PATH)
