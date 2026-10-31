# how good is the tag pose? sweep image noise, compare corner refinement methods,
# and compare full PnP position vs the ray/table-plane trick. ground truth is known because
# we render the images ourselves.
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import camera, tag_detect as td
from transforms import pose_xy_yaw, inv, yaw_of, rot_angle, make_T

TAG = 0.04
CUBE = 0.05
T_REF = pose_xy_yaw(0.10, -0.12, 0.0, 0.0)
rng = np.random.default_rng(0)


def random_scene():
    while True:
        r = rng.uniform(0.12, 0.28)
        th = rng.uniform(np.deg2rad(-40), np.deg2rad(55))
        x, y = r * np.cos(th), r * np.sin(th)
        if np.hypot(x - T_REF[0, 3], y - T_REF[1, 3]) > 0.09:     # don't park the cube on the reference tag
            break
    return pose_xy_yaw(x, y, CUBE / 2, rng.uniform(-np.pi, np.pi))


def one_trial(T_obj, noise, detector, seed):
    T_tag = T_obj @ make_T(t=[0, 0, CUBE / 2])
    img = camera.render([(0, TAG, T_REF), (5, TAG, T_tag)], noise=noise, seed=seed, boxes=[(T_obj, CUBE, 150)])
    dets = td.detect(img, detector)
    if 0 not in dets or 5 not in dets:
        return None
    T_cam_ref = td.estimate_pose(dets[0], TAG)
    T_cam_tag = td.estimate_pose(dets[5], TAG)
    # errors in the camera frame (pure pnp)
    T_true_cam = inv(camera.T_BASE_CAM_TRUE) @ T_tag
    e_cam = np.linalg.norm(T_cam_tag[:3, 3] - T_true_cam[:3, 3])
    e_rot = rot_angle(T_cam_tag[:3, :3].T @ T_true_cam[:3, :3])
    # in the base frame, through the calibrated camera
    T_base_cam = td.camera_from_reference_tag(T_cam_ref, T_REF)
    e_campos = np.linalg.norm(T_base_cam[:3, 3] - camera.CAM_POS)
    T_base_tag = T_base_cam @ T_cam_tag
    e_pnp = np.linalg.norm(T_base_tag[:3, 3] - T_tag[:3, 3])
    p_plane = td.point_on_plane(dets[5], T_base_cam, CUBE)
    e_plane = np.linalg.norm(p_plane[:2] - T_tag[:2, 3])
    e_yaw = abs(np.arctan2(np.sin(yaw_of(T_base_tag) - yaw_of(T_tag)), np.cos(yaw_of(T_base_tag) - yaw_of(T_tag))))
    return dict(e_cam=e_cam, e_rot=e_rot, e_campos=e_campos, e_pnp=e_pnp, e_plane=e_plane, e_yaw=e_yaw)


# ---- 1. noise sweep with the default (apriltag refinement) detector
noises = [0, 2, 5, 10, 15, 20, 30, 40]
NTRIAL = 40
rows = []
det = td.make_detector("apriltag")
scenes = [random_scene() for _ in range(NTRIAL)]
for noise in noises:
    res = [one_trial(T, noise, det, seed=i) for i, T in enumerate(scenes)]
    ok = [r for r in res if r is not None]
    row = dict(noise=noise, detected=len(ok) / NTRIAL)
    for k in ("e_cam", "e_pnp", "e_plane", "e_campos", "e_yaw", "e_rot"):
        v = np.array([r[k] for r in ok]) if ok else np.array([np.nan])
        row[k + "_mean"] = v.mean()
        row[k + "_p90"] = np.percentile(v, 90)
    rows.append(row)
    print("noise %2d: detected %3.0f%%  pos err base: pnp %.2f mm / plane %.2f mm   yaw %.2f deg   cam pos %.2f mm" % (
        noise, 100 * row["detected"], 1000 * row["e_pnp_mean"], 1000 * row["e_plane_mean"],
        np.rad2deg(row["e_yaw_mean"]), 1000 * row["e_campos_mean"]))

with open("results/tag_pose_error.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

fig, axs = plt.subplots(1, 3, figsize=(14, 4))
n = [r["noise"] for r in rows]
axs[0].plot(n, [1000 * r["e_pnp_mean"] for r in rows], "o-", label="PnP position (mean)")
axs[0].plot(n, [1000 * r["e_pnp_p90"] for r in rows], "o--", color="C0", alpha=0.5, label="PnP (90th pct)")
axs[0].plot(n, [1000 * r["e_plane_mean"] for r in rows], "s-", label="ray-plane position (mean)")
axs[0].plot(n, [1000 * r["e_plane_p90"] for r in rows], "s--", color="C1", alpha=0.5, label="ray-plane (90th pct)")
axs[0].set_xlabel("image noise std (gray levels)")
axs[0].set_ylabel("object position error in base frame (mm)")
axs[0].legend(fontsize=8)
axs[0].grid(alpha=0.3)
axs[1].plot(n, [np.rad2deg(r["e_yaw_mean"]) for r in rows], "o-", label="yaw error (mean)")
axs[1].plot(n, [np.rad2deg(r["e_yaw_p90"]) for r in rows], "o--", alpha=0.5, label="yaw (90th pct)")
axs[1].set_xlabel("image noise std (gray levels)")
axs[1].set_ylabel("yaw error (deg)")
axs[1].legend(fontsize=8)
axs[1].grid(alpha=0.3)
axs[2].plot(n, [100 * r["detected"] for r in rows], "o-")
axs[2].set_xlabel("image noise std (gray levels)")
axs[2].set_ylabel("both tags detected (%)")
axs[2].set_ylim(0, 105)
axs[2].grid(alpha=0.3)
fig.suptitle("tag pose error vs image noise (%d random object poses per level, 4 cm tag, 640x480)" % NTRIAL)
fig.tight_layout()
fig.savefig("results/tag_pose_error.png", dpi=120)

# ---- 2. corner refinement comparison at a moderate noise level
print("\ncorner refinement method comparison (noise = 5):")
ref_rows = []
for method in ("none", "subpix", "apriltag"):
    d = td.make_detector(method)
    res = [one_trial(T, 5, d, seed=i) for i, T in enumerate(scenes)]
    ok = [r for r in res if r is not None]
    e_cam = 1000 * np.mean([r["e_cam"] for r in ok])
    e_pnp = 1000 * np.mean([r["e_pnp"] for r in ok])
    e_plane = 1000 * np.mean([r["e_plane"] for r in ok])
    e_yaw = np.rad2deg(np.mean([r["e_yaw"] for r in ok]))
    print("  %-9s  cam-frame pos %.2f mm   base pnp %.2f mm   base plane %.2f mm   yaw %.2f deg" % (method, e_cam, e_pnp, e_plane, e_yaw))
    ref_rows.append(dict(method=method, e_cam_mm=e_cam, e_pnp_mm=e_pnp, e_plane_mm=e_plane, e_yaw_deg=e_yaw))
with open("results/corner_refinement.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(ref_rows[0].keys()))
    w.writeheader()
    w.writerows(ref_rows)

# save one noisy example picture
T_obj = scenes[0]
T_tag = T_obj @ make_T(t=[0, 0, CUBE / 2])
img = camera.render([(0, TAG, T_REF), (5, TAG, T_tag)], noise=20, seed=0, boxes=[(T_obj, CUBE, 150)])
dets = td.detect(img)
poses = {k: td.estimate_pose(c, TAG) for k, c in dets.items()}
import cv2
cv2.imwrite("results/tag_noisy_example.png", td.draw(img, dets, poses, TAG))
print("saved results/tag_pose_error.png/.csv, results/corner_refinement.csv, results/tag_noisy_example.png")
