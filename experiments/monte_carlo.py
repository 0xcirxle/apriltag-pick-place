# run the perception + planning part of the pipeline on many random scenes and
# count how often the grasp would have worked. no animation, so it's quick.
# "works" = gripper centre within 5 mm of the cube centre (6 cm opening on a 5 cm cube)
# and fingers within 10 deg of the faces.
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pick_place as pp

TOL_MM, TOL_DEG = 5.0, 10.0
NSCENE = 40

# conditions: image noise with a perfect calibration, then the kind of errors we would actually
# have on a real setup: the reference tag not exactly where we measured it, focal length a bit off
conditions = [
    ("noise 2", dict(noise=2)),
    ("noise 5", dict(noise=5)),
    ("noise 10", dict(noise=10)),
    ("noise 20", dict(noise=20)),
    ("noise 30", dict(noise=30)),
    ("ref tag off 3 mm / 1 deg", dict(noise=5, ref_err=(0.002, -0.002, np.deg2rad(1.0)))),
    ("ref tag off 5 mm / 2 deg", dict(noise=5, ref_err=(0.004, 0.003, np.deg2rad(2.0)))),
    ("focal length 2% off", dict(noise=5, f_err=0.02)),
    ("focal length 5% off", dict(noise=5, f_err=0.05)),
    ("ref 3mm/1deg + focal 2%", dict(noise=5, ref_err=(0.002, -0.002, np.deg2rad(1.0)), f_err=0.02)),
]

rows, box_data = [], []
for label, kw in conditions:
    offs, yaws, n_ok, n_det, n_reach = [], [], 0, 0, 0
    offs_pnp = []
    for seed in range(NSCENE):
        T_obj, T_place = pp.make_scene(100 + seed)
        perc = pp.perceive(T_obj, T_place, seed=seed, verbose=False, **kw)
        if not perc["ok"]:
            continue
        n_det += 1
        segs = pp.plan(perc, verbose=False)
        if segs is None:
            continue
        n_reach += 1
        dxy, dyaw = pp.grasp_offset(perc, T_obj)
        offs.append(dxy); yaws.append(dyaw)
        offs_pnp.append(np.linalg.norm(perc["obj_xy_pnp"] - T_obj[:2, 3]) * 1000)
        if dxy < TOL_MM and dyaw < TOL_DEG:
            n_ok += 1
    rows.append(dict(condition=label, scenes=NSCENE, detected=n_det, reachable=n_reach, success=n_ok,
                     offset_mean_mm=np.mean(offs), offset_max_mm=np.max(offs),
                     offset_pnp_mean_mm=np.mean(offs_pnp),
                     yaw_mean_deg=np.mean(yaws), yaw_max_deg=np.max(yaws)))
    box_data.append(offs)
    print("%-26s detected %2d/%d  reachable %2d  success %2d  offset mean %.2f max %.2f mm (pnp would be %.2f)  yaw mean %.2f max %.2f deg" % (
        label, n_det, NSCENE, n_reach, n_ok, np.mean(offs), np.max(offs), np.mean(offs_pnp), np.mean(yaws), np.max(yaws)))

with open("results/monte_carlo.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

labels = [c[0] for c in conditions]
fig, axs = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
axs[0].boxplot(box_data, tick_labels=labels)
axs[0].axhline(TOL_MM, color="r", ls="--", lw=1, label="grasp tolerance (%.0f mm)" % TOL_MM)
axs[0].set_ylabel("gripper offset from cube centre (mm)")
axs[0].legend()
axs[0].grid(alpha=0.3)
axs[1].bar(labels, [100 * r["success"] / NSCENE for r in rows], color="C2")
axs[1].bar(labels, [100 * (NSCENE - r["detected"]) / NSCENE for r in rows],
           bottom=[100 * r["success"] / NSCENE for r in rows], color="0.7", label="tag not detected / occluded")
axs[1].set_ylabel("% of scenes")
axs[1].set_ylim(0, 105)
axs[1].legend(fontsize=8)
axs[1].grid(alpha=0.3, axis="y")
plt.setp(axs[1].get_xticklabels(), rotation=25, ha="right", fontsize=8)
fig.suptitle("%d random scenes per condition, green = successful grasp" % NSCENE)
fig.tight_layout()
fig.savefig("results/monte_carlo.png", dpi=120)
print("saved results/monte_carlo.csv, results/monte_carlo.png")
