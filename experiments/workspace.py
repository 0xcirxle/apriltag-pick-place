# reachable workspace of the arm: random sampling within joint limits (side + top view),
# and the part of the table we can actually do a top-down grasp on.
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import arm

rng = np.random.default_rng(0)
N = 30000
Q = rng.uniform(arm.LIMITS[:, 0], arm.LIMITS[:, 1], (N, 5))
P = np.array([arm.fk(q)[:3, 3] for q in Q])
R = np.hypot(P[:, 0], P[:, 1])

fig, axs = plt.subplots(1, 3, figsize=(15, 4.8))
axs[0].scatter(R, P[:, 2], s=1, alpha=0.3)
axs[0].set_xlabel("r = sqrt(x^2+y^2) (m)")
axs[0].set_ylabel("z (m)")
axs[0].set_title("side view (all joints random within limits)")
axs[0].axhline(0, color="k", lw=0.8)
axs[0].set_aspect("equal")
axs[0].grid(alpha=0.3)

axs[1].scatter(P[:, 0], P[:, 1], s=1, alpha=0.3)
axs[1].set_xlabel("x (m)")
axs[1].set_ylabel("y (m)")
axs[1].set_title("top view")
axs[1].set_aspect("equal")
axs[1].grid(alpha=0.3)

# top-down grasp map: for each (x, y) on the table at the grasp height, does IK give a config inside limits?
z_grasp = 0.025
xs = np.linspace(-0.35, 0.35, 141)
ys = np.linspace(-0.35, 0.35, 141)
ok = np.zeros((len(ys), len(xs)))
for i, y in enumerate(ys):
    for j, x in enumerate(xs):
        for elbow in ("up", "down"):
            q = arm.ik(x, y, z_grasp, elbow=elbow)
            if q is not None and arm.in_limits(q):
                ok[i, j] = 1 if elbow == "up" else 0.5
                break
axs[2].imshow(ok, origin="lower", extent=[xs[0], xs[-1], ys[0], ys[-1]], cmap="Greens", vmin=0, vmax=1)
axs[2].plot(0, 0, "ko")
axs[2].set_xlabel("x (m)")
axs[2].set_ylabel("y (m)")
axs[2].set_title("where a top-down grasp at z=%.3f m is possible\n(dark = elbow up, light = only elbow down)" % z_grasp)
axs[2].grid(alpha=0.3)
fig.tight_layout()
fig.savefig("results/workspace.png", dpi=120)

rr = np.hypot(*np.meshgrid(xs, ys))[ok > 0]
print("max reach (any pose): %.3f m" % R.max())
print("top-down grasp on the table possible for r in [%.3f, %.3f] m" % (rr.min(), rr.max()))
print("theoretical: wrist must be within L2+L3 = %.2f m of the shoulder" % (arm.L2 + arm.L3))
print("saved results/workspace.png")
