# inverse kinematics checks:
#  1. closed form IK -> FK round trip error over many random targets
#  2. both elbow branches for the same target (picture)
#  3. CLIK (numerical, jacobian based) converging to the same point, error decay vs the e^{-Kt} prediction
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import arm, viz

rng = np.random.default_rng(3)

# 1. round trip
errs, n_both, n_one, n_none, n_inlim = [], 0, 0, 0, 0
t0 = time.perf_counter()
N = 5000
for _ in range(N):
    q = rng.uniform(arm.LIMITS[:, 0], arm.LIMITS[:, 1])
    p, pitch, roll = arm.tool_pose(q)
    sols = [arm.ik(*p, pitch=pitch, roll=roll, elbow=e) for e in ("up", "down")]
    sols = [s for s in sols if s is not None]
    if len(sols) == 2: n_both += 1
    elif len(sols) == 1: n_one += 1
    else: n_none += 1
    for s in sols:
        p2, pitch2, _ = arm.tool_pose(s)
        errs.append(np.linalg.norm(p2 - p))
        if arm.in_limits(s):
            n_inlim += 1
dt = (time.perf_counter() - t0) / N
errs = np.array(errs)
print("IK round trip over %d random poses:" % N)
print("  position error: max %.2e m, mean %.2e m" % (errs.max(), errs.mean()))
print("  both branches real: %d, one: %d, none: %d" % (n_both, n_one, n_none))
print("  solutions inside joint limits: %d of %d" % (n_inlim, len(errs)))
print("  time per IK call (both branches): %.1f us" % (dt * 1e6))

# 2. branches picture
target = (0.22, 0.08, 0.06)
fig = plt.figure(figsize=(10, 4.5))
for i, elbow in enumerate(("up", "down")):
    q = arm.ik(*target, pitch=np.deg2rad(-60), elbow=elbow)
    ax = fig.add_subplot(1, 2, i + 1, projection="3d")
    viz.setup_axes(ax, "elbow %s: q = %s deg\nin limits: %s" % (elbow, np.round(np.rad2deg(q), 1), arm.in_limits(q)))
    viz.draw_table(ax)
    viz.draw_arm(ax, q)
    ax.scatter(*target, color="r", s=40)
    ax.view_init(elev=15, azim=-70)
fig.tight_layout()
fig.savefig("results/ik_branches.png", dpi=120)

# 3. CLIK from home to a fixed point, with a few gains. the notes say e(t) = e(0) exp(-K t)
p_goal = np.array([0.20, 0.10, 0.05])
pitch_goal = -np.pi / 2
dt = 0.01
fig, ax = plt.subplots(figsize=(7, 4))
for K in (2.0, 5.0, 10.0):
    q = arm.Q_HOME.copy()
    e_hist, t_hist = [], []
    for k in range(int(3.0 / dt)):
        q, e = arm.clik_step(q, p_goal, np.zeros(3), pitch_goal, dt, K=K)
        e_hist.append(np.linalg.norm(e[:3]))
        t_hist.append(k * dt)
    e_hist = np.array(e_hist)
    ax.semilogy(t_hist, e_hist * 1000, label="CLIK, K = %g" % K)
    ax.semilogy(t_hist, e_hist[0] * 1000 * np.exp(-K * np.array(t_hist)), "k--", lw=0.8)
    q_cf = arm.ik(*p_goal, pitch=pitch_goal, roll=arm.tool_pose(q)[2])
    print("K=%4.1f: final CLIK error %.3e m, joint diff vs closed form %.2e rad" % (K, e_hist[-1], np.abs(arm.wrap(q - q_cf)).max()))
ax.set_xlabel("time (s)")
ax.set_ylabel("position error (mm)")
ax.set_title("CLIK convergence from home to (0.20, 0.10, 0.05), dashed = e0 exp(-Kt)")
ax.grid(alpha=0.3, which="both")
ax.legend()
fig.tight_layout()
fig.savefig("results/clik_convergence.png", dpi=120)
print("saved results/ik_branches.png, results/clik_convergence.png")
