# trajectories: cubic vs quintic for one move, and joint-space cubic vs straight line (CLIK)
# between the two hover points of a pick and place.
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import arm, trajectory as tr
import pick_place as pp

# 1. cubic vs quintic, same move same duration
q0, qf, T = np.deg2rad(20), np.deg2rad(60), 4.0
t = np.linspace(0, T, 400)
qc, qdc, qddc = tr.cubic(q0, qf, T, t)
qq, qdq, qddq = tr.quintic(q0, qf, T, t)
fig, axs = plt.subplots(1, 3, figsize=(13, 3.6))
for ax, (a, b), lab in zip(axs, [(qc, qq), (qdc, qdq), (qddc, qddq)], ["position (deg)", "velocity (deg/s)", "acceleration (deg/s^2)"]):
    ax.plot(t, np.rad2deg(a[:, 0]), label="cubic")
    ax.plot(t, np.rad2deg(b[:, 0]), label="quintic")
    ax.set_xlabel("t (s)")
    ax.set_ylabel(lab)
    ax.grid(alpha=0.3)
axs[0].legend()
axs[0].set_title("20 deg -> 60 deg in 4 s")
fig.tight_layout()
fig.savefig("results/cubic_vs_quintic.png", dpi=120)
print("cubic peak vel %.2f deg/s (theory 1.5*dq/T = %.2f), quintic peak %.2f deg/s (theory 1.875*dq/T = %.2f)" % (
    np.rad2deg(qdc.max()), np.rad2deg(1.5 * (qf - q0) / T), np.rad2deg(qdq.max()), np.rad2deg(1.875 * (qf - q0) / T)))
print("cubic accel at t=0: %.2f deg/s^2, quintic: %.2f" % (np.rad2deg(qddc[0, 0]), np.rad2deg(qddq[0, 0])))

# 2. joint-space move vs cartesian straight line between two hover points
p_a = np.array([0.25, 0.04, 0.085])
p_b = np.array([0.16, 0.16, 0.085])
q_a = arm.ik(*p_a, roll=0.3)
q_b = arm.ik(*p_b, roll=0.8)
_, qj, _, _, _ = tr.plan_through([q_a, q_b], vmax=pp.VMAX, dt=pp.DT)
path_j = np.array([arm.fk(q)[:3, 3] for q in qj])
qc_, pdes, err = pp.clik_line(q_a, p_a, p_b, -np.pi / 2, v=0.08)
path_c = np.array([arm.fk(q)[:3, 3] for q in qc_])

# distance of the joint-space path from the straight line
d = p_b - p_a
dev = np.linalg.norm(np.cross(path_j - p_a, d), axis=1) / np.linalg.norm(d)
print("joint-space cubic: max deviation from straight line %.1f mm, z wanders by %.1f mm" % (1000 * dev.max(), 1000 * (path_j[:, 2].max() - path_j[:, 2].min())))
print("CLIK straight line: max tracking error %.3f mm" % (1000 * err.max()))

fig, axs = plt.subplots(1, 2, figsize=(11, 4.5))
axs[0].plot(path_j[:, 0], path_j[:, 1], label="joint-space cubic")
axs[0].plot(path_c[:, 0], path_c[:, 1], label="CLIK straight line")
axs[0].plot(*p_a[:2], "ko"); axs[0].plot(*p_b[:2], "ks")
axs[0].plot(0, 0, "r^", label="base")
axs[0].set_aspect("equal"); axs[0].grid(alpha=0.3); axs[0].legend(fontsize=8)
axs[0].set_xlabel("x (m)"); axs[0].set_ylabel("y (m)"); axs[0].set_title("top view of the tool path")
tj = np.arange(len(qj)) * pp.DT
tc = np.arange(len(qc_)) * pp.DT
for i in range(4):
    axs[1].plot(tj, np.rad2deg(qj[:, i]), color="C%d" % i, label="q%d cubic" % (i + 1))
    axs[1].plot(tc, np.rad2deg(qc_[:, i]), "--", color="C%d" % i, label="q%d CLIK" % (i + 1))
axs[1].set_xlabel("t (s)"); axs[1].set_ylabel("joint angle (deg)"); axs[1].grid(alpha=0.3)
axs[1].legend(fontsize=7, ncol=2); axs[1].set_title("joint angles for the same move")
fig.tight_layout()
fig.savefig("results/joint_vs_cartesian.png", dpi=120)
print("saved results/cubic_vs_quintic.png, results/joint_vs_cartesian.png")
