# forward kinematics sanity check: DH product vs the geometric formula for the planar part,
# plus a picture of the arm in a few configurations.
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import arm, viz

np.set_printoptions(precision=4, suppress=True)

print("DH table (a, alpha, d, theta offset):")
for i, row in enumerate(arm.DH):
    print("  joint %d: a=%.2f alpha=%6.1f deg d=%.2f offset=%5.1f deg" % (i + 1, row[0], np.rad2deg(row[1]), row[2], np.rad2deg(row[3])))


def fk_by_hand(q):
    """the arm is a base yaw + a planar 3R chain (shoulder, elbow, wrist) with link lengths L2, L3, L5.
    r = L2 c2 + L3 c23 + L5 c234,  z = L1 + L2 s2 + L3 s23 + L5 s234"""
    q1, q2, q3, q4, q5 = q
    r = arm.L2 * np.cos(q2) + arm.L3 * np.cos(q2 + q3) + arm.L5 * np.cos(q2 + q3 + q4)
    z = arm.L1 + arm.L2 * np.sin(q2) + arm.L3 * np.sin(q2 + q3) + arm.L5 * np.sin(q2 + q3 + q4)
    return np.array([r * np.cos(q1), r * np.sin(q1), z])


configs = {
    "all zero (stretched out)": np.zeros(5),
    "home": arm.Q_HOME,
    "reaching forward": np.deg2rad([20, 50, -80, -60, 0]),
    "reaching sideways, low": np.deg2rad([-70, 30, -110, -10, 40]),
}

worst = 0
for name, q in configs.items():
    T = arm.fk(q)
    p_hand = fk_by_hand(q)
    err = np.linalg.norm(T[:3, 3] - p_hand)
    worst = max(worst, err)
    print("\n%s  q = %s deg" % (name, np.rad2deg(q)))
    print(T)
    print("  tip from DH  :", T[:3, 3])
    print("  tip by hand  :", p_hand, "  diff = %.1e m" % err)

rng = np.random.default_rng(0)
for _ in range(1000):
    q = rng.uniform(-np.pi, np.pi, 5)
    worst = max(worst, np.linalg.norm(arm.fk(q)[:3, 3] - fk_by_hand(q)))
print("\nworst DH vs hand-formula mismatch over 1000 random configs: %.2e m" % worst)

fig = plt.figure(figsize=(10, 8))
for i, (name, q) in enumerate(configs.items()):
    ax = fig.add_subplot(2, 2, i + 1, projection="3d")
    viz.setup_axes(ax, name)
    viz.draw_table(ax)
    viz.draw_arm(ax, q)
    p = arm.fk(q)[:3, 3]
    ax.text(p[0], p[1], p[2] + 0.03, "(%.2f, %.2f, %.2f)" % tuple(p), fontsize=7)
    ax.view_init(elev=22, azim=-60)
fig.tight_layout()
fig.savefig("results/fk_configs.png", dpi=120)
print("saved results/fk_configs.png")
