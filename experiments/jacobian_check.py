# jacobian: analytic (column rule) vs finite differences, and manipulability over the table
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import arm

np.set_printoptions(precision=4, suppress=True)
rng = np.random.default_rng(0)
worst = 0
for _ in range(500):
    q = rng.uniform(arm.LIMITS[:, 0], arm.LIMITS[:, 1])
    worst = max(worst, np.abs(arm.jacobian(q)[:3] - arm.jacobian_numeric(q)).max())
print("max |J_analytic - J_numeric| over 500 configs: %.2e" % worst)

q = arm.Q_HOME
print("\nJ at home config (6x5):")
print(arm.jacobian(q))
print("manipulability at home: %.5f" % arm.manipulability(q))

# stretched out = singular (shoulder, elbow, wrist all in line)
q_s = np.zeros(5)
print("manipulability fully stretched: %.2e  (singular)" % arm.manipulability(q_s))
print("rank of J at stretch:", np.linalg.matrix_rank(arm.jacobian(q_s)[:3]))

# manipulability map for top-down grasp configs on the table
xs = np.linspace(-0.32, 0.32, 97)
ys = np.linspace(-0.32, 0.32, 97)
W = np.full((len(ys), len(xs)), np.nan)
for i, y in enumerate(ys):
    for j, x in enumerate(xs):
        q = arm.ik(x, y, 0.025)
        if q is not None and arm.in_limits(q):
            W[i, j] = arm.manipulability(q)
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(W, origin="lower", extent=[xs[0], xs[-1], ys[0], ys[-1]], cmap="viridis")
fig.colorbar(im, label="w = sqrt(det(Jp Jp^T))")
ax.plot(0, 0, "wo")
ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")
ax.set_title("manipulability of the top-down grasp pose (elbow up)")
fig.tight_layout()
fig.savefig("results/manipulability_map.png", dpi=120)
print("saved results/manipulability_map.png")
