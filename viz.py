# viz.py
# matplotlib drawing helpers for the 3d scene (arm, table, tags, cube, camera)

import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
import arm
from transforms import apply


def setup_axes(ax, title=None):
    ax.set_xlim(-0.12, 0.48)
    ax.set_ylim(-0.30, 0.30)
    ax.set_zlim(0.0, 0.50)
    ax.set_box_aspect((0.6, 0.6, 0.5))
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    if title:
        ax.set_title(title)


def draw_table(ax):
    x0, x1, y0, y1 = -0.12, 0.48, -0.30, 0.30
    quad = [[(x0, y0, 0), (x1, y0, 0), (x1, y1, 0), (x0, y1, 0)]]
    ax.add_collection3d(Poly3DCollection(quad, color="#e8e0d0", alpha=0.6, zorder=0))


def draw_arm(ax, q, grip=1.0, color="k"):
    """draws links as thick lines, joints as dots, and a simple 2 finger gripper.
    grip: 1 = open, 0 = closed. returns the artists so an animation can remove them."""
    arts = []
    pts = arm.joint_positions(q)
    # base pedestal first so the links draw over it
    arts += ax.plot([0, 0], [0, 0], [0, arm.L1], "-", color="#777777", lw=7)
    arts += ax.plot(pts[1:, 0], pts[1:, 1], pts[1:, 2], "-", color=color, lw=4, solid_capstyle="round")
    arts += ax.plot(pts[1:4, 0], pts[1:4, 1], pts[1:4, 2], "o", color="#cc4400", ms=6)

    # gripper: two fingers along the tool z axis, offset along tool x
    T = arm.fk(q)
    tip = T[:3, 3]
    xt, zt = T[:3, 0], T[:3, 2]
    half = 0.012 + 0.018 * grip          # half opening (6 cm open, 2.4 cm closed)
    back = tip - 0.03 * zt               # where the fingers attach (the "palm")
    for s in (-1, 1):
        f0 = back + s * half * xt
        f1 = tip + s * half * xt
        arts += ax.plot(*zip(f0, f1), "-", color="#2255aa", lw=2.5)
    arts += ax.plot(*zip(back - half * xt, back + half * xt), "-", color="#2255aa", lw=2.5)
    return arts


def draw_tag(ax, T, size, color="k", label=None):
    s = size / 2
    sq = np.array([[-s, -s, 0], [s, -s, 0], [s, s, 0], [-s, s, 0]])
    pts = apply(T, sq)
    arts = [ax.add_collection3d(Poly3DCollection([pts], facecolor=color, edgecolor="k", alpha=0.9))]
    # little arrow for the tag x axis
    o = T[:3, 3]
    xa = o + T[:3, 0] * size * 0.8
    arts += ax.plot([o[0], xa[0]], [o[1], xa[1]], [o[2], xa[2]], "-", color="r", lw=1.5)
    if label:
        arts.append(ax.text(o[0], o[1], o[2] + 0.02, label, fontsize=8))
    return arts


def draw_cube(ax, T, size, color="#d9a441"):
    """T is the pose of the cube centre"""
    s = size / 2
    c = np.array([[x, y, z] for x in (-s, s) for y in (-s, s) for z in (-s, s)])
    p = apply(T, c)
    faces_idx = [[0, 1, 3, 2], [4, 5, 7, 6], [0, 1, 5, 4], [2, 3, 7, 6], [0, 2, 6, 4], [1, 3, 7, 5]]
    faces = [p[i] for i in faces_idx]
    return [ax.add_collection3d(Poly3DCollection(faces, facecolor=color, edgecolor="k", lw=0.5, alpha=0.95))]


def draw_camera(ax, T_base_cam, K, W, H, depth=0.08, color="#444444"):
    """small pyramid showing where the camera is and roughly what it sees"""
    o = T_base_cam[:3, 3]
    Kinv = np.linalg.inv(K)
    corners = []
    for u, v in [(0, 0), (W, 0), (W, H), (0, H)]:
        d = Kinv @ np.array([u, v, 1.0])
        d = d / d[2] * depth
        corners.append(apply(T_base_cam, d))
    segs = [[o, c] for c in corners] + [[corners[i], corners[(i + 1) % 4]] for i in range(4)]
    ax.add_collection3d(Line3DCollection(segs, colors=color, lw=1))
    ax.scatter(*o, color=color, s=20)
    ax.text(o[0], o[1], o[2] + 0.02, "camera", fontsize=8)


def draw_path(ax, pts, **kw):
    pts = np.asarray(pts)
    return ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], **kw)
