# pick_place.py
# the whole pipeline in one place:
#   render camera image -> detect apriltags -> tag poses in camera frame
#   -> camera extrinsics from the reference tag -> object + place poses in base frame
#   -> IK + joint space cubics for the big moves, CLIK for the straight up/down bits
#   -> simulate, plot, animate
#
# run:  python pick_place.py            (default scene)
#       python pick_place.py --seed 3 --noise 6 --video

import argparse
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation

import arm
import trajectory as tr
import camera
import tag_detect as td
import viz
from transforms import pose_xy_yaw, inv, apply, yaw_of, make_T

TAG_SIZE = 0.04       # black square edge, m
CUBE = 0.05           # object is a 5 cm wooden cube with the tag stuck on top (tag + white margin = 5 cm, fits exactly)
REF_TAG_ID = 0
OBJ_TAG_ID = 5
PLACE_TAG_ID = 7
T_BASE_REF = pose_xy_yaw(0.10, -0.12, 0.0, 0.0)    # reference tag taped to the table, measured with a ruler

DT = 0.02
VMAX = np.deg2rad(70)       # joint speed limit for the cubic segments
V_LIN = 0.05                # m/s for the straight line CLIK segments
HOVER = 0.06                # approach height above the grasp
GRASP_Z = CUBE / 2          # grab the cube around its middle
PLACE_CLEAR = 0.004         # drop from slightly above the table


def make_scene(seed=0):
    """random object pose in the reachable part of the table, fixed place tag"""
    rng = np.random.default_rng(seed)
    while True:
        r = rng.uniform(0.14, 0.26)
        th = rng.uniform(np.deg2rad(-35), np.deg2rad(50))
        x, y = r * np.cos(th), r * np.sin(th)
        # keep away from the place spot and the ref tag
        if np.hypot(x - 0.16, y - 0.16) > 0.07 and np.hypot(x - 0.10, y + 0.12) > 0.06:
            break
    yaw = rng.uniform(-np.pi, np.pi)
    T_obj = pose_xy_yaw(x, y, CUBE / 2, yaw)                  # cube centre
    T_place = pose_xy_yaw(0.16, 0.16, 0.0, np.deg2rad(10))    # place tag on the table
    return T_obj, T_place


def perceive(T_obj, T_place, noise=3.0, seed=0, verbose=True, f_err=0.0, ref_err=(0.0, 0.0, 0.0)):
    """camera image -> everything we need in the base frame. all estimated, nothing taken from ground truth
    except the ref tag pose which we 'measured' when setting up.
    f_err:   fractional error in the focal length we *think* the camera has (the image is rendered with the real one)
    ref_err: (dx, dy, dyaw) how far the reference tag really is from where we measured it"""
    T_tag_on_cube = T_obj @ make_T(t=[0, 0, CUBE / 2])        # tag is on the top face
    T_ref_true = pose_xy_yaw(T_BASE_REF[0, 3] + ref_err[0], T_BASE_REF[1, 3] + ref_err[1], 0.0, ref_err[2])
    K_true = camera.K.copy()
    K_true[0, 0] *= (1 + f_err)
    K_true[1, 1] *= (1 + f_err)
    tags = [(REF_TAG_ID, TAG_SIZE, T_ref_true), (OBJ_TAG_ID, TAG_SIZE, T_tag_on_cube), (PLACE_TAG_ID, TAG_SIZE, T_place)]
    img = camera.render(tags, noise=noise, seed=seed, boxes=[(T_obj, CUBE, 150)], K_true=K_true)
    dets = td.detect(img)
    poses = {t: td.estimate_pose(c, TAG_SIZE) for t, c in dets.items()}
    out = {"img": img, "dets": dets, "poses_cam": poses, "ok": False}
    if not all(t in poses for t in (REF_TAG_ID, OBJ_TAG_ID, PLACE_TAG_ID)):
        if verbose:
            print("missing tags, saw", sorted(poses))
        return out

    T_base_cam = td.camera_from_reference_tag(poses[REF_TAG_ID], T_BASE_REF)
    # sanity check: we know roughly where the camera is bolted. if the calibration is way off
    # the reference tag was probably misread (or something is sitting on it)
    if np.linalg.norm(T_base_cam[:3, 3] - camera.CAM_POS) > 0.05:
        if verbose:
            print("camera calibration looks wrong, got", T_base_cam[:3, 3])
        return out
    T_base_objtag = T_base_cam @ poses[OBJ_TAG_ID]
    T_base_place = T_base_cam @ poses[PLACE_TAG_ID]

    # the pnp depth of a 4 cm tag is the shaky part, so for the position use the
    # ray / table-plane intersection (we know the cube height). yaw from pnp.
    p_obj = td.point_on_plane(dets[OBJ_TAG_ID], T_base_cam, CUBE)
    p_place = td.point_on_plane(dets[PLACE_TAG_ID], T_base_cam, 0.0)

    out.update(ok=True, T_base_cam=T_base_cam,
               obj_xy=p_obj[:2], obj_yaw=yaw_of(T_base_objtag),
               place_xy=p_place[:2], place_yaw=yaw_of(T_base_place),
               obj_xy_pnp=T_base_objtag[:2, 3], obj_z_pnp=T_base_objtag[2, 3])
    if verbose:
        print("camera position est %s  true %s" % (np.round(T_base_cam[:3, 3], 4), camera.CAM_POS))
        print("object: est (%.4f, %.4f) yaw %.1f   true (%.4f, %.4f) yaw %.1f" % (
            *out["obj_xy"], np.rad2deg(out["obj_yaw"]), T_obj[0, 3], T_obj[1, 3], np.rad2deg(yaw_of(T_obj))))
    return out


def pick_roll(yaw, q1):
    """the cube looks the same every 90 deg, pick the finger heading that needs the least wrist roll"""
    cands = yaw + np.arange(4) * np.pi / 2
    k = np.argmin(np.abs(arm.wrap(cands - q1)))
    return arm.wrap(cands[k])


def clik_line(q0, p_from, p_to, pitch, v=V_LIN, K=6.0):
    """move the tip in a straight line from p_from to p_to using CLIK at fixed pitch.
    returns q (N,5), the desired positions and the tracking error norms"""
    d = p_to - p_from
    T = np.linalg.norm(d) / v
    n = int(np.ceil(T / DT))
    qs, pdes, errs = [], [], []
    q = q0.copy()
    for k in range(n + 1):
        s = min(k * DT / T, 1.0)
        p_d = p_from + s * d
        v_d = d / T if k < n else np.zeros(3)
        q, e = arm.clik_step(q, p_d, v_d, pitch, DT, K=K)
        qs.append(q.copy()); pdes.append(p_d); errs.append(np.linalg.norm(e[:3]))
    return np.array(qs), np.array(pdes), np.array(errs)


def plan(perc, q_start=arm.Q_HOME, verbose=True):
    """returns a list of segments. each is a dict with q (N,5), grip (N,), carrying (bool), name"""
    ox, oy = perc["obj_xy"]
    px, py = perc["place_xy"]
    q1_obj = np.arctan2(oy, ox)
    q1_pl = np.arctan2(py, px)
    roll_pick = pick_roll(perc["obj_yaw"], q1_obj)
    roll_place = pick_roll(perc["place_yaw"], q1_pl)

    p_grasp = np.array([ox, oy, GRASP_Z])
    p_hover = p_grasp + [0, 0, HOVER]
    p_drop = np.array([px, py, GRASP_Z + PLACE_CLEAR])
    p_hover2 = p_drop + [0, 0, HOVER]

    q_hover = arm.ik(*p_hover, roll=roll_pick)
    q_hover2 = arm.ik(*p_hover2, roll=roll_place)
    if q_hover is None or q_hover2 is None or not arm.in_limits(q_hover) or not arm.in_limits(q_hover2):
        if verbose:
            print("target not reachable")
        return None

    segs = []

    def joint_move(qa, qb, grip, carrying, name):
        _, q, _, _, _ = tr.plan_through([qa, qb], vmax=VMAX, dt=DT)
        segs.append(dict(q=q, grip=np.full(len(q), grip), carrying=carrying, name=name))
        return q[-1]

    def line_move(qa, pa, pb, grip, carrying, name):
        q, pdes, err = clik_line(qa, pa, pb, -np.pi / 2)
        segs.append(dict(q=q, grip=np.full(len(q), grip), carrying=carrying, name=name, pdes=pdes, err=err))
        return q[-1]

    def dwell(qa, grip_from, grip_to, carrying, name, T=0.6):
        n = int(T / DT)
        g = np.linspace(grip_from, grip_to, n)
        segs.append(dict(q=np.tile(qa, (n, 1)), grip=g, carrying=carrying, name=name))
        return qa

    q = joint_move(q_start, q_hover, 1.0, False, "to hover (cubic)")
    q = line_move(q, p_hover, p_grasp, 1.0, False, "descend (CLIK)")
    q = dwell(q, 1.0, 0.0, False, "close gripper")
    q = line_move(q, p_grasp, p_hover, 0.0, True, "lift (CLIK)")
    q = joint_move(q, q_hover2, 0.0, True, "to place (cubic)")
    q = line_move(q, p_hover2, p_drop, 0.0, True, "lower (CLIK)")
    q = dwell(q, 0.0, 1.0, False, "open gripper")
    q = line_move(q, p_drop, p_hover2, 1.0, False, "retreat (CLIK)")
    q = joint_move(q, arm.Q_HOME, 1.0, False, "home (cubic)")
    return segs


def simulate(segs, T_obj_true):
    """stitch the segments and work out where the cube is at each step.
    the cube rides with the gripper from the moment it closes (we assume the grasp holds
    as long as the gripper centre is within the opening of the fingers)."""
    q = np.vstack([s["q"] for s in segs])
    grip = np.concatenate([s["grip"] for s in segs])
    carrying = np.concatenate([np.full(len(s["q"]), s["carrying"]) for s in segs])
    t = np.arange(len(q)) * DT
    tool = np.array([arm.fk(qi)[:3, 3] for qi in q])
    manip = np.array([arm.manipulability(qi) for qi in q])

    T_obj = np.tile(T_obj_true, (len(q), 1, 1))
    T_tool_obj = None
    for k in range(len(q)):
        if carrying[k]:
            if T_tool_obj is None:
                T_tool_obj = inv(arm.fk(q[k])) @ T_obj_true
            T_obj[k] = arm.fk(q[k]) @ T_tool_obj
        elif T_tool_obj is not None:
            T_obj[k] = T_obj[k - 1]       # released, stays where it was dropped
            T_obj[k][2, 3] = CUBE / 2     # and settles on the table
    seg_bounds = np.cumsum([0] + [len(s["q"]) for s in segs])
    return dict(t=t, q=q, grip=grip, carrying=carrying, tool=tool, manip=manip, T_obj=T_obj,
                seg_bounds=seg_bounds, names=[s["name"] for s in segs])


def grasp_offset(perc, T_obj_true):
    """how far off centre the gripper will close on the cube, mm, and the yaw misalignment in deg"""
    dxy = np.linalg.norm(perc["obj_xy"] - T_obj_true[:2, 3]) * 1000
    dyaw = arm.wrap(perc["obj_yaw"] - yaw_of(T_obj_true))
    dyaw = np.rad2deg(arm.wrap(dyaw + np.pi / 4) - np.pi / 4)     # modulo 90 deg symmetry
    return dxy, abs(dyaw)


def plot_results(sim, perc, T_obj_true, T_place, prefix="results/run"):
    t, q = sim["t"], sim["q"]
    names = ["q1 base", "q2 shoulder", "q3 elbow", "q4 wrist pitch", "q5 wrist roll"]

    fig, (axn, *axs) = plt.subplots(3, 1, figsize=(9, 6.5), sharex=True,
                                    gridspec_kw=dict(height_ratios=[0.5, 3, 3]))
    axn.axis("off")
    for i in range(5):
        axs[0].plot(t, np.rad2deg(q[:, i]), label=names[i])
    for tb in sim["seg_bounds"][1:-1] * DT:
        axs[0].axvline(tb, color="0.8", lw=0.8)
        axs[1].axvline(tb, color="0.8", lw=0.8)
    axs[0].set_ylabel("joint angle (deg)")
    axs[0].legend(fontsize=8, ncol=3, loc="center right")
    axs[0].grid(alpha=0.3)
    axs[1].plot(t, sim["tool"][:, 0] * 100, label="x")
    axs[1].plot(t, sim["tool"][:, 1] * 100, label="y")
    axs[1].plot(t, sim["tool"][:, 2] * 100, label="z")
    axs[1].plot(t, sim["grip"] * 5, "k--", lw=1, label="gripper (open=5)")
    axs[1].set_ylabel("tool position (cm)")
    axs[1].set_xlabel("time (s)")
    axs[1].legend(fontsize=8, ncol=4)
    axs[1].grid(alpha=0.3)
    # segment names along the top
    for i, (a, b, nm) in enumerate(zip(sim["seg_bounds"][:-1], sim["seg_bounds"][1:], sim["names"])):
        axn.text((a + b) / 2 * DT, 0.5 if i % 2 == 0 else 0.0, nm, fontsize=7, ha="center", va="center")
    fig.suptitle("pick and place run: joint angles and tool position")
    fig.tight_layout()
    fig.savefig(prefix + "_joints.png", dpi=130)
    plt.close(fig)

    # top view of the tool path, shows that the cubic segments are curved in cartesian space
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(sim["tool"][:, 0], sim["tool"][:, 1], "b-", lw=1.5, label="tool path")
    ax.plot(*T_obj_true[:2, 3], "s", color="#d9a441", ms=10, label="cube (true)")
    ax.plot(*perc["obj_xy"], "r+", ms=12, label="cube (estimated)")
    ax.plot(*T_place[:2, 3], "gx", ms=10, label="place tag")
    ax.plot(0, 0, "ko", label="arm base")
    ax.plot(*T_BASE_REF[:2, 3], "k^", label="reference tag")
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.legend(fontsize=8)
    ax.set_title("tool path, top view")
    fig.tight_layout()
    fig.savefig(prefix + "_topview.png", dpi=130)
    plt.close(fig)

    # manipulability along the way
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(t, sim["manip"])
    ax.set_xlabel("time (s)")
    ax.set_ylabel("w = sqrt(det(J J^T))")
    ax.set_title("manipulability during the run")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(prefix + "_manip.png", dpi=130)
    plt.close(fig)

    cv2.imwrite(prefix + "_camera.png", td.draw(perc["img"], perc["dets"], perc["poses_cam"], TAG_SIZE))


def animate(sim, perc, T_place, fname="results/pick_place.mp4", every=2, fps=25):
    import imageio_ffmpeg
    matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()

    fig = plt.figure(figsize=(11, 6))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    ax2 = fig.add_subplot(1, 2, 2)
    viz.setup_axes(ax)
    viz.draw_table(ax)
    viz.draw_camera(ax, perc["T_base_cam"], camera.K, camera.W, camera.H)
    viz.draw_tag(ax, T_BASE_REF, TAG_SIZE, "0.3", "ref tag")
    viz.draw_tag(ax, T_place, TAG_SIZE, "#99cc99", "place")
    ax.view_init(elev=24, azim=-55)
    cam_img = cv2.cvtColor(td.draw(perc["img"], perc["dets"], perc["poses_cam"], TAG_SIZE), cv2.COLOR_BGR2RGB)
    ax2.imshow(cam_img)
    ax2.set_title("camera view (detections)")
    ax2.axis("off")
    txt = fig.text(0.05, 0.93, "", fontsize=10)

    idx = np.arange(0, len(sim["t"]), every)
    arts = []

    def update(k):
        nonlocal arts
        for a in arts:
            a.remove()
        arts = []
        arts += viz.draw_arm(ax, sim["q"][k], sim["grip"][k])
        arts += viz.draw_cube(ax, sim["T_obj"][k], CUBE)
        arts += viz.draw_path(ax, sim["tool"][: k + 1], color="b", lw=1, alpha=0.6)
        seg = np.searchsorted(sim["seg_bounds"], k, side="right") - 1
        txt.set_text("t = %.2f s    %s" % (sim["t"][k], sim["names"][min(seg, len(sim["names"]) - 1)]))
        return arts

    ani = animation.FuncAnimation(fig, update, frames=idx, interval=1000 / fps, blit=False)
    ani.save(fname, writer=animation.FFMpegWriter(fps=fps, bitrate=1800))
    plt.close(fig)
    print("saved", fname)


def make_gif(mp4, gif, width=880, fps=12):
    """small palette based gif from the mp4 for the readme (pillow's gif writer gave 14 MB files)"""
    import subprocess
    import imageio_ffmpeg
    vf = "fps=%d,scale=%d:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=96[p];[b][p]paletteuse=dither=bayer" % (fps, width)
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", mp4, "-vf", vf, gif], check=True)
    print("saved", gif)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--noise", type=float, default=3.0)
    ap.add_argument("--video", action="store_true", help="also write results/pick_place.mp4 (slow)")
    ap.add_argument("--gif", action="store_true", help="also make a small gif from the video")
    args = ap.parse_args()

    T_obj, T_place = make_scene(args.seed)
    perc = perceive(T_obj, T_place, noise=args.noise, seed=args.seed)
    if not perc["ok"]:
        raise SystemExit("perception failed")
    segs = plan(perc)
    if segs is None:
        raise SystemExit("planning failed")
    sim = simulate(segs, T_obj)
    dxy, dyaw = grasp_offset(perc, T_obj)
    print("total time %.1f s, %d steps" % (sim["t"][-1], len(sim["t"])))
    print("gripper closes %.2f mm off the cube centre, %.2f deg misaligned" % (dxy, dyaw))
    final = sim["T_obj"][-1][:2, 3]
    print("cube ends up %.2f mm from the place tag centre" % (np.linalg.norm(final - T_place[:2, 3]) * 1000))
    for s in segs:
        if "err" in s:
            print("  %-16s max CLIK tracking error %.3f mm" % (s["name"], s["err"].max() * 1000))
    plot_results(sim, perc, T_obj, T_place)
    if args.video or args.gif:
        animate(sim, perc, T_place, "results/pick_place.mp4")
    if args.gif:
        make_gif("results/pick_place.mp4", "results/pick_place.gif")
