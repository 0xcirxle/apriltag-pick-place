# camera.py
# a fake pinhole camera. we don't have the physical setup so the camera image is
# rendered from the known tag poses, then fed to the normal detector. this lets us
# check the whole pipeline (detect -> pose -> transform to base -> IK) against ground truth.

import numpy as np
import cv2
from transforms import make_T, inv, apply

W, H = 640, 480
# intrinsics, similar to a cheap 640x480 webcam
K = np.array([[600.0, 0.0, 320.0],
              [0.0, 600.0, 240.0],
              [0.0,   0.0,   1.0]])
DIST = np.zeros(5)      # assume calibrated / undistorted

DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)

# where the camera is bolted, in the base frame. on a stand in front of the arm, looking down at the table
CAM_POS = np.array([0.40, -0.08, 0.40])
CAM_LOOK = np.array([0.18, 0.00, 0.00])


def look_at(cam_pos, target, up=(0, 0, 1)):
    """T_base_cam for a camera at cam_pos looking at target.
    opencv convention: z forward (optical axis), x right, y down in the image."""
    cam_pos = np.asarray(cam_pos, float)
    z = np.asarray(target, float) - cam_pos
    z /= np.linalg.norm(z)
    x = np.cross(z, up)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.column_stack([x, y, z])
    return make_T(R, cam_pos)


T_BASE_CAM_TRUE = look_at(CAM_POS, CAM_LOOK)


def tag_corners(size):
    """the 4 corners of the black square, in the tag's own frame, in the order
    the opencv detector returns them: top-left, top-right, bottom-right, bottom-left.
    tag frame: x right, y up, z out of the tag."""
    s = size / 2.0
    return np.array([[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]], float)


def project(pts_base, T_base_cam=T_BASE_CAM_TRUE, K=K):
    """base frame points -> pixel coords"""
    T_cam_base = inv(T_base_cam)
    rvec, _ = cv2.Rodrigues(T_cam_base[:3, :3])
    tvec = T_cam_base[:3, 3]
    uv, _ = cv2.projectPoints(np.asarray(pts_base, float).reshape(-1, 3), rvec, tvec, K, DIST)
    return uv.reshape(-1, 2)


def render(tags, T_base_cam=T_BASE_CAM_TRUE, noise=3.0, blur=0.6, bg=165, seed=None, ss=2, boxes=(), K_true=None):
    """tags: list of (id, size, T_base_tag). returns a grayscale image.
    each tag is drawn with a 1-cell white margin (as a printed tag would have) and
    warped into place with the homography between the tag plane and the image.
    ss = supersampling factor: render ss times bigger and shrink, so the edges are
    anti-aliased like a real photo instead of hard pixel steps."""
    rng = np.random.default_rng(seed)
    Kss = (K if K_true is None else K_true).copy()    # K_true: what the camera really is, if not the nominal K
    Kss[:2] *= ss
    Ws, Hs = W * ss, H * ss
    img = np.full((Hs, Ws), bg, np.uint8)

    # a bit of texture so the picture isn't perfectly flat (table grain)
    img = np.clip(img.astype(float) + rng.normal(0, 2, (Hs, Ws)), 0, 255).astype(np.uint8)

    # boxes = list of (T_centre, side, shade): the objects the tags are stuck on. drawn before the
    # tags, faces painted back to front so the nearer ones cover the farther ones.
    T_cam_base = inv(T_base_cam)
    for T, side, shade in boxes:
        h = side / 2.0
        c = np.array([[x, y, z] for x in (-h, h) for y in (-h, h) for z in (-h, h)])
        pb = apply(T, c)
        pc = apply(T_cam_base, pb)
        uv = project(pb, T_base_cam, Kss)
        faces = [[0, 1, 3, 2], [4, 5, 7, 6], [0, 1, 5, 4], [2, 3, 7, 6], [0, 2, 6, 4], [1, 3, 7, 5]]
        depth = [pc[f, 2].mean() for f in faces]
        for i in np.argsort(depth)[::-1]:
            f = faces[i]
            # crude shading: top face lightest, use the face normal's z component in the base frame
            n = np.cross(pb[f[1]] - pb[f[0]], pb[f[2]] - pb[f[0]])
            n = n / np.linalg.norm(n)
            val = int(np.clip(shade * (0.65 + 0.35 * abs(n[2])), 0, 255))
            cv2.fillConvexPoly(img, np.round(uv[f]).astype(np.int32), val, lineType=cv2.LINE_AA)

    N = 10                         # 36h11 = 6 bits + 1 black border cell = 8 cells, + 1 white cell each side
    cell_px = 12
    # draw the far tags first so a nearer one covers them like it would in a photo
    depth = [apply(T_cam_base, T[:3, 3])[2] for _, _, T in tags]
    for k in np.argsort(depth)[::-1]:
        tid, size, T = tags[k]
        marker = cv2.aruco.generateImageMarker(DICT, int(tid), 8 * cell_px)
        canvas = np.full((N * cell_px, N * cell_px), 255, np.uint8)
        canvas[cell_px:-cell_px, cell_px:-cell_px] = marker

        # canvas edges (incl. the white margin) in the tag frame
        s = size * N / 8.0 / 2.0
        quad_tag = np.array([[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]])
        quad_img = project(apply(T, quad_tag), T_base_cam, Kss).astype(np.float32)
        src = np.array([[0, 0], [N * cell_px, 0], [N * cell_px, N * cell_px], [0, N * cell_px]], np.float32)
        Hm = cv2.getPerspectiveTransform(src, quad_img)
        warped = cv2.warpPerspective(canvas, Hm, (Ws, Hs), flags=cv2.INTER_LINEAR)
        mask = np.zeros((Hs, Ws), np.uint8)
        cv2.fillConvexPoly(mask, np.round(quad_img).astype(np.int32), 255)
        img[mask > 0] = warped[mask > 0]

    if ss > 1:
        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
    if blur > 0:
        img = cv2.GaussianBlur(img, (0, 0), blur)
    if noise > 0:
        img = np.clip(img.astype(float) + rng.normal(0, noise, img.shape), 0, 255).astype(np.uint8)
    return img
