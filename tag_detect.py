# tag_detect.py
# apriltag detection + pose using opencv's aruco module (it has the 36h11 family built in)

import numpy as np
import cv2
from camera import DICT, K, DIST, tag_corners
from transforms import make_T, inv


REFINE = {
    "none": cv2.aruco.CORNER_REFINE_NONE,
    "subpix": cv2.aruco.CORNER_REFINE_SUBPIX,
    "apriltag": cv2.aruco.CORNER_REFINE_APRILTAG,
}


def make_detector(refine="apriltag"):
    """corner refinement matters a lot for pose accuracy, see experiments/tag_pose_error.py.
    the apriltag style refinement (fit lines to the edges) was the clear winner."""
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = REFINE[refine]
    return cv2.aruco.ArucoDetector(DICT, params)


_default = make_detector()


def detect(img, detector=None):
    """returns {tag_id: corners (4,2)}"""
    det = detector or _default
    corners, ids, _ = det.detectMarkers(img)
    out = {}
    if ids is None:
        return out
    for c, i in zip(corners, ids.flatten()):
        out[int(i)] = c.reshape(4, 2)
    return out


def estimate_pose(corners, size, K=K, dist=DIST):
    """T_cam_tag from the 4 image corners. planar square target so IPPE_SQUARE is the right solver"""
    ok, rvec, tvec = cv2.solvePnP(tag_corners(size), corners.astype(np.float64), K, dist,
                                  flags=cv2.SOLVEPNP_IPPE_SQUARE)
    if not ok:
        return None
    R, _ = cv2.Rodrigues(rvec)
    return make_T(R, tvec.flatten())


def camera_from_reference_tag(T_cam_ref, T_base_ref):
    """extrinsic calibration from one tag whose base-frame pose we know (taped to the table):
         T_base_cam = T_base_ref * T_ref_cam = T_base_ref * inv(T_cam_ref)"""
    return T_base_ref @ inv(T_cam_ref)


def point_on_plane(corners, T_base_cam, z_plane, K=K):
    """alternative to full PnP for the position: shoot a ray through the tag centre
    and intersect it with the horizontal plane z = z_plane (we know how tall the objects are).
    depth from PnP of a small tag is the noisy part, this sidesteps it."""
    c = corners.mean(axis=0)
    ray_cam = np.linalg.inv(K) @ np.array([c[0], c[1], 1.0])
    R, o = T_base_cam[:3, :3], T_base_cam[:3, 3]
    d = R @ ray_cam                        # ray direction in base frame
    lam = (z_plane - o[2]) / d[2]
    return o + lam * d


def draw(img, dets, poses=None, size=0.04):
    """colour picture with the outlines, ids and (optionally) the tag axes drawn on"""
    out = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for tid, c in dets.items():
        cv2.polylines(out, [np.round(c).astype(np.int32)], True, (0, 200, 0), 2)
        cv2.putText(out, "id %d" % tid, tuple(np.round(c[0]).astype(int) + [0, -6]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        if poses is not None and tid in poses and poses[tid] is not None:
            T = poses[tid]
            rvec, _ = cv2.Rodrigues(T[:3, :3])
            cv2.drawFrameAxes(out, K, DIST, rvec, T[:3, 3], size * 0.8, 2)
    return out
