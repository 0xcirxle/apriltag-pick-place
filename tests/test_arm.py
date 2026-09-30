# quick sanity tests, run with: python -m pytest tests  (or just python tests/test_arm.py)
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import arm
import trajectory as tr


def test_fk_home():
    T = arm.fk(arm.Q_HOME)
    # shoulder straight up, forearm forward, gripper down
    np.testing.assert_allclose(T[:3, 3], [arm.L3, 0, arm.L1 + arm.L2 - arm.L5], atol=1e-9)
    np.testing.assert_allclose(T[:3, 2], [0, 0, -1], atol=1e-9)


def test_fk_zero():
    # all zeros: whole arm stretched out along +x at shoulder height
    T = arm.fk(np.zeros(5))
    np.testing.assert_allclose(T[:3, 3], [arm.L2 + arm.L3 + arm.L5, 0, arm.L1], atol=1e-9)


def test_ik_roundtrip():
    rng = np.random.default_rng(0)
    n_ok = 0
    for _ in range(300):
        q = rng.uniform(arm.LIMITS[:, 0], arm.LIMITS[:, 1])
        p, pitch, roll = arm.tool_pose(q)
        for elbow in ("up", "down"):
            qs = arm.ik(*p, pitch=pitch, roll=roll, elbow=elbow)
            if qs is None:
                continue
            p2, pitch2, roll2 = arm.tool_pose(qs)
            np.testing.assert_allclose(p2, p, atol=1e-6)
            assert abs(arm.wrap(pitch2 - pitch)) < 1e-6
            n_ok += 1
    assert n_ok > 300   # both branches exist for most points


def test_ik_topdown_roll():
    # roll only makes sense when pointing straight down. check it there.
    for roll in np.deg2rad([0, 30, -60, 120]):
        q = arm.ik(0.18, 0.05, 0.03, pitch=-np.pi/2, roll=roll)
        assert q is not None
        p, pitch, r2 = arm.tool_pose(q)
        np.testing.assert_allclose(p, [0.18, 0.05, 0.03], atol=1e-9)
        assert abs(arm.wrap(r2 - roll)) < 1e-9


def test_ik_unreachable():
    assert arm.ik(1.0, 0, 0) is None
    assert arm.ik(0.0, 0, 0.9) is None


def test_jacobian_matches_numeric():
    rng = np.random.default_rng(1)
    for _ in range(20):
        q = rng.uniform(-1.5, 1.5, 5)
        Ja = arm.jacobian(q)[:3]
        Jn = arm.jacobian_numeric(q)
        np.testing.assert_allclose(Ja, Jn, atol=1e-5)


def test_cubic_lecture_example():
    # stanford arm joint, 20 -> 60 deg in 4 s. table from the notes
    t = np.array([0, 1, 2, 3, 4.0])
    q, qd, qdd = tr.cubic(20.0, 60.0, 4.0, t)
    np.testing.assert_allclose(q[:, 0], [20, 26.25, 40, 53.75, 60])
    np.testing.assert_allclose(qd[:, 0], [0, 11.25, 15, 11.25, 0])
    np.testing.assert_allclose(qdd[:, 0], [15, 7.5, 0, -7.5, -15])


def test_quintic_ends():
    q, qd, qdd = tr.quintic([0, 1], [1, 3], 2.0, [0, 2.0])
    np.testing.assert_allclose(q, [[0, 1], [1, 3]], atol=1e-12)
    np.testing.assert_allclose(qd, 0, atol=1e-12)
    np.testing.assert_allclose(qdd, 0, atol=1e-12)


if __name__ == "__main__":
    for name, f in list(globals().items()):
        if name.startswith("test_"):
            f()
            print("ok", name)
