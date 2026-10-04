# trajectory.py
# joint space trajectories. cubic and quintic polynomials from the lecture notes,
# and a helper that strings several cubic segments through waypoints.

import numpy as np


def cubic_coeffs(q0, qf, T, v0=0.0, vf=0.0):
    """a0..a3 for q(t) = a0 + a1 t + a2 t^2 + a3 t^3, with the 4 boundary conditions
    q(0)=q0, q(T)=qf, qdot(0)=v0, qdot(T)=vf.  works elementwise on arrays."""
    q0, qf = np.asarray(q0, float), np.asarray(qf, float)
    a0 = q0
    a1 = v0 * np.ones_like(q0)
    a2 = (3 * (qf - q0) - (2 * v0 + vf) * T) / T**2
    a3 = (-2 * (qf - q0) + (v0 + vf) * T) / T**3
    return a0, a1, a2, a3


def cubic(q0, qf, T, t, v0=0.0, vf=0.0):
    """q, qdot, qddot at time(s) t (t can be an array). shapes: (len(t), n)"""
    a0, a1, a2, a3 = cubic_coeffs(q0, qf, T, v0, vf)
    t = np.atleast_1d(np.asarray(t, float))[:, None]
    q = a0 + a1 * t + a2 * t**2 + a3 * t**3
    qd = a1 + 2 * a2 * t + 3 * a3 * t**2
    qdd = 2 * a2 + 6 * a3 * t
    return q, qd, qdd


def quintic(q0, qf, T, t):
    """rest to rest quintic (zero vel and accel at both ends).
    q = q0 + dq (10 u^3 - 15 u^4 + 6 u^5),  u = t/T"""
    q0, qf = np.asarray(q0, float), np.asarray(qf, float)
    dq = qf - q0
    u = (np.atleast_1d(np.asarray(t, float)) / T)[:, None]
    q = q0 + dq * (10 * u**3 - 15 * u**4 + 6 * u**5)
    qd = dq * (30 * u**2 - 60 * u**3 + 30 * u**4) / T
    qdd = dq * (60 * u - 180 * u**2 + 120 * u**3) / T**2
    return q, qd, qdd


def segment_time(q0, qf, vmax, tmin=0.4):
    """how long a rest to rest cubic needs so that no joint exceeds vmax.
    peak speed of a rest to rest cubic is 1.5*dq/T  (from the notes)"""
    dq = np.max(np.abs(np.asarray(qf) - np.asarray(q0)))
    return max(tmin, 1.5 * dq / vmax)


def plan_through(waypoints, vmax=np.deg2rad(60), dt=0.02, kind="cubic"):
    """rest-to-rest polynomial between consecutive waypoints.
    returns t (N,), q (N,n), qd (N,n), qdd (N,n) and the list of segment end times"""
    waypoints = [np.asarray(w, float) for w in waypoints]
    ts, qs, qds, qdds = [], [], [], []
    t_offset = 0.0
    seg_ends = []
    for a, b in zip(waypoints[:-1], waypoints[1:]):
        T = segment_time(a, b, vmax)
        tt = np.arange(0, T, dt)
        if kind == "cubic":
            q, qd, qdd = cubic(a, b, T, tt)
        else:
            q, qd, qdd = quintic(a, b, T, tt)
        ts.append(tt + t_offset)
        qs.append(q); qds.append(qd); qdds.append(qdd)
        t_offset += T
        seg_ends.append(t_offset)
    # add the final point
    ts.append([t_offset]); qs.append(waypoints[-1][None]); qds.append(np.zeros((1, len(waypoints[-1]))))
    qdds.append(np.zeros((1, len(waypoints[-1]))))
    return (np.concatenate(ts), np.vstack(qs), np.vstack(qds), np.vstack(qdds), seg_ends)


if __name__ == "__main__":
    # the lecture example: 20 deg to 60 deg in 4 s
    t = np.arange(0, 4.01, 1.0)
    q, qd, qdd = cubic(20.0, 60.0, 4.0, t)
    for row in zip(t, q[:, 0], qd[:, 0], qdd[:, 0]):
        print("t=%.0f  q=%.2f  qd=%.2f  qdd=%.2f" % row)
