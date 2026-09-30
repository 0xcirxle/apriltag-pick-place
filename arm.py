# arm.py
# kinematics of our 5 dof arm: base yaw, shoulder, elbow, wrist pitch, wrist roll
# everything uses the standard DH convention from class,
#   A_i = Rz(theta_i) Tz(d_i) Tx(a_i) Rx(alpha_i)

import numpy as np

# link lengths (m). roughly a desktop hobby arm
L1 = 0.10   # table to shoulder axis
L2 = 0.15   # upper arm
L3 = 0.15   # forearm
L5 = 0.08   # wrist centre to gripper tip (tool length)

# joint limits in deg, guessed from typical servo travel
LIMITS_DEG = np.array([
    [-150, 150],   # q1 base yaw
    [-10, 190],    # q2 shoulder (0 = horizontal, +ve = up)
    [-160, 0],     # q3 elbow (only folds one way)
    [-120, 120],   # q4 wrist pitch
    [-180, 180],   # q5 wrist roll
])
LIMITS = np.deg2rad(LIMITS_DEG)

# DH table, one row per joint: (a, alpha, d, theta_offset)
# theta_i = q_i + theta_offset
DH = [
    (0.0, np.pi/2, L1, 0.0),
    (L2,  0.0,     0.0, 0.0),
    (L3,  0.0,     0.0, 0.0),
    (0.0, np.pi/2, 0.0, np.pi/2),   # offset so that q4 = 0 keeps gripper in line with forearm
    (0.0, 0.0,     L5,  0.0),
]

Q_HOME = np.deg2rad([0, 90, -90, -90, 0])   # folded up, gripper pointing down


def dh_matrix(a, alpha, d, theta):
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([
        [ct, -st * ca,  st * sa, a * ct],
        [st,  ct * ca, -ct * sa, a * st],
        [0.0, sa,       ca,      d],
        [0.0, 0.0,      0.0,     1.0],
    ])


def fk_all(q):
    """list of T_0_i for i = 0..5  (T_0_0 is identity)"""
    T = np.eye(4)
    out = [T]
    for (a, alpha, d, off), qi in zip(DH, q):
        T = T @ dh_matrix(a, alpha, d, qi + off)
        out.append(T)
    return out


def fk(q):
    """T_0_tool  (4x4)"""
    return fk_all(q)[-1]


def tool_pose(q):
    """position (3,), pitch, roll(=yaw of fingers about world z) of the tool.
    pitch is the angle of the gripper axis from horizontal, -90 deg = pointing down"""
    T = fk(q)
    p = T[:3, 3]
    zt = T[:3, 2]                       # gripper axis
    pitch = np.arctan2(zt[2], np.hypot(zt[0], zt[1]))
    # finger direction = tool x axis. its heading in the xy plane
    xt = T[:3, 0]
    roll = np.arctan2(xt[1], xt[0])
    return p, pitch, roll


def joint_positions(q):
    """xyz of base, shoulder, elbow, wrist, tool tip  (5x3). for drawing"""
    Ts = fk_all(q)
    idx = [0, 1, 2, 3, 5]      # frame 4 sits on the wrist as well, skip it
    return np.array([Ts[i][:3, 3] for i in idx])


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def in_limits(q):
    q = wrap(np.asarray(q))
    return np.all(q >= LIMITS[:, 0]) and np.all(q <= LIMITS[:, 1])


def ik(x, y, z, pitch=-np.pi/2, roll=0.0, elbow="up"):
    """closed form inverse kinematics.
    (x,y,z): where the gripper tip should go, in base frame
    pitch:   gripper axis angle from horizontal (use -90 deg for top down grasp)
    roll:    heading of the fingers in the xy plane (only meaningful when pitch = -90)
    elbow:   'up' or 'down' branch
    returns q (5,) or None if not reachable"""
    q1 = np.arctan2(y, x)
    r = np.hypot(x, y)

    # step back from the tip to the wrist centre along the gripper direction
    rw = r - L5 * np.cos(pitch)
    zw = z - L1 - L5 * np.sin(pitch)

    # planar 2 link problem, exactly like tutorial 2 Q6
    D = (rw**2 + zw**2 - L2**2 - L3**2) / (2 * L2 * L3)
    if abs(D) > 1.0:
        return None
    s3 = np.sqrt(1 - D**2)
    if elbow == "up":
        s3 = -s3          # elbow above the shoulder-wrist line -> negative q3 with our sign convention
    q3 = np.arctan2(s3, D)
    q2 = np.arctan2(zw, rw) - np.arctan2(L3 * np.sin(q3), L2 + L3 * np.cos(q3))
    q4 = pitch - q2 - q3

    # roll: with the tool pointing down the finger heading works out to q1 - q5
    # (tool z points down so +ve q5 is clockwise seen from above). checked numerically in tests.
    q5 = q1 - roll

    q = wrap(np.array([q1, q2, q3, q4, q5]))
    return q


def jacobian(q):
    """geometric jacobian 6x5. column rule for revolute joints:
       Jv_i = z_{i-1} x (p_tool - p_{i-1}),   Jw_i = z_{i-1}"""
    Ts = fk_all(q)
    pe = Ts[-1][:3, 3]
    J = np.zeros((6, 5))
    for i in range(5):
        z = Ts[i][:3, 2]
        p = Ts[i][:3, 3]
        J[:3, i] = np.cross(z, pe - p)
        J[3:, i] = z
    return J


def jacobian_numeric(q, h=1e-6):
    """position jacobian by finite differences, just to check the analytic one"""
    q = np.asarray(q, float)
    J = np.zeros((3, 5))
    p0 = fk(q)[:3, 3]
    for i in range(5):
        dq = np.zeros(5)
        dq[i] = h
        J[:, i] = (fk(q + dq)[:3, 3] - p0) / h
    return J


def manipulability(q):
    Jp = jacobian(q)[:3]
    return np.sqrt(max(np.linalg.det(Jp @ Jp.T), 0.0))   # det can come out -1e-20 at a singularity


def clik_step(q, p_des, v_des, pitch_des, dt, K=5.0):
    """one step of closed loop inverse kinematics.
    task = (x, y, z, pitch). joints 1..4 only, q5 does not move the tip anyway.
    qdot = J^-1 (xdot_d + K e),  then euler integrate."""
    p, pitch, _ = tool_pose(q)
    e = np.r_[p_des - p, pitch_des - pitch]
    xdot = np.r_[v_des, 0.0] + K * e
    Jv = jacobian(q)[:3, :4]
    Ja = np.vstack([Jv, [0, 1, 1, 1]])   # pitch = q2 + q3 + q4
    qdot = np.linalg.solve(Ja, xdot)
    qn = q.copy()
    qn[:4] += qdot * dt
    return qn, e


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    print("home config", np.rad2deg(Q_HOME))
    print(fk(Q_HOME))
    p, pitch, roll = tool_pose(Q_HOME)
    print("tip", p, "pitch", np.rad2deg(pitch), "roll", np.rad2deg(roll))
