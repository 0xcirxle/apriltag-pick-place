---
title: "AprilTag based pick and place with a simulated 5-DOF arm"
subtitle: "EEL-683 Introduction to Robotics, term project"
author:
  - Aniruddh Yadav (23115014)
  - Harendra (23115048)
date: November 2026
---

# 1. What we set out to do

Our proposal was an AprilTag based pick and place arm: a camera detects a tag stuck on an object,
estimates its pose, the pose is transformed into the robot base frame, inverse kinematics gives the
joint angles, and the arm picks the object up and puts it somewhere else. We did not manage to get
hardware together this semester, so the project became a simulation of the same pipeline, with a few
things added that we would not have been able to measure on a real arm (ground truth errors, Monte
Carlo over many scenes).

Everything in the chain is implemented by us except the tag detector itself and the PnP solver, for
which we use OpenCV. The camera image is *rendered* from the known tag poses using a pinhole model,
and then that image goes through the normal detector. This matters: because we render the image we
know the true pose of every tag, so every error in the pipeline can be measured instead of guessed.

The pipeline, frame by frame:

```
  camera image  --detect-->  tag corners (pixels)
                --PnP-->     T_cam_tag          for the reference tag, the object tag, the place tag
                --calib-->   T_base_cam = T_base_ref * inv(T_cam_ref)
                --chain-->   T_base_obj = T_base_cam * T_cam_obj      (and the same for the place tag)
                --IK-->      joint angles for hover / grasp / place
                --cubic-->   joint space trajectory between the big waypoints
                --CLIK-->    straight line descent and lift
                --sim-->     joint angles vs time, tool path, animation
```

Frames used throughout: base frame $\{0\}$ at the bottom of the arm (z up), camera frame $\{c\}$ in the
OpenCV convention (z forward, x right, y down in the image), one frame per tag (z out of the tag),
and the tool frame $\{t\}$ at the gripper tip (z along the gripper, x across the fingers).

# 2. The arm

The arm is the layout of the cheap 6 servo desktop arms we were going to build with: base yaw,
shoulder, elbow, wrist pitch, wrist roll, and a two finger gripper. Link lengths are
$L_1 = 0.10$ m (table to shoulder axis), $L_2 = 0.15$ m, $L_3 = 0.15$ m, $L_5 = 0.08$ m (wrist
centre to finger tips).

## 2.1 DH table and forward kinematics

Standard DH as in the lectures: joint $i$ rotates about $z_{i-1}$ and
$A_i = R_z(\theta_i)\,T_z(d_i)\,T_x(a_i)\,R_x(\alpha_i)$.

| $i$ | $a_i$ | $\alpha_i$ | $d_i$ | $\theta_i$ |
|---|---|---|---|---|
| 1 | 0 | $+90^\circ$ | $L_1$ | $q_1$ |
| 2 | $L_2$ | 0 | 0 | $q_2$ |
| 3 | $L_3$ | 0 | 0 | $q_3$ |
| 4 | 0 | $+90^\circ$ | 0 | $q_4 + 90^\circ$ |
| 5 | 0 | 0 | $L_5$ | $q_5$ |

The $+90^\circ$ offset on $\theta_4$ is the kind of fixed offset mentioned in the notes
($\theta_i = q_i + \theta_{i0}$). Without it, $q_4 = 0$ has the roll axis $z_4$ perpendicular to the
forearm, i.e. the gripper sticking out sideways. With it, $q_4 = 0$ keeps the gripper in line with the
forearm and positive $q_4$ pitches it up, the same sense as $q_2$ and $q_3$.

The tool pose is $T_0^5 = A_1 A_2 A_3 A_4 A_5$ (`arm.fk`). Because joints 2, 3, 4 are parallel the
arm is a base rotation plus a planar 3R chain, so the tip position can also be written down by hand:

$$ r = L_2 c_2 + L_3 c_{23} + L_5 c_{234},\qquad z = L_1 + L_2 s_2 + L_3 s_{23} + L_5 s_{234},\qquad x = r\cos q_1,\; y = r \sin q_1 $$

with $\phi = q_2 + q_3 + q_4$ the pitch of the gripper from horizontal. `experiments/fk_check.py`
compares the DH product with this formula at 1000 random configurations: worst mismatch
$1.9\times10^{-16}$ m, i.e. round off. The home configuration $q = (0, 90, -90, -90, 0)^\circ$ (shoulder
straight up, forearm forward, gripper pointing down) gives the tip at $(0.15, 0, 0.17)$ m which is
easy to verify by eye.

![The arm in four configurations, drawn from the FK (`results/fk_configs.png`).](../results/fk_configs.png)

## 2.2 Workspace

Joint limits were guessed from typical servo travel: $q_1 \in [-150, 150]^\circ$,
$q_2 \in [-10, 190]^\circ$, $q_3 \in [-160, 0]^\circ$ (the elbow only folds one way),
$q_4 \in [-120, 120]^\circ$, $q_5$ free. Sampling 30000 random joint vectors inside the limits gives
the usual spherical shell (left two panels below). The more useful picture is the right panel: for every
point on the table we ask the inverse kinematics whether a *top down* grasp (gripper vertical,
tip at $z = 2.5$ cm) exists inside the limits. That works for $0.052 < r < 0.300$ m from the base axis
and inside the $\pm150^\circ$ base travel. The outer radius is $L_2 + L_3$ with the wrist nearly level
with the shoulder; the inner hole is the forearm folding onto the upper arm.

![Workspace: random sampling (side and top view) and the top down grasp map (`results/workspace.png`).](../results/workspace.png)

# 3. Inverse kinematics

For a pick and place we always want the gripper pointing straight down, so the task is: tip position
$(x, y, z)$, gripper pitch $\phi$ ($= -90^\circ$), and the heading $\psi$ of the fingers in the
horizontal plane so that they close across the faces of the cube. That is 5 numbers for 5 joints, and
the arm decouples nicely (`arm.ik`):

1. Base yaw: $q_1 = \operatorname{atan2}(y, x)$, and $r = \sqrt{x^2 + y^2}$.
2. Wrist centre: step back from the tip along the gripper direction,
   $r_w = r - L_5\cos\phi$, $z_w = z - L_1 - L_5 \sin\phi$.
3. Shoulder and elbow: exactly the two link problem from Tutorial 2 Q6, with links $L_2, L_3$ and
   target $(r_w, z_w)$:
   $$ D = \frac{r_w^2 + z_w^2 - L_2^2 - L_3^2}{2 L_2 L_3},\qquad
      q_3 = \operatorname{atan2}(\pm\sqrt{1 - D^2},\, D),\qquad
      q_2 = \operatorname{atan2}(z_w, r_w) - \operatorname{atan2}(L_3 \sin q_3,\; L_2 + L_3 \cos q_3). $$
   $|D| > 1$ means out of reach. The two signs are the two elbow branches; with our sign convention
   elbow up is $q_3 < 0$.
4. Wrist pitch: $q_4 = \phi - q_2 - q_3$.
5. Wrist roll: with the gripper vertical the finger heading works out to $\psi = q_1 - q_5$
   (tool $z$ points down, so positive roll looks clockwise from above), hence $q_5 = q_1 - \psi$.
   We found this sign numerically from the FK and then wrote the test for it.

Checks (`experiments/ik_check.py`, `tests/test_arm.py`): 5000 random configurations were pushed
through FK and back through IK; the position error of the round trip is at most $3\times10^{-16}$ m
for both branches, and one IK call (both branches) takes about 45 $\mu$s. Of the 10000 branch
solutions only 4205 are inside the joint limits, almost all of them elbow up. The figure shows both
branches for one target; the elbow down one pokes through the table, which is the usual reason it is
rejected.

![Both IK branches for the same target (`results/ik_branches.png`).](../results/ik_branches.png)

A cube looks the same every $90^\circ$, so for the grasp we pick, out of the four equivalent finger
headings $\psi_{tag} + k\,90^\circ$, the one needing the least wrist roll from the current base angle.

# 4. Jacobian, manipulability and CLIK

The geometric Jacobian is built column by column with the rule from the notes, for revolute joints
$J_{v,i} = z_{i-1}\times(p_e - p_{i-1})$ and $J_{\omega,i} = z_{i-1}$, with $z_{i-1}$ and $p_{i-1}$
read off the partial DH products (`arm.jacobian`). Against a finite difference Jacobian the largest
difference over 500 random configurations is $1.9\times10^{-7}$, which is the finite difference step.
The fifth column of the position part is zero because the roll axis passes through the tool tip.
With the arm stretched straight out the position Jacobian drops to rank 2 and the manipulability
$w = \sqrt{\det(J_p J_p^T)}$ is 0, the expected boundary singularity. For the top down grasp pose, $w$
over the table is shown below; it is largest near the outer edge where the elbow is open and smallest
near the base where the arm is folded up.

![Manipulability of the top down grasp configuration over the table (`results/manipulability_map.png`).](../results/manipulability_map.png)

Closed loop inverse kinematics is used for the parts of the motion where the tip must move in a
straight line (the final descent onto the cube and the lift). The task is $x = (p_x, p_y, p_z, \phi)$
and joints 1 to 4 (the roll does not move the tip), so the analytical Jacobian is square:

$$ J_A = \begin{bmatrix} J_p[:, 1{:}4] \\ 0\;\;1\;\;1\;\;1 \end{bmatrix},\qquad
   \dot q = J_A^{-1}\left(\dot x_d + K e\right),\qquad e = x_d - x, $$

integrated with $q_{k+1} = q_k + \dot q\,\Delta t$ (`arm.clik_step`). The lecture result
$\dot e + K e = 0 \Rightarrow e(t) = e(0) e^{-Kt}$ is reproduced in the figure: from the home pose to
a fixed target the error falls along the dashed $e^{-Kt}$ lines for $K = 2, 5, 10$ s$^{-1}$ until
numerical noise around $10^{-10}$ mm.

![CLIK error decay vs the $e^{-Kt}$ prediction (`results/clik_convergence.png`).](../results/clik_convergence.png)

# 5. Camera, AprilTags and coordinate transforms

## 5.1 The fake camera

The camera is a $640\times480$ pinhole with $f = 600$ px and the principal point at the image centre,
no distortion, mounted on a stand at $(0.40, -0.08, 0.40)$ m in the base frame looking at
$(0.18, 0, 0)$. `camera.render` draws each tag (the 36h11 marker image from OpenCV with a one cell
white margin, as a printed tag would have) by projecting its four corners with the camera model and
warping the marker image onto that quadrilateral. The cube under the object tag is drawn too so it can
occlude things. The image is rendered at twice the resolution and shrunk, so edges are anti aliased
like a real photo, then blurred slightly and Gaussian noise is added.

![The rendered camera image for the run in section 7, with detections drawn (`results/run_camera.png`).](../results/run_camera.png)

## 5.2 Detection and pose

OpenCV's `aruco` module ships the AprilTag 36h11 family, so detection is a few lines
(`tag_detect.detect`). Pose comes from `solvePnP` with the `IPPE_SQUARE` solver, which is the right
one for a planar square target; the object points are the four corners of the black square in the
tag frame in the same order the detector returns them. This gives $T^{c}_{tag}$.

One thing we did not expect: the *corner refinement* setting changes the pose accuracy by an order of
magnitude (table in section 7). The default sub pixel refinement pulls all four corners slightly
inwards, the tag looks smaller, and PnP puts it a few mm too far away. The AprilTag style refinement
(fit a line to each edge, intersect) does not have that bias.

Depth of a 4 cm tag from 45 cm is still the weak number: the tag is about 45 px wide, so half a pixel
of corner error is 1% of depth, 5 mm. Since our objects sit on the table and we know how tall they
are, `tag_detect.point_on_plane` instead shoots the ray through the tag centre and intersects it with
the plane $z = h$. Position then only depends on lateral pixel accuracy. The yaw still comes from PnP.

## 5.3 Camera extrinsics from a reference tag

Rather than measure the camera pose with a ruler, a reference tag (id 0) is taped to the table at a
measured spot $T^{0}_{ref}$, and from its detection
$$ T^{0}_{c} = T^{0}_{ref}\,\left(T^{c}_{ref}\right)^{-1}. $$
Then any other tag is $T^{0}_{tag} = T^{0}_{c}\,T^{c}_{tag}$. This is just chaining and inverting
homogeneous transforms (Chapter 2 of the notes); the inverse is done as
$[R\;t]^{-1} = [R^T\;{-R^T t}]$ (`transforms.inv`). In the run below the calibrated camera position is
$(0.4004, -0.0795, 0.4004)$ against the true $(0.40, -0.08, 0.40)$.

The catch, which the Monte Carlo makes very clear, is that any error in $T^{0}_{ref}$ is copied into
every object position: 2 degrees of yaw error on the reference tag at a 15 cm lever arm is already 5 mm.
We also added a sanity check: if the calibrated camera lands more than 5 cm from where it is bolted,
the frame is rejected. This caught the case where the cube happened to sit on top of the reference tag.

# 6. Trajectories and the pick and place sequence

Between the big waypoints (home, hover above the cube, hover above the place tag, home) we use rest
to rest cubics per joint (`trajectory.cubic`), $q(t) = q_0 + \Delta q\,(3u^2 - 2u^3)$, $u = t/T$. The
segment duration comes from the peak speed of a rest to rest cubic, $1.5\,\Delta q / T$ from the notes,
so that the fastest joint stays under $70^\circ$/s. The lecture example (Stanford arm joint, 20° to 60°
in 4 s) is one of the unit tests and reproduces the table in the notes exactly. The quintic is
implemented as well; it removes the acceleration jump at the ends at the cost of a 25% higher peak
velocity ($1.875\,\Delta q/T$ vs $1.5\,\Delta q/T$).

![Cubic vs quintic for the same move (`results/cubic_vs_quintic.png`).](../results/cubic_vs_quintic.png)

A joint space cubic does not move the tool in a straight line. Between the two hover points of a
typical run the tool path bows out by 13 mm (figure below). That is fine while moving through the air,
but the final approach must come straight down onto the cube, so the descent, lift, lower and retreat
segments are done with CLIK along a straight line at 5 cm/s, with the pitch held at $-90^\circ$.
Tracking error on those segments is below 0.03 mm.

![Joint space cubic vs CLIK straight line between the two hover points (`results/joint_vs_cartesian.png`).](../results/joint_vs_cartesian.png)

The complete sequence (`pick_place.plan`):

1. home $\to$ hover 6 cm above the grasp point (cubic)
2. descend to the grasp height, cube mid height (CLIK)
3. close gripper (0.6 s dwell)
4. lift back to hover (CLIK); from here the cube rides with the tool frame
5. hover above pick $\to$ hover above the place tag (cubic)
6. lower to 4 mm above the table (CLIK)
7. open gripper
8. retreat (CLIK)
9. back home (cubic)

# 7. Results

## 7.1 One full run

`python pick_place.py --seed 1` (image noise $\sigma = 3$ grey levels):

| | estimated | true |
|---|---|---|
| camera position (m) | (0.4004, -0.0795, 0.4004) | (0.40, -0.08, 0.40) |
| cube centre xy (m) | (0.1778, 0.0026) | (0.1774, 0.0030) |
| cube yaw | 118.1° | 118.0° |

The gripper closes 0.60 mm from the cube centre with the fingers 0.15° off the faces, and the cube ends
up 0.63 mm from the centre of the place tag. The whole motion takes 9.1 s; CLIK tracking error on the
straight segments is at most 0.025 mm. The joint angles and tool position over time are below; the
vertical lines separate the segments. The video is `results/pick_place.mp4` (gif in the README).

![Joint angles and tool position during the run (`results/run_joints.png`).](../results/run_joints.png)

![Tool path seen from above. The curved parts are the joint space cubics, the straight bits in and out of the pick and place points are CLIK (`results/run_topview.png`).](../results/run_topview.png)

## 7.2 Tag pose accuracy

`experiments/tag_pose_error.py`: 40 random object poses per noise level, errors measured against the
rendered ground truth, all in the base frame after the reference tag calibration.

| image noise $\sigma$ | PnP position (mm) | ray/plane position (mm) | yaw (deg) | camera position (mm) |
|---|---|---|---|---|
| 0 | 0.66 | 0.69 | 0.08 | 0.54 |
| 5 | 0.79 | 0.65 | 0.11 | 1.08 |
| 10 | 1.05 | 0.73 | 0.11 | 1.67 |
| 20 | 1.33 | 0.74 | 0.14 | 2.09 |
| 40 | 1.80 | 1.08 | 0.20 | 2.93 |

Detection rate was 100% at every level (an 8 bit camera with $\sigma = 40$ is a very bad image, see
`results/tag_noisy_example.png`). The ray/plane position is a bit better than PnP and degrades more
slowly, as expected since it ignores the noisy depth.

![Position and yaw error vs image noise (`results/tag_pose_error.png`).](../results/tag_pose_error.png)

Corner refinement comparison at $\sigma = 5$ (same 40 scenes):

| refinement | tag position in camera frame (mm) | object position in base frame (mm) | yaw (deg) |
|---|---|---|---|
| none | 7.03 | 4.58 | 0.48 |
| sub pixel (OpenCV default) | 3.05 | 2.23 | 0.27 |
| AprilTag (line fit) | 0.62 | 0.79 | 0.11 |

## 7.3 Monte Carlo over scenes and calibration errors

`experiments/monte_carlo.py` runs perception and planning (no animation) on 40 random scenes per
condition and counts a grasp as successful if the gripper centre is within 5 mm of the cube centre
(6 cm opening on a 5 cm cube) and the fingers within 10° of the faces. Besides image noise we
also simulated the two calibration errors a real setup would have: the reference tag not being exactly
where we measured it, and the focal length being off.

| condition | tags found | successful grasps | gripper offset mean / max (mm) | yaw error mean (deg) |
|---|---|---|---|---|
| noise 2 | 36 / 40 | 36 | 0.65 / 1.08 | 0.08 |
| noise 10 | 36 / 40 | 36 | 0.63 / 1.19 | 0.11 |
| noise 30 | 36 / 40 | 36 | 0.82 / 1.69 | 0.18 |
| ref tag off by 3 mm, 1° | 36 / 40 | 36 | 1.39 / 2.98 | 1.05 |
| ref tag off by 5 mm, 2° | 36 / 40 | **4** | 7.01 / 8.87 | 1.98 |
| focal length 2% off | 36 / 40 | 36 | 0.61 / 1.21 | 0.14 |
| focal length 5% off | 36 / 40 | 36 | 0.49 / 1.15 | 0.30 |
| ref 3 mm/1° and focal 2% | 36 / 40 | 36 | 1.30 / 3.14 | 1.12 |

![Monte Carlo summary (`results/monte_carlo.png`).](../results/monte_carlo.png)

Three things stand out. Image noise is almost irrelevant once the corners are refined properly.
A focal length error barely matters for the ray/plane position because the same wrong $K$ is used
for the calibration and for the object, and the errors largely cancel (with plain PnP it would be
about twice as bad, 0.95 mm at 5%). The reference tag placement is what decides success: 5 mm and 2°
of error on it moves every object estimate by about 7 mm and the grasp fails in 36 of 40 scenes.
On a real build this is where the effort should go (a bigger reference tag, several of them, or a proper
hand eye calibration). The 4 scenes with missing tags in every row are the same 4: the cube lands
between the camera and the place tag or on top of the reference tag and occludes it. A second camera
or simply nudging the object would fix that.

# 8. Limitations and what we would do next

* No dynamics. Joints follow the planned angles exactly; there is no servo lag, no gravity sag, no
  gripper compliance. The "grasp" is a geometric test, not a contact model.
* The camera is ideal apart from blur and noise: no lens distortion, no lighting gradients, no motion
  blur, and the tag print is assumed to be exactly 4 cm.
* The table is not treated as an obstacle in the workspace sampling (the side view shows the arm
  reaching below $z = 0$); only the grasp map respects it.
* CLIK has no joint velocity limit; with a large initial error $K e$ could demand an impossible speed.
* Things we wanted to try: lens distortion in the renderer, a 1280x720 camera with a 3 cm tag,
  hand eye calibration from several reference tags instead of one, and of course the real arm.

# 9. Reproducing the figures

```
pip install -r requirements.txt
python tests/test_arm.py
python pick_place.py --seed 1 --video --gif
python experiments/fk_check.py
python experiments/workspace.py
python experiments/ik_check.py
python experiments/jacobian_check.py
python experiments/tag_pose_error.py
python experiments/traj_compare.py
python experiments/monte_carlo.py
```

# References

1. EEL-683 lecture notes and tutorials 1 to 4 (frames, DH, inverse position, Jacobians, cubic trajectories, CLIK).
2. B. Siciliano, L. Sciavicco, L. Villani, G. Oriolo, *Robotics: Modelling, Planning and Control*, Springer, 2009.
3. J. J. Craig, *Introduction to Robotics: Mechanics and Control*, 3rd ed., Pearson, 2005.
4. E. Olson, "AprilTag: A robust and flexible visual fiducial system", ICRA 2011.
5. J. Wang, E. Olson, "AprilTag 2: Efficient and robust fiducial detection", IROS 2016.
6. T. Collins, A. Bartoli, "Infinitesimal plane-based pose estimation", IJCV 2014 (the IPPE solver used by OpenCV).
7. OpenCV documentation, `cv::aruco` module and `cv::solvePnP`.
