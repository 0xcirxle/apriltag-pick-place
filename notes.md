# working notes

decisions, gotchas and todo, so we remember why the code is the way it is.

## the arm
- 5 dof: base yaw, shoulder, elbow, wrist pitch, wrist roll, plus a 2 finger gripper. same layout as the
  cheap 6 servo desktop arms, which is what we would have built if we had got to hardware.
- link lengths 10 / 15 / 15 / 8 cm. reach is about 30 cm from the shoulder axis.
- DH: standard convention as done in class (joint i about z_{i-1}, A = Rz Tz Tx Rx). needed a +90 deg
  offset on theta_4 so that q4 = 0 keeps the gripper in line with the forearm, otherwise the zero pose
  has the gripper sticking out sideways. took a while to see why.
- sign convention: q2, q3, q4 positive = "up" (counter clockwise in the vertical plane through the arm).
  elbow up therefore means q3 < 0.
- IK: base yaw from atan2(y, x); step back from the tip to the wrist centre along the gripper direction;
  2-link formula from tutorial 2 Q6 for shoulder/elbow (both branches via +-sqrt(1-D^2)); q4 = pitch - q2 - q3.
  roll: q5 = q1 - roll. found numerically that the finger heading is q1 - q5 when pointing down
  (tool z points down so positive roll looks clockwise from above).
- the elbow down branch is almost never inside the joint limits for a top-down grasp, so elbow up is default.
- checked FK against the hand formula r = L2 c2 + L3 c23 + L5 c234, agree to 1e-16.

## camera / tags
- no real camera, so the tags are rendered into a fake image with the pinhole model and that image goes
  through the normal detector. ground truth is then known exactly and errors can be measured properly.
- opencv's aruco module has the apriltag 36h11 family built in, no extra library needed.
- big lesson: the corner refinement method changes pose accuracy by ~10x. CORNER_REFINE_SUBPIX pulls the
  corners slightly inward -> tag looks smaller -> depth overestimated by a few mm. CORNER_REFINE_APRILTAG
  fits lines to the edges and is far better. numbers in results/corner_refinement.csv.
- depth from pnp of a 4 cm tag at 0.5 m is the weak part: tag is ~45 px wide, 0.5 px error = 1 % of depth
  = 5 mm. since the objects sit on the table we intersect the ray through the tag centre with the plane
  z = cube height instead (tag_detect.point_on_plane). position then only depends on lateral pixel accuracy.
- extrinsics: one reference tag taped at a measured spot, T_base_cam = T_base_ref * inv(T_cam_ref).
  any error in where we think the ref tag is goes straight into every object position. 2 deg of yaw error
  on the ref tag is already ~5 mm at a 15 cm lever arm. this is the dominant error in the monte carlo runs,
  not image noise.
- rendering: supersample 2x then shrink so the edges are anti-aliased, otherwise the corner refinement
  behaves differently from a real photo. tags drawn far to near so a close one covers a far one.
- had one scene where the cube sat on the reference tag -> calibration garbage (camera 60 cm off).
  added a sanity check: calibrated camera more than 5 cm from where it is bolted -> reject the frame.
- 4 of 40 random scenes lose a tag to occlusion (cube between camera and place tag, or on top of the ref tag).
  a second camera or just moving the object would fix it; we count them as failures.

## trajectories
- cubic rest-to-rest between joint waypoints. segment time from the 1.5*dq/T peak velocity rule so no
  joint goes above 70 deg/s.
- a joint space cubic between the two hover points curves by ~13 mm in xy. fine for moving through the
  air, but for the final approach the gripper should come straight down -> CLIK on (x, y, z, pitch)
  with the 4x4 analytical jacobian (position rows of the geometric J for joints 1-4, plus the row
  [0 1 1 1] for pitch = q2+q3+q4). tracking error < 0.03 mm at 5 cm/s.
- CLIK error decays as e^{-Kt} like in the notes (results/clik_convergence.png). K = 10 with dt = 0.01
  starts to show numerical noise around 1e-10 mm, obviously irrelevant.
- quintic implemented too but not used in the run, the cubic jerk at the ends is not a problem in simulation.

## todo / ideas
- [ ] joint velocity limits inside CLIK (K*e could ask for anything if the error were large)
- [ ] collision check of the fingers against the cube on the way down (we come straight from above, so ok for now)
- [ ] lens distortion in the fake camera
- [ ] try a 3 cm tag and a 1280x720 camera
- [ ] gripper force / slip model instead of "within 5 mm = grasped"
- [ ] actual hardware
