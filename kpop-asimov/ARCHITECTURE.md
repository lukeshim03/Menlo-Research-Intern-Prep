# Architecture

A complete explanation of how **K-pop Asimov Dance** turns a dance video into
robot motion, from motion capture to MuJoCo playback.

---

## 1. The Big Picture

The system is a **motion-retargeting pipeline**: it takes the movement of a
human dancer in a 2D video and maps it onto a physical humanoid robot model with
a completely different body. Four stages, each a separate module:

```
 ┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
 │  Video / URL │ → │ Motion Capture│ → │  Retargeting │ → │   Playback   │
 │ input_video  │   │ extract_pose │   │  retarget.py │   │ simulate /   │
 │   .mp4       │   │   .py         │   │              │   │ visualize.py │
 └──────────────┘   └──────────────┘   └──────────────┘   └──────────────┘
        │                  │                   │                   │
     yt-dlp           MediaPipe            MuJoCo IK          MuJoCo render
                    33 landmarks         25-DOF solve         + OpenCV
```

The hard part is the middle: a human and the Asimov robot have **different limb
lengths, different proportions, and different joint axes**. You can't just copy
angles. The pipeline solves this with inverse kinematics.

---

## 2. Stage 1 — Motion Capture (`extract_pose.py`)

### What is motion capture here?

Traditional mocap uses physical markers and multi-camera rigs. We do
**markerless monocular mocap**: a neural network estimates a 3D skeleton from a
single ordinary video.

### MediaPipe PoseLandmarker

We use Google's **MediaPipe PoseLandmarker** (Tasks API, `pose_landmarker_full.task`
model, auto-downloaded on first run). For every video frame it returns **33 body
landmarks**, each with:

- **World landmarks** — 3D coordinates in meters, origin at the hip center.
  Used for *pose* (limb directions). Coordinate frame: `x = right`, `y = down`,
  `z = toward camera`.
- **Image landmarks** — normalized `[0,1]` screen coordinates. Used for *global
  stage position* (where on screen the dancer is).

```
        0 nose
   11 ●──────● 12      shoulders
      │      │
   13 ●      ● 14      elbows
      │      │
   15 ●      ● 16      wrists
   23 ●──────● 24      hips
      │      │
   25 ●      ● 26      knees
      │      │
   27 ●      ● 28      ankles
```

Only these 13 indices matter for our robot (arms, legs, torso, head).

### Robustness

- Runs in `RunningMode.VIDEO` for temporal tracking consistency.
- Frames where key landmarks have `visibility < 0.5` reuse the previous frame
  (no flicker / dropouts).
- Output is pickled to `dance.pkl` so capture (the slow step) runs once.

---

## 3. Stage 2 — Retargeting (`retarget.py`) — the core

This is where human motion becomes robot joint angles. Three sub-problems:
**coordinate alignment**, **limb IK**, and **root motion**.

### 3.1 Coordinate alignment

MediaPipe and the robot use different axis conventions. We convert every vector
with one rotation:

```python
# MediaPipe: x=right, y=DOWN, z=toward camera
# Robot:     x=forward, y=left, z=up
def mp_to_robot(v):
    return np.array([-v[2], v[0], -v[1]])
```

Two subtleties that caused real bugs:

- **`y` is flipped** because MediaPipe's `y` points *down*. Without `-v[1]` the
  legs pointed up (the robot looked like it was sitting in the air).
- **`+v[0]` (not `-v[0]`)** keeps the mapping **right-handed** (`det = +1`).
  A left-handed transform mirrors the body, so the robot's left foot copied the
  dancer's right foot.

### 3.2 Limb IK (inverse kinematics)

We never compute "elbow angle = …". Instead, per limb we define a small
**kinematic chain** and let MuJoCo solve it:

```python
LIMBS = {
  "left_arm": {
    "joints": [shoulder_pitch, shoulder_roll, shoulder_yaw, elbow],
    "anchor_body": "left_shoulder_pitch_link",
    "chain": [(shoulder→elbow, "left_elbow_link"),
              (elbow→wrist,   "left_wrist_yaw_link")],
  }, ... right_arm, left_leg, right_leg
}
```

**Building targets** (`build_targets`): walk the chain from the robot's actual
shoulder/hip position, stepping along the *human's* limb direction but by the
*robot's* segment length:

```
target_elbow = robot_shoulder + dir(human shoulder→elbow) · L_robot_upperarm
target_wrist = target_elbow   + dir(human elbow→wrist)    · L_robot_forearm
```

This is **bone-length retargeting**: it preserves *pose* (angles/direction)
while respecting the robot's real dimensions (measured once from the MJCF with
`measure_segment_lengths`, e.g. upper arm 23.5 cm, forearm 11.3 cm).

**Solving** (`solve_limb_ik`): iterative **damped least-squares IK**. Each
iteration:

1. `mj_forward` → current body positions.
2. `error = target − current_position` for elbow and wrist (a 6-vector).
3. `mj_jacBody` → Jacobian `J` (how each joint moves each body).
4. `dq = Jᵀ(JJᵀ + λ²I)⁻¹ · error` (damped pseudo-inverse, λ = 0.15 for stability).
5. Step the joints by `dq` (clamped to ±0.3 rad), clip to MJCF joint limits.

15 iterations per limb per frame. The **previous frame's solution is the warm
start**, which makes the trajectory naturally smooth and fast to converge.

### 3.3 Root motion (`compute_root_motion`)

Limb IK only moves arms/legs relative to the torso. To make the robot *travel
the stage, turn, and crouch* like the dancer, we drive the **pelvis free joint**
(`qpos[0:7]` = position + quaternion):

| Root DOF | Source | Why |
|----------|--------|-----|
| **Side-to-side** (`y`) | image hip x-position | dancer moving across the floor |
| **Vertical** (`z`) | leg compression ratio | crouches/jumps; keeps feet on the ground |
| **Yaw** (quaternion) | hip facing direction | full-body turns |
| **Waist twist** | shoulder yaw − hip yaw | torso twisting over the hips |

Key detail — **start facing the camera**: raw yaw is offset by the median of the
first 10 frames (`yaw_baseline`), so the robot always begins front-on regardless
of which way the dancer happened to start.

### 3.4 Smoothing

Every hinge joint trajectory and the root channels are smoothed with a
**Savitzky-Golay filter** (window 9–11, order 3). Yaw is `np.unwrap`-ed first to
avoid ±π discontinuities.

**Output**: a `(N, 34)` array `qpos_frames` — one full robot pose per video
frame — plus `fps` and `joint_limits`.

---

## 4. Stage 3 — Playback

### `simulate.py` — robot only

Loads the MJCF, then each frame writes `data.qpos = qpos_frames[i]` and calls
`mj_forward` (kinematics only, no physics — we're replaying a pose, not
simulating dynamics). A fixed free camera (azimuth 180°) shows the robot's front
so translation across the stage is visible. Loops automatically; `SPACE` pauses.

### `visualize.py` — side-by-side

Renders the original video (left) next to the MuJoCo render (right) with OpenCV,
overlays timecode, and optionally writes `result.mp4`. Uses an off-screen
`mujoco.Renderer` with the same fixed front camera.

---

## 5. The Asimov Robot Model

From `asimov-1/sim-model/xmls/asimov.xml` (MuJoCo MJCF). **25 actuated DOF**:

| Region | Joints | DOF |
|--------|--------|-----|
| Each leg | hip pitch/roll/yaw, knee, ankle pitch/roll | 6 × 2 |
| Each arm | shoulder pitch/roll/yaw, elbow, wrist yaw | 5 × 2 |
| Torso | waist yaw | 1 |
| Head | neck yaw, neck pitch | 2 |
| (passive toes) | — | 2 |

We retarget arms, legs and waist; neck and wrist are left near-neutral. Joint
limits are parsed straight from the MJCF and enforced during IK, so the robot
never adopts an impossible pose.

---

## 6. Data Flow Summary

```
input_video.mp4
   │  extract_pose.py  (MediaPipe, ~slow, run once)
   ▼
dance.pkl                { fps, landmarks(N,33,3), image_landmarks(N,33,3) }
   │  retarget.py       (coordinate align → root motion → per-limb IK → smooth)
   ▼
retargeted.pkl           { fps, qpos_frames(N,34), joint_limits }
   │  simulate.py / visualize.py
   ▼
on-screen playback  +  result.mp4
```

---

## 7. Design Decisions & Trade-offs

- **IK over angle formulas.** Direct angle math kept breaking because each robot
  joint has its own axis and sign; matching *positions* with a Jacobian is far
  more robust and generalizes to any humanoid MJCF.
- **Kinematic replay, not dynamics.** We inject poses rather than torque-control
  the robot. This guarantees the dance looks exactly like the capture; making
  the robot *physically* balance through the motion would require a controller
  (a much larger problem — see Menlo's Cyclotron).
- **Monocular capture.** One video, no depth sensor — cheap and accessible, at
  the cost of imperfect depth (front/back ambiguity), which is why depth-only
  root translation is intentionally left out.
- **Warm-started IK + Savitzky-Golay.** Together they turn noisy per-frame
  estimates into a smooth, temporally coherent motion.

---

## 8. Where This Could Go Next

- Feed `qpos_frames` as reference motion to a **balancing controller** so the
  robot actually keeps its footing (imitation / RL).
- Stream poses to the **Menlo Platform API** to drive a real or virtual Asimov.
- Add wrist/finger and facial detail with MediaPipe Holistic.
- Beat-align the motion to the music for tighter choreography.
