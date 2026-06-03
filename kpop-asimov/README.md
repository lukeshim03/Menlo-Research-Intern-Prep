# K-pop Asimov Dance

Make the **Asimov humanoid robot** dance to K-pop videos inside a MuJoCo simulation.

> Demo song: **BTS — Dynamite** 🕺

```
YouTube  →  MediaPipe  →  IK Retargeting  →  MuJoCo Playback
            (33 landmarks)  (25 DOF Jacobian)    Asimov sim
```

For a deep dive into how it works, see **[ARCHITECTURE.md](ARCHITECTURE.md)**.

---

## How It Works

Instead of hand-written angle formulas, motion is transferred with **inverse
kinematics (IK)** — every frame, the captured joint positions (elbow, wrist,
knee, ankle) become targets that a MuJoCo Jacobian solver drives the robot's
corresponding bodies toward.

| Stage | What happens |
|-------|--------------|
| **Pose capture** | MediaPipe PoseLandmarker extracts 33 3D landmarks per frame |
| **IK retargeting** | Human limb *direction* is kept, *length* is rescaled to the robot's segments → targets solved with damped least-squares IK |
| **Root motion** | The pelvis carries global translation (side-to-side), vertical (crouch/jump), body yaw (turns) and torso twist — the robot is **not** pinned in place |
| **Playback** | Full `qpos` pose is injected into MuJoCo each frame |

The coordinate transform aligns MediaPipe (y-down) with the robot (z-up) as a
right-handed rotation, so there is no left/right mirroring.

---

## Setup

```powershell
# Python 3.12 virtual env (MuJoCo doesn't support 3.14 yet)
uv venv .venv --python 3.12
uv pip install -r requirements.txt --python .venv\Scripts\python.exe

# Asimov robot model (if not already cloned)
git clone https://github.com/asimovinc/asimov-1 ../asimov-1
```

---

## Run

### 1. Download the video (BTS Dynamite)

```powershell
.venv\Scripts\yt-dlp.exe -f "best[ext=mp4]" `
  --js-runtimes "node:$((Get-Command node).Source)" `
  --remote-components "ejs:github" `
  -o "input_video.mp4" `
  "https://www.youtube.com/watch?v=S6il2Ud0SPA"
```

### 2. Pose extraction + IK retargeting + side-by-side (all in one)

```powershell
.venv\Scripts\python.exe -c @'
import sys, pickle
sys.path.insert(0, ".")
from extract_pose import extract_pose
from retarget import retarget
from visualize import render_side_by_side

MJCF = "../asimov-1/sim-model/xmls/asimov.xml"

pose = extract_pose("input_video.mp4", "dance.pkl")      # 1) MediaPipe
r = retarget(pose, MJCF)                                  # 2) IK
with open("retargeted.pkl", "wb") as f: pickle.dump(r, f)
render_side_by_side("input_video.mp4", r, MJCF,
                    output_path="result.mp4")            # 3) left: original / right: robot
'@
```

### 3. Re-use the cached retargeting

```powershell
# Re-play the side-by-side only
.venv\Scripts\python.exe -c @'
import sys, pickle
sys.path.insert(0, ".")
from visualize import render_side_by_side
with open("retargeted.pkl", "rb") as f: r = pickle.load(f)
render_side_by_side("input_video.mp4", r,
                    "../asimov-1/sim-model/xmls/asimov.xml",
                    output_path="result.mp4")
'@
```

```powershell
# MuJoCo viewer only (robot alone)
.venv\Scripts\python.exe -c @'
import sys, pickle
sys.path.insert(0, ".")
from simulate import simulate
with open("retargeted.pkl", "rb") as f: r = pickle.load(f)
simulate(r, "../asimov-1/sim-model/xmls/asimov.xml")
'@
```

---

## Controls

| Key | Action |
|-----|--------|
| `SPACE` | Pause / resume |
| `R` | Restart (viewer) |
| `Q` / `ESC` | Quit |

The side-by-side run also writes `result.mp4` (left: original, right: Asimov).

---

## Project Layout

```
kpop-asimov/
├── extract_pose.py   # MediaPipe PoseLandmarker → 33 landmarks
├── retarget.py       # IK retargeting + root motion (core)
├── simulate.py       # MuJoCo viewer playback
├── visualize.py      # original video + simulation side-by-side
├── requirements.txt
├── README.md
├── ARCHITECTURE.md   # full system explanation
└── asimov-1/         # Asimov robot MJCF (cloned separately)
```

---

## Requirements

- Python 3.12 (MuJoCo compatibility)
- mediapipe ≥ 0.10 · mujoco ≥ 3.0 · opencv-python · scipy · numpy
- yt-dlp (+ Node.js for YouTube downloads)
