"""
IK 기반 리타게팅.

각도 공식을 쓰지 않고, 매 프레임마다 캡처된 landmark 위치(팔꿈치/손목/무릎/발목)에
로봇의 대응 body가 가도록 MuJoCo Jacobian IK로 관절각을 푼다.

흐름 (프레임마다):
  1. MediaPipe world 좌표를 로봇 좌표계로 회전 변환
  2. 사람 팔/다리의 "방향"은 그대로 쓰되 "길이"는 로봇 segment 길이로 스케일
     → 로봇 어깨/엉덩이를 기준으로 목표 elbow/wrist/knee/ankle 위치를 만든다
  3. 팔/다리별로 damped least-squares IK 반복 → 관절각 갱신
  4. 직전 프레임 해를 warm-start로 사용해 시간적으로 부드럽게
"""

import numpy as np
import mujoco
import xml.etree.ElementTree as ET
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation as Rot


MP_IDX = {
    "nose": 0,
    "left_shoulder": 11,  "right_shoulder": 12,
    "left_elbow": 13,     "right_elbow": 14,
    "left_wrist": 15,     "right_wrist": 16,
    "left_hip": 23,       "right_hip": 24,
    "left_knee": 25,      "right_knee": 26,
    "left_ankle": 27,     "right_ankle": 28,
}

# MediaPipe world: x=right(이미지), y=DOWN(아래), z=toward cam
# Robot: x=forward, y=left, z=up
#   robot_x(forward) = -mp_z
#   robot_y(left)    = +mp_x   ← 해부학적 left_*(11,13..)는 이미지 +x에 있음
#   robot_z(up)      = -mp_y   ← MediaPipe y가 아래 방향이므로 부호 반전
# (det=+1 right-handed → 거울 반전 없음)
def mp_to_robot(v):
    return np.array([-v[2], v[0], -v[1]])


# 각 사지(limb) IK 정의:
#   joints: 그 사지를 움직이는 actuated joint 이름들 (root → tip 순서)
#   targets: [(MP landmark, robot body name), ...]  맞춰야 할 위치들
#   anchor_body: 목표 위치 계산의 기준이 되는 로봇 body (어깨/엉덩이)
LIMBS = {
    "left_arm": {
        "joints": ["left_shoulder_pitch_joint", "left_shoulder_roll_joint",
                   "left_shoulder_yaw_joint", "left_elbow_joint"],
        "anchor_body": "left_shoulder_pitch_link",
        "chain": [  # (MP_from, MP_to, robot_target_body)
            ("left_shoulder", "left_elbow", "left_elbow_link"),
            ("left_elbow", "left_wrist", "left_wrist_yaw_link"),
        ],
    },
    "right_arm": {
        "joints": ["right_shoulder_pitch_joint", "right_shoulder_roll_joint",
                   "right_shoulder_yaw_joint", "right_elbow_joint"],
        "anchor_body": "right_shoulder_pitch_link",
        "chain": [
            ("right_shoulder", "right_elbow", "right_elbow_link"),
            ("right_elbow", "right_wrist", "right_wrist_yaw_link"),
        ],
    },
    "left_leg": {
        "joints": ["left_hip_pitch_joint", "left_hip_roll_joint",
                   "left_hip_yaw_joint", "left_knee_joint"],
        "anchor_body": "left_hip_pitch_link",
        "chain": [
            ("left_hip", "left_knee", "left_knee_link"),
            ("left_knee", "left_ankle", "left_ankle_roll_link"),
        ],
    },
    "right_leg": {
        "joints": ["right_hip_pitch_joint", "right_hip_roll_joint",
                   "right_hip_yaw_joint", "right_knee_joint"],
        "anchor_body": "right_hip_pitch_link",
        "chain": [
            ("right_hip", "right_knee", "right_knee_link"),
            ("right_knee", "right_ankle", "right_ankle_roll_link"),
        ],
    },
}


def parse_joint_limits(mjcf_path: str) -> dict:
    limits = {}
    for joint in ET.parse(mjcf_path).getroot().iter("joint"):
        name, r = joint.get("name"), joint.get("range")
        if name and r:
            lo, hi = map(float, r.split())
            limits[name] = (lo, hi)
    return limits


def nrm(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-8 else v


def bid(model, name):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)


def jid(model, name):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)


def measure_segment_lengths(model, data):
    """중립 자세에서 로봇 segment 길이를 잰다."""
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    seg = {}
    pairs = [
        ("left_shoulder_pitch_link", "left_elbow_link"),
        ("left_elbow_link", "left_wrist_yaw_link"),
        ("right_shoulder_pitch_link", "right_elbow_link"),
        ("right_elbow_link", "right_wrist_yaw_link"),
        ("left_hip_pitch_link", "left_knee_link"),
        ("left_knee_link", "left_ankle_roll_link"),
        ("right_hip_pitch_link", "right_knee_link"),
        ("right_knee_link", "right_ankle_roll_link"),
    ]
    for a, b in pairs:
        pa = data.xpos[bid(model, a)].copy()
        pb = data.xpos[bid(model, b)].copy()
        seg[(a, b)] = float(np.linalg.norm(pb - pa))
    return seg


def build_targets(lm, model, data, limb, seg):
    """
    사람 사지의 방향 + 로봇 segment 길이로 목표 위치들을 만든다.
    로봇의 현재 anchor body 위치(어깨/엉덩이)를 기준으로 누적.
    """
    anchor_pos = data.xpos[bid(model, limb["anchor_body"])].copy()
    targets = []
    cur = anchor_pos
    for (mp_from, mp_to, target_body) in limb["chain"]:
        d_human = lm[MP_IDX[mp_to]] - lm[MP_IDX[mp_from]]
        d_robot = mp_to_robot(d_human)
        d_robot = nrm(d_robot)
        # segment 길이
        # target_body 기준 robot segment 찾기
        for (a, b), L in seg.items():
            if b == target_body:
                length = L
                break
        else:
            length = 0.1
        cur = cur + d_robot * length
        targets.append((target_body, cur.copy()))
    return targets


def damped_pinv(J, lam=0.15):
    JT = J.T
    m = J.shape[0]
    return JT @ np.linalg.inv(J @ JT + (lam ** 2) * np.eye(m))


def solve_limb_ik(model, data, limb, lm, seg, limits, iters=20):
    """한 사지에 대해 IK 반복. data.qpos를 직접 갱신."""
    joint_ids = [jid(model, j) for j in limb["joints"]]
    dof_adrs = [model.jnt_dofadr[ji] for ji in joint_ids]
    qpos_adrs = [model.jnt_qposadr[ji] for ji in joint_ids]

    for _ in range(iters):
        mujoco.mj_forward(model, data)
        targets = build_targets(lm, model, data, limb, seg)

        err = []
        J_rows = []
        for (body_name, tgt) in targets:
            b = bid(model, body_name)
            cur = data.xpos[b].copy()
            err.append(tgt - cur)
            jacp = np.zeros((3, model.nv))
            mujoco.mj_jacBody(model, data, jacp, None, b)
            J_rows.append(jacp[:, dof_adrs])  # 3 x len(joints)

        err = np.concatenate(err)            # 3*ntarget
        J = np.vstack(J_rows)                # (3*ntarget) x njoints

        dq = damped_pinv(J) @ err
        dq = np.clip(dq, -0.3, 0.3)          # 한 스텝 변화 제한

        for k, qa in enumerate(qpos_adrs):
            new_val = data.qpos[qa] + dq[k]
            jn = limb["joints"][k]
            if jn in limits:
                lo, hi = limits[jn]
                new_val = np.clip(new_val, lo, hi)
            data.qpos[qa] = new_val


def facing_yaw(left_pt, right_pt):
    """좌우 마커(어깨 or 엉덩이)로 몸이 향하는 방향(yaw)을 구한다."""
    d = mp_to_robot(left_pt - right_pt)   # 정면일 때 로봇 +y(왼쪽)
    d[2] = 0.0
    d = nrm(d)
    forward = np.cross(d, np.array([0, 0, 1]))   # 가슴이 향하는 수평 방향
    return float(np.arctan2(forward[1], forward[0]))


def compute_root_motion(world_lm, image_lm):
    """
    골반(root)의 전역 이동/회전을 계산.
      - 좌우 이동: 화면 속 엉덩이 x 위치 (왔다갔다)
      - 상하 이동: 다리가 접힌 정도 (앉기/점프 시 발이 땅에 붙도록)
      - yaw 회전: 엉덩이 라인이 향하는 방향 (턴)
      - waist_yaw: 어깨 방향 - 엉덩이 방향 (상체 비틀기)
    반환: root_pos(N,3), root_quat(N,4 wxyz), waist_yaw(N,)
    """
    N = len(world_lm)
    Y_SCALE = 1.6       # 화면 폭 → 좌우 이동(m)
    Z_BASE = 0.85       # 기본 골반 높이

    # 다리 길이 정규화 기준 (서있을 때) — y축 부호 무관하게 절대거리
    leg_v = []
    for wl in world_lm:
        ank = (wl[MP_IDX["left_ankle"]][1] + wl[MP_IDX["right_ankle"]][1]) / 2
        hip = (wl[MP_IDX["left_hip"]][1] + wl[MP_IDX["right_hip"]][1]) / 2
        leg_v.append(abs(ank - hip))
    leg_v = np.array(leg_v)
    neutral_leg = np.median(leg_v)

    hip_x = np.array([(il[MP_IDX["left_hip"]][0] + il[MP_IDX["right_hip"]][0]) / 2
                      for il in image_lm])
    hip_x_base = np.median(hip_x)

    root_pos = np.zeros((N, 3))
    yaws = np.zeros(N)
    waist = np.zeros(N)

    for i in range(N):
        wl = world_lm[i]
        # 좌우 이동 (화면 오른쪽 = 로봇 -y)
        ry = -(hip_x[i] - hip_x_base) * Y_SCALE
        # 상하 (다리 접힘 비율)
        rz = Z_BASE * float(np.clip(leg_v[i] / (neutral_leg + 1e-6), 0.55, 1.05))
        root_pos[i] = [0.0, ry, rz]

        hip_yaw = facing_yaw(wl[MP_IDX["left_hip"]], wl[MP_IDX["right_hip"]])
        sh_yaw  = facing_yaw(wl[MP_IDX["left_shoulder"]], wl[MP_IDX["right_shoulder"]])
        yaws[i] = hip_yaw
        # 상체 비틀기 = 어깨 - 엉덩이
        tw = sh_yaw - hip_yaw
        tw = np.arctan2(np.sin(tw), np.cos(tw))   # -π~π 정규화
        waist[i] = tw

    # yaw: 시작 프레임 기준 정면을 보도록 baseline 보정
    yaws = np.unwrap(yaws)
    yaw_baseline = np.median(yaws[:min(10, N)])
    yaws = yaws - yaw_baseline

    # 스무딩
    root_pos[:, 1] = savgol_filter(root_pos[:, 1], 11, 3) if N >= 11 else root_pos[:, 1]
    root_pos[:, 2] = savgol_filter(root_pos[:, 2], 11, 3) if N >= 11 else root_pos[:, 2]
    yaws = savgol_filter(yaws, 11, 3) if N >= 11 else yaws
    waist = savgol_filter(waist, 11, 3) if N >= 11 else waist

    root_quat = np.zeros((N, 4))
    for i in range(N):
        q = Rot.from_euler('z', yaws[i]).as_quat()   # (x,y,z,w)
        root_quat[i] = [q[3], q[0], q[1], q[2]]       # → (w,x,y,z) MuJoCo
    return root_pos, root_quat, waist


def retarget(pose_data: dict, mjcf_path: str) -> dict:
    landmarks = pose_data["landmarks"]   # (N, 33, 3)
    limits = parse_joint_limits(mjcf_path)

    model = mujoco.MjModel.from_xml_path(mjcf_path)
    data = mujoco.MjData(model)

    seg = measure_segment_lengths(model, data)
    print("로봇 segment 길이:")
    for (a, b), L in seg.items():
        print(f"  {a} → {b}: {L*100:.1f}cm")

    # 전역 골반 모션 (좌우 이동 / 상하 / 회전 / 상체 비틀기)
    image_lm = pose_data["image_landmarks"]
    root_pos, root_quat, waist_seq = compute_root_motion(landmarks, image_lm)
    print(f"root 이동 좌우: [{root_pos[:,1].min():.2f}, {root_pos[:,1].max():.2f}]m  "
          f"높이: [{root_pos[:,2].min():.2f}, {root_pos[:,2].max():.2f}]m")

    mujoco.mj_resetData(model, data)

    n = len(landmarks)
    qpos_frames = np.zeros((n, model.nq))

    wj = jid(model, "waist_yaw_joint")
    wa = model.jnt_qposadr[wj]
    wlo, whi = limits["waist_yaw_joint"]

    print(f"\nIK 풀이 중 ({n} 프레임)...")
    for i, lm in enumerate(landmarks):
        # 골반 전역 위치/자세 적용 (IK 전에 — anchor body가 제 위치로 가도록)
        data.qpos[0:3] = root_pos[i]
        data.qpos[3:7] = root_quat[i]
        # 상체 비틀기
        data.qpos[wa] = np.clip(waist_seq[i], wlo, whi)

        # 각 사지 IK (warm-start: 직전 프레임 qpos 유지)
        for name, limb in LIMBS.items():
            solve_limb_ik(model, data, limb, lm, seg, limits, iters=15)

        # IK가 흔든 root 다시 고정
        data.qpos[0:3] = root_pos[i]
        data.qpos[3:7] = root_quat[i]

        qpos_frames[i] = data.qpos.copy()

        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{n}")

    # 관절별 시간축 스무딩
    print("스무딩...")
    for ji in range(model.njnt):
        qa = model.jnt_qposadr[ji]
        jtype = model.jnt_type[ji]
        if jtype == mujoco.mjtJoint.mjJNT_HINGE:
            seq = qpos_frames[:, qa]
            if len(seq) >= 9:
                qpos_frames[:, qa] = savgol_filter(seq, 9, 3)

    return {
        "fps": pose_data["fps"],
        "qpos_frames": qpos_frames,   # (N, nq) 전체 자세 재생용
        "joint_limits": limits,
    }
