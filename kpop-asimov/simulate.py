import mujoco
import mujoco.viewer
import numpy as np
import time


def simulate(retargeted: dict, mjcf_path: str):
    """qpos_frames를 그대로 재생 (IK 결과 full-body 자세)."""
    model = mujoco.MjModel.from_xml_path(mjcf_path)
    data = mujoco.MjData(model)

    fps = retargeted["fps"]
    dt = 1.0 / fps
    qpos_frames = retargeted["qpos_frames"]   # (N, nq)
    n_frames = len(qpos_frames)

    print(f"\nLoaded {n_frames} frames @ {fps:.1f} fps ({n_frames/fps:.1f}s)")
    print("Controls: SPACE=pause/resume, R=restart, Q/ESC=quit")
    print("Starting simulation in 2s...")
    time.sleep(2)

    mujoco.mj_resetData(model, data)
    data.qpos[:] = qpos_frames[0]
    mujoco.mj_forward(model, data)

    frame_idx = 0

    def key_callback(key):
        nonlocal frame_idx
        if key in (ord('R'), ord('r')):
            frame_idx = 0

    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        viewer.cam.lookat[:] = [0.0, 0.0, 0.6]
        viewer.cam.distance = 3.2
        viewer.cam.azimuth = 180.0     # 로봇 정면
        viewer.cam.elevation = -10

        while viewer.is_running():
            step_start = time.time()

            if frame_idx >= n_frames:
                frame_idx = 0  # loop

            data.qpos[:] = qpos_frames[frame_idx]
            mujoco.mj_forward(model, data)
            frame_idx += 1

            viewer.sync()
            sleep_t = dt - (time.time() - step_start)
            if sleep_t > 0:
                time.sleep(sleep_t)

    print("Simulation ended.")
