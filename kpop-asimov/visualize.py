import cv2
import mujoco
import numpy as np


def render_side_by_side(video_path: str, retargeted: dict, mjcf_path: str, output_path: str = None):
    model = mujoco.MjModel.from_xml_path(mjcf_path)
    data = mujoco.MjData(model)

    fps = retargeted["fps"]
    dt = 1.0 / fps
    qpos_frames = retargeted["qpos_frames"]   # (N, nq)
    n_frames = len(qpos_frames)

    cap = cv2.VideoCapture(video_path)
    vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    render_w, render_h = 480, 640   # 세로 영상에 맞춤
    renderer = mujoco.Renderer(model, height=render_h, width=render_w)

    # 고정 카메라 (추적 X → 로봇의 좌우 이동이 보이도록)
    fixed_cam = mujoco.MjvCamera()
    fixed_cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    fixed_cam.lookat[:] = [0.0, 0.0, 0.6]
    fixed_cam.distance = 3.2
    fixed_cam.azimuth = 180.0     # 로봇 정면을 봄
    fixed_cam.elevation = -10.0

    # video를 render 높이에 맞춰 스케일
    scale = render_h / vid_h
    new_w = int(vid_w * scale)
    out_w = new_w + render_w
    out_h = render_h

    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (out_w, out_h))

    print(f"Rendering side-by-side: {n_frames} frames...")
    print("Q=quit, SPACE=pause")

    frame_idx = 0
    paused = False

    while frame_idx < n_frames:
        ret, vid_frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, vid_frame = cap.read()
        if not ret:
            break

        if not paused:
            data.qpos[:] = qpos_frames[frame_idx]
            mujoco.mj_forward(model, data)

            renderer.update_scene(data, camera=fixed_cam)
            sim_rgb = renderer.render()
            sim_bgr = cv2.cvtColor(sim_rgb, cv2.COLOR_RGB2BGR)

            frame_idx += 1

        vid_resized = cv2.resize(vid_frame, (new_w, render_h))

        canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
        canvas[:, :new_w] = vid_resized
        canvas[:, new_w:] = sim_bgr

        t = frame_idx / fps
        cv2.putText(canvas, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(canvas, "Asimov (IK)", (new_w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 120), 2)
        cv2.putText(canvas, f"t={t:.1f}s  {frame_idx}/{n_frames}", (10, out_h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        cv2.imshow("K-pop Asimov Dance", canvas)
        if writer:
            writer.write(canvas)

        key = cv2.waitKey(int(dt * 1000)) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key == ord(" "):
            paused = not paused

    cap.release()
    renderer.close()
    if writer:
        writer.release()
        print(f"Saved to {output_path}")
    cv2.destroyAllWindows()
