"""
control_panda_joint.py

목적:
- Panda 로봇팔의 특정 actuator에 사인파 제어 입력을 넣는다.
- 로봇팔 움직임을 MP4 영상으로 저장한다.
- qpos 로그를 CSV로 저장한다.

실무 포인트:
- actuator 제어값은 반드시 ctrl range를 확인하고 작은 값부터 넣는다.
- 처음부터 큰 값을 넣으면 로봇이 튀거나 비정상적으로 움직일 수 있다.
"""

from pathlib import Path
import csv
import math

import imageio.v2 as imageio
import mujoco
import numpy as np


def get_first_camera_name(model: mujoco.MjModel) -> str | None:
    for camera_id in range(model.ncam):
        name = mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_CAMERA,
            camera_id
        )
        if name:
            return name
    return None


def main():
    project_root = Path(__file__).resolve().parents[1]

    xml_path = (
        project_root
        / "mujoco_menagerie"
        / "franka_emika_panda"
        / "scene.xml"
    )

    output_dir = project_root / "data"
    output_dir.mkdir(exist_ok=True)

    video_path = output_dir / "panda_joint_control.mp4"
    csv_path = output_dir / "panda_joint_log.csv"

    if not xml_path.exists():
        raise FileNotFoundError(f"XML 파일을 찾을 수 없습니다: {xml_path}")

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)

    print("MuJoCo version:", mujoco.__version__)
    print("Number of joints:", model.njnt)
    print("Number of actuators:", model.nu)
    print("qpos shape:", data.qpos.shape)
    print("qvel shape:", data.qvel.shape)
    print("ctrl shape:", data.ctrl.shape)

    if model.nu == 0:
        raise RuntimeError("이 모델에는 actuator가 없습니다. data.ctrl로 제어할 수 없습니다.")

    # 제어할 actuator 번호
    target_actuator_id = 0

    # actuator 제어 범위 확인
    ctrl_min = model.actuator_ctrlrange[target_actuator_id][0]
    ctrl_max = model.actuator_ctrlrange[target_actuator_id][1]

    print(
        f"Target actuator id={target_actuator_id}, "
        f"ctrl range=[{ctrl_min:.3f}, {ctrl_max:.3f}]"
    )

    camera_name = get_first_camera_name(model)
    print("Selected camera:", camera_name)

    renderer = mujoco.Renderer(model, height=480, width=640)

    fps = 30
    duration_sec = 6
    total_frames = fps * duration_sec

    frames = []
    logs = []

    # 너무 큰 값을 넣지 않도록 actuator range의 일부만 사용한다.
    # range가 너무 크거나 비정상적인 경우를 대비해 기본 amplitude를 제한한다.
    range_based_amp = 0.2 * min(abs(ctrl_min), abs(ctrl_max))
    amplitude = min(range_based_amp, 0.5)

    # 만약 ctrl range가 [0, 0]처럼 나오거나 너무 작으면 기본값 사용
    if amplitude <= 1e-6:
        amplitude = 0.2

    frequency_hz = 0.5

    for frame_idx in range(total_frames):
        t = frame_idx / fps

        # 모든 제어 입력 초기화
        data.ctrl[:] = 0.0

        # 사인파 제어 입력
        command = amplitude * math.sin(2.0 * math.pi * frequency_hz * t)

        # 제어 범위를 넘지 않도록 제한
        command = float(np.clip(command, ctrl_min, ctrl_max))

        data.ctrl[target_actuator_id] = command

        # 한 프레임당 여러 번 물리 step 실행
        for _ in range(5):
            mujoco.mj_step(model, data)

        # 로그 저장
        row = {
            "time": data.time,
            "command": command,
        }

        # qpos 앞쪽 값 일부 저장
        for i in range(min(10, len(data.qpos))):
            row[f"qpos_{i}"] = data.qpos[i]

        # qvel 앞쪽 값 일부 저장
        for i in range(min(10, len(data.qvel))):
            row[f"qvel_{i}"] = data.qvel[i]

        logs.append(row)

        # 렌더링
        if camera_name:
            renderer.update_scene(data, camera=camera_name)
        else:
            renderer.update_scene(data)

        frame = renderer.render()
        frames.append(frame)

    imageio.mimsave(video_path, frames, fps=fps)

    # CSV 로그 저장
    if logs:
        fieldnames = list(logs[0].keys())

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(logs)

    renderer.close()

    print(f"Saved video: {video_path}")
    print(f"Saved CSV log: {csv_path}")
    print("Panda joint control completed successfully.")


if __name__ == "__main__":
    main()