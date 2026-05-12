"""
position_control_panda.py

목적:
- MuJoCo Menagerie의 Franka Panda 로봇팔을 로드한다.
- 목표 관절 자세를 정의한다.
- 현재 관절 위치와 목표 관절 위치의 오차를 계산한다.
- 단순 P 제어 방식으로 data.ctrl에 제어 입력을 넣는다.
- 결과 영상, CSV 로그, 오차 그래프용 데이터를 저장한다.

주의:
- 이 예제는 교육용 단순 P 제어이다.
- 실제 로봇 제어에서는 actuator와 joint 매핑, torque/position actuator 타입,
  joint limit, 안정성, collision, controller tuning을 더 엄격히 다뤄야 한다.
"""

from pathlib import Path
import csv

import imageio.v2 as imageio
import mujoco
import numpy as np


def get_first_camera_name(model: mujoco.MjModel) -> str | None:
    """
    모델에 정의된 첫 번째 카메라 이름을 반환한다.
    카메라가 없으면 None을 반환한다.
    """
    for camera_id in range(model.ncam):
        name = mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_CAMERA,
            camera_id
        )
        if name:
            return name
    return None


def clip_ctrl(model: mujoco.MjModel, actuator_id: int, value: float) -> float:
    """
    actuator의 허용 제어 범위를 벗어나지 않도록 값을 제한한다.
    """
    ctrl_min = model.actuator_ctrlrange[actuator_id][0]
    ctrl_max = model.actuator_ctrlrange[actuator_id][1]

    return float(np.clip(value, ctrl_min, ctrl_max))


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

    video_path = output_dir / "panda_position_control.mp4"
    csv_path = output_dir / "panda_position_control_log.csv"

    if not xml_path.exists():
        raise FileNotFoundError(f"XML 파일을 찾을 수 없습니다: {xml_path}")

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)

    print("MuJoCo version:", mujoco.__version__)
    print("model.nq:", model.nq)
    print("model.nv:", model.nv)
    print("model.nu:", model.nu)
    print("data.qpos shape:", data.qpos.shape)
    print("data.ctrl shape:", data.ctrl.shape)

    if model.nu == 0:
        raise RuntimeError("이 모델에는 actuator가 없습니다.")

    # 제어할 관절/actuator 개수
    # 교육용으로 앞쪽 actuator 7개 이하만 제어한다.
    controlled_count = min(7, model.nu, model.nq)

    print("Controlled count:", controlled_count)

    # 초기 상태를 몇 step 진행해 안정화한다.
    for _ in range(100):
        mujoco.mj_step(model, data)

    initial_qpos = data.qpos.copy()

    print("Initial qpos first values:")
    for i in range(controlled_count):
        print(f"  qpos_{i}: {initial_qpos[i]:.4f}")

    # 목표 관절 자세 정의
    # 너무 큰 목표값을 주면 모델이 불안정할 수 있으므로 작은 변화량을 사용한다.
    target_qpos = initial_qpos.copy()

    if controlled_count >= 1:
        target_qpos[0] = initial_qpos[0] + 0.25
    if controlled_count >= 2:
        target_qpos[1] = initial_qpos[1] - 0.20
    if controlled_count >= 3:
        target_qpos[2] = initial_qpos[2] + 0.20
    if controlled_count >= 4:
        target_qpos[3] = initial_qpos[3] - 0.15
    if controlled_count >= 5:
        target_qpos[4] = initial_qpos[4] + 0.10
    if controlled_count >= 6:
        target_qpos[5] = initial_qpos[5] + 0.10
    if controlled_count >= 7:
        target_qpos[6] = initial_qpos[6] - 0.10

    print("Target qpos first values:")
    for i in range(controlled_count):
        print(f"  target_qpos_{i}: {target_qpos[i]:.4f}")

    # P 제어 gain
    # 초보자 실습에서는 작은 값부터 시작한다.
    kp = 2.0

    fps = 30
    duration_sec = 8
    total_frames = fps * duration_sec

    renderer = mujoco.Renderer(model, height=480, width=640)
    camera_name = get_first_camera_name(model)

    frames = []
    logs = []

    for frame_idx in range(total_frames):
        # 모든 actuator 명령 초기화
        data.ctrl[:] = 0.0

        # P 제어 적용
        errors = []

        for i in range(controlled_count):
            current = data.qpos[i]
            target = target_qpos[i]
            error = target - current
            raw_ctrl = kp * error

            safe_ctrl = clip_ctrl(model, i, raw_ctrl)

            data.ctrl[i] = safe_ctrl
            errors.append(error)

        # 한 프레임 사이에 여러 물리 step 실행
        for _ in range(5):
            mujoco.mj_step(model, data)

        # 로그 저장
        row = {
            "time": data.time,
        }

        for i in range(controlled_count):
            row[f"target_qpos_{i}"] = target_qpos[i]
            row[f"current_qpos_{i}"] = data.qpos[i]
            row[f"error_{i}"] = target_qpos[i] - data.qpos[i]
            row[f"ctrl_{i}"] = data.ctrl[i]

        row["mean_abs_error"] = float(np.mean(np.abs(errors)))
        logs.append(row)

        # 렌더링
        if camera_name:
            renderer.update_scene(data, camera=camera_name)
        else:
            renderer.update_scene(data)

        frame = renderer.render()
        frames.append(frame)

    imageio.mimsave(video_path, frames, fps=fps)

    if logs:
        fieldnames = list(logs[0].keys())

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(logs)

    renderer.close()

    print(f"Saved video: {video_path}")
    print(f"Saved CSV log: {csv_path}")
    print("Position control completed successfully.")


if __name__ == "__main__":
    main()