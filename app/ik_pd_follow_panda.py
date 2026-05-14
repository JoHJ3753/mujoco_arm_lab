"""
ik_pd_follow_panda.py

목적:
- ik_compute_target_qpos.py에서 저장한 target_qpos를 읽는다.
- 초기 상태에서 시작한 Panda 로봇팔이 PD Controller로 target_qpos를 추종하게 한다.
- data.ctrl에 제어 입력을 넣고 mj_step으로 물리 시뮬레이션을 진행한다.
- 결과 영상과 CSV 로그를 저장한다.

핵심 구조:
    IK 계산 결과 = target_qpos
    실제 시뮬레이션 = sim_data
    PD 제어 = ctrl = Kp * qpos_error - Kd * qvel
"""

from pathlib import Path
import csv

import imageio.v2 as imageio
import mujoco
import numpy as np


def get_camera_name(model: mujoco.MjModel) -> str | None:
    camera_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_CAMERA,
        "front_camera"
    )

    if camera_id >= 0:
        return "front_camera"

    for i in range(model.ncam):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, i)
        if name:
            return name

    return None


def find_first_existing_body(model: mujoco.MjModel, candidate_names: list[str]) -> tuple[str, int]:
    for name in candidate_names:
        body_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            name
        )

        if body_id >= 0:
            return name, body_id

    raise RuntimeError("End-Effector 후보 body를 찾지 못했습니다.")


def clip_ctrl(model: mujoco.MjModel, actuator_id: int, value: float) -> float:
    ctrl_min = model.actuator_ctrlrange[actuator_id][0]
    ctrl_max = model.actuator_ctrlrange[actuator_id][1]
    return float(np.clip(value, ctrl_min, ctrl_max))


def build_actuator_joint_map(model: mujoco.MjModel) -> list[tuple[int, int, int, int]]:
    """
    교육용 단순 매핑:
    - actuator i가 qpos i, qvel i를 제어한다고 단순 가정하지 않고,
      hinge/slide joint의 qpos_addr, dof_addr를 이용해 앞쪽 관절들을 매핑한다.
    - 실제 모델에서는 actuator가 어떤 joint를 제어하는지 XML의 actuator 설정을 확인하는 것이 더 정확하다.

    반환:
    [
      (actuator_id, joint_id, qpos_addr, dof_addr),
      ...
    ]
    """
    mapping = []

    actuator_id = 0

    for joint_id in range(model.njnt):
        if actuator_id >= model.nu:
            break

        joint_type = model.jnt_type[joint_id]

        if joint_type not in [
            mujoco.mjtJoint.mjJNT_HINGE,
            mujoco.mjtJoint.mjJNT_SLIDE,
        ]:
            continue

        qpos_addr = model.jnt_qposadr[joint_id]
        dof_addr = model.jnt_dofadr[joint_id]

        if qpos_addr < model.nq and dof_addr < model.nv:
            mapping.append((actuator_id, joint_id, qpos_addr, dof_addr))
            actuator_id += 1

    return mapping


def main():
    project_root = Path(__file__).resolve().parents[1]

    xml_path = (
        project_root
        / "mujoco_menagerie"
        / "franka_emika_panda"
        / "scene_with_cube.xml"
    )

    target_qpos_path = project_root / "data" / "ik_pd" / "ik_target_qpos.npy"

    output_dir = project_root / "data" / "ik_pd"
    output_dir.mkdir(parents=True, exist_ok=True)

    video_path = output_dir / "panda_ik_pd_follow.mp4"
    csv_path = output_dir / "panda_ik_pd_follow_log.csv"

    if not target_qpos_path.exists():
        raise FileNotFoundError(
            f"IK target qpos 파일이 없습니다: {target_qpos_path}\n"
            "먼저 python app/ik_compute_target_qpos.py를 실행하세요."
        )

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)

    target_qpos = np.load(target_qpos_path)

    if target_qpos.shape[0] != model.nq:
        raise RuntimeError(
            f"target_qpos 크기와 model.nq가 다릅니다. "
            f"target_qpos={target_qpos.shape}, model.nq={model.nq}"
        )

    ee_name, ee_body_id = find_first_existing_body(
        model,
        ["hand", "panda_hand", "link7", "attachment"],
    )

    cube_body_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        "target_cube"
    )

    if cube_body_id < 0:
        raise RuntimeError("target_cube body를 찾지 못했습니다.")

    camera_name = get_camera_name(model)
    renderer = mujoco.Renderer(model, height=480, width=640)

    actuator_joint_map = build_actuator_joint_map(model)

    print("Selected End-Effector body:", ee_name)
    print("Selected camera:", camera_name)
    print("model.nq:", model.nq)
    print("model.nv:", model.nv)
    print("model.nu:", model.nu)
    print("Actuator-joint mapping count:", len(actuator_joint_map))

    for actuator_id, joint_id, qpos_addr, dof_addr in actuator_joint_map:
        joint_name = mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            joint_id
        )
        actuator_name = mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            actuator_id
        )

        print(
            f"actuator_id={actuator_id}, actuator={actuator_name}, "
            f"joint_id={joint_id}, joint={joint_name}, "
            f"qpos_addr={qpos_addr}, dof_addr={dof_addr}"
        )

    # PD gain
    # 너무 크게 잡으면 튀고, 너무 작으면 목표 추종이 느리다.
    kp = 8.0
    kd = 1.0

    fps = 30
    duration_sec = 8
    total_frames = fps * duration_sec

    frames = []
    logs = []

    # 초기 상태 안정화
    for _ in range(50):
        mujoco.mj_step(model, data)

    for frame_idx in range(total_frames):
        data.ctrl[:] = 0.0

        qpos_errors = []

        for actuator_id, joint_id, qpos_addr, dof_addr in actuator_joint_map:
            current_qpos = data.qpos[qpos_addr]
            current_qvel = data.qvel[dof_addr]
            target = target_qpos[qpos_addr]

            error = target - current_qpos
            raw_ctrl = kp * error - kd * current_qvel

            safe_ctrl = clip_ctrl(model, actuator_id, raw_ctrl)
            data.ctrl[actuator_id] = safe_ctrl

            qpos_errors.append(error)

        for _ in range(5):
            mujoco.mj_step(model, data)

        mujoco.mj_forward(model, data)

        ee_pos = data.xpos[ee_body_id].copy()
        cube_pos = data.xpos[cube_body_id].copy()
        approach_target_pos = cube_pos + np.array([0.0, 0.0, 0.15])
        ee_target_distance = float(np.linalg.norm(approach_target_pos - ee_pos))

        mean_abs_qpos_error = float(np.mean(np.abs(qpos_errors))) if qpos_errors else 0.0

        row = {
            "frame": frame_idx,
            "time": data.time,
            "mean_abs_qpos_error": mean_abs_qpos_error,
            "ee_target_distance": ee_target_distance,
            "ee_x": ee_pos[0],
            "ee_y": ee_pos[1],
            "ee_z": ee_pos[2],
            "target_x": approach_target_pos[0],
            "target_y": approach_target_pos[1],
            "target_z": approach_target_pos[2],
        }

        for idx, (actuator_id, joint_id, qpos_addr, dof_addr) in enumerate(actuator_joint_map):
            row[f"joint_{idx}_target_qpos"] = target_qpos[qpos_addr]
            row[f"joint_{idx}_current_qpos"] = data.qpos[qpos_addr]
            row[f"joint_{idx}_current_qvel"] = data.qvel[dof_addr]
            row[f"joint_{idx}_ctrl"] = data.ctrl[actuator_id]
            row[f"joint_{idx}_qpos_error"] = target_qpos[qpos_addr] - data.qpos[qpos_addr]

        logs.append(row)

        if camera_name:
            renderer.update_scene(data, camera=camera_name)
        else:
            renderer.update_scene(data)

        frames.append(renderer.render())

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
    print("IK target qpos PD following completed successfully.")


if __name__ == "__main__":
    main()