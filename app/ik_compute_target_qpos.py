"""
ik_compute_target_qpos.py

목적:
- Panda 로봇팔 End-Effector가 target_cube 위쪽 접근점에 가까워지도록 IK를 수행한다.
- IK 결과로 얻은 qpos를 target_qpos로 저장한다.
- 이후 PD Controller가 이 target_qpos를 추종하도록 사용한다.

중요:
- 이 파일은 IK 목표 자세 계산 전용이다.
- 실제 물리 제어는 ik_pd_follow_panda.py에서 수행한다.
"""

from pathlib import Path
import csv

import mujoco
import numpy as np


def find_first_existing_body(model: mujoco.MjModel, candidate_names: list[str]) -> tuple[str, int]:
    for name in candidate_names:
        body_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            name
        )

        if body_id >= 0:
            return name, body_id

    raise RuntimeError(
        "End-Effector 후보 body를 찾지 못했습니다. "
        "python app/list_body_names.py로 body 이름을 확인하세요."
    )


def clamp_qpos_to_joint_limits(model: mujoco.MjModel, data: mujoco.MjData):
    """
    hinge/slide joint 중 range 제한이 있는 joint에 대해 qpos를 제한한다.
    """
    for joint_id in range(model.njnt):
        if not model.jnt_limited[joint_id]:
            continue

        joint_type = model.jnt_type[joint_id]
        qpos_addr = model.jnt_qposadr[joint_id]

        if joint_type in [
            mujoco.mjtJoint.mjJNT_HINGE,
            mujoco.mjtJoint.mjJNT_SLIDE,
        ]:
            qmin = model.jnt_range[joint_id][0]
            qmax = model.jnt_range[joint_id][1]
            data.qpos[qpos_addr] = np.clip(data.qpos[qpos_addr], qmin, qmax)


def main():
    project_root = Path(__file__).resolve().parents[1]

    xml_path = (
        project_root
        / "mujoco_menagerie"
        / "franka_emika_panda"
        / "scene_with_cube.xml"
    )

    output_dir = project_root / "data" / "ik_pd"
    output_dir.mkdir(parents=True, exist_ok=True)

    target_qpos_path = output_dir / "ik_target_qpos.npy"
    log_path = output_dir / "ik_compute_target_qpos_log.csv"

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)

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

    mujoco.mj_forward(model, data)

    cube_pos = data.xpos[cube_body_id].copy()

    # 큐브 중심이 아니라 큐브 위쪽 접근점을 목표로 둔다.
    target_pos = cube_pos + np.array([0.0, 0.0, 0.15])

    print("Selected End-Effector body:", ee_name)
    print("Cube position:", cube_pos)
    print("IK target position:", target_pos)

    # Damped Least Squares 설정
    damping = 0.05
    step_scale = 0.5
    max_dq = 0.03
    max_iterations = 300
    tolerance = 0.03

    logs = []

    for iteration in range(max_iterations):
        mujoco.mj_forward(model, data)

        ee_pos = data.xpos[ee_body_id].copy()
        error = target_pos - ee_pos
        distance = float(np.linalg.norm(error))

        row = {
            "iteration": iteration,
            "distance": distance,
            "ee_x": ee_pos[0],
            "ee_y": ee_pos[1],
            "ee_z": ee_pos[2],
            "target_x": target_pos[0],
            "target_y": target_pos[1],
            "target_z": target_pos[2],
        }

        for i in range(min(10, model.nq)):
            row[f"qpos_{i}"] = data.qpos[i]

        logs.append(row)

        if iteration % 20 == 0:
            print(f"iter={iteration:03d}, distance={distance:.4f}")

        if distance < tolerance:
            print(f"IK target reached. iteration={iteration}, distance={distance:.4f}")
            break

        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))

        mujoco.mj_jacBody(model, data, jacp, jacr, ee_body_id)

        J = jacp

        # Damped Least Squares:
        # dq = J.T @ inv(J @ J.T + lambda^2 I) @ error
        A = J @ J.T + (damping ** 2) * np.eye(3)
        dqvel_space = J.T @ np.linalg.solve(A, error)

        dqvel_space = step_scale * dqvel_space
        dqvel_space = np.clip(dqvel_space, -max_dq, max_dq)

        for joint_id in range(model.njnt):
            joint_type = model.jnt_type[joint_id]

            if joint_type not in [
                mujoco.mjtJoint.mjJNT_HINGE,
                mujoco.mjtJoint.mjJNT_SLIDE,
            ]:
                continue

            qpos_addr = model.jnt_qposadr[joint_id]
            dof_addr = model.jnt_dofadr[joint_id]

            if qpos_addr < model.nq and dof_addr < model.nv:
                data.qpos[qpos_addr] += dqvel_space[dof_addr]

        clamp_qpos_to_joint_limits(model, data)

    mujoco.mj_forward(model, data)

    final_ee_pos = data.xpos[ee_body_id].copy()
    final_distance = float(np.linalg.norm(target_pos - final_ee_pos))

    print("Final End-Effector position:", final_ee_pos)
    print("Final distance:", final_distance)

    # IK 결과 qpos 저장
    np.save(target_qpos_path, data.qpos.copy())

    if logs:
        fieldnames = list(logs[0].keys())

        with open(log_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(logs)

    print(f"Saved IK target qpos: {target_qpos_path}")
    print(f"Saved IK log: {log_path}")


if __name__ == "__main__":
    main()