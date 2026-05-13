"""
ik_jacobian_transpose.py

목적:
- Panda 로봇팔 End-Effector를 target_cube 근처로 이동시키는 기초 IK 실습.
- MuJoCo mj_jacBody로 End-Effector Jacobian을 계산한다.
- Jacobian transpose 방식으로 qpos를 조금씩 업데이트한다.
- 접근 과정을 MP4 영상과 CSV 로그로 저장한다.

주의:
- 이 코드는 교육용 IK 예제이다.
- 실제 로봇 actuator 제어가 아니라 qpos를 직접 갱신한다.
- 실제 Pick & Place에서는 joint limit, collision, orientation, gripper, controller를 함께 고려해야 한다.
"""

from pathlib import Path
import csv

import imageio.v2 as imageio
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


def get_first_camera_name(model: mujoco.MjModel) -> str | None:
    # 우리가 만든 scene_with_cube.xml에는 front_camera가 있다.
    front_camera_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_CAMERA,
        "front_camera"
    )

    if front_camera_id >= 0:
        return "front_camera"

    for camera_id in range(model.ncam):
        name = mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_CAMERA,
            camera_id
        )
        if name:
            return name

    return None


def clamp_qpos_to_joint_limits(model: mujoco.MjModel, data: mujoco.MjData):
    """
    hinge/slide joint 중 range 제한이 있는 joint에 대해 qpos를 제한한다.
    freejoint, ball joint는 여기서 단순 처리하지 않는다.
    """
    for joint_id in range(model.njnt):
        limited = model.jnt_limited[joint_id]

        if not limited:
            continue

        joint_type = model.jnt_type[joint_id]
        qpos_addr = model.jnt_qposadr[joint_id]

        # hinge 또는 slide joint는 qpos 1개를 사용한다.
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

    output_dir = project_root / "data" / "ik"
    output_dir.mkdir(parents=True, exist_ok=True)

    video_path = output_dir / "panda_ik_jacobian_transpose.mp4"
    csv_path = output_dir / "panda_ik_jacobian_transpose_log.csv"

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)

    ee_candidate_names = [
        "hand",
        "panda_hand",
        "link7",
        "attachment",
    ]

    ee_name, ee_body_id = find_first_existing_body(model, ee_candidate_names)

    cube_body_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        "target_cube"
    )

    if cube_body_id < 0:
        raise RuntimeError("target_cube body를 찾지 못했습니다.")

    renderer = mujoco.Renderer(model, height=480, width=640)
    camera_name = get_first_camera_name(model)

    print("Selected End-Effector body:", ee_name)
    print("Selected camera:", camera_name)
    print("model.nq:", model.nq)
    print("model.nv:", model.nv)

    # 초기 상태 계산
    mujoco.mj_forward(model, data)

    # 목표 위치:
    # 큐브 중심으로 바로 가면 충돌할 수 있으므로, 큐브 위쪽 15cm 지점을 목표로 둔다.
    cube_pos = data.xpos[cube_body_id].copy()
    target_pos = cube_pos + np.array([0.0, 0.0, 0.15])

    print("Cube position:", cube_pos)
    print("IK target position:", target_pos)

    # Jacobian transpose step 크기
    step_size = 0.08

    # 너무 큰 관절 변화 방지
    max_dq = 0.03

    max_iterations = 240
    tolerance = 0.03

    frames = []
    logs = []

    for iteration in range(max_iterations):
        mujoco.mj_forward(model, data)

        ee_pos = data.xpos[ee_body_id].copy()
        error = target_pos - ee_pos
        distance = float(np.linalg.norm(error))

        # 로그 저장
        row = {
            "iteration": iteration,
            "distance": distance,
            "ee_x": ee_pos[0],
            "ee_y": ee_pos[1],
            "ee_z": ee_pos[2],
            "target_x": target_pos[0],
            "target_y": target_pos[1],
            "target_z": target_pos[2],
            "error_x": error[0],
            "error_y": error[1],
            "error_z": error[2],
        }

        for i in range(min(10, model.nq)):
            row[f"qpos_{i}"] = data.qpos[i]

        logs.append(row)

        # 렌더링
        if camera_name:
            renderer.update_scene(data, camera=camera_name)
        else:
            renderer.update_scene(data)

        frames.append(renderer.render())

        if iteration % 20 == 0:
            print(f"iter={iteration:03d}, distance={distance:.4f}, ee_pos={ee_pos}")

        if distance < tolerance:
            print(f"Target reached. iteration={iteration}, distance={distance:.4f}")
            break

        # Jacobian 계산
        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))

        mujoco.mj_jacBody(model, data, jacp, jacr, ee_body_id)

        # 위치 제어만 하므로 jacp 사용
        # dqvel_space는 nv 차원의 변화 방향
        dqvel_space = step_size * jacp.T @ error

        # 너무 큰 업데이트 방지
        dqvel_space = np.clip(dqvel_space, -max_dq, max_dq)

        # qpos와 qvel 인덱스가 완전히 같다고 가정하면 위험하다.
        # 교육용으로 hinge/slide joint의 dof 주소와 qpos 주소를 이용해 qpos를 갱신한다.
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

    # 영상 저장
    imageio.mimsave(video_path, frames, fps=30)

    # CSV 저장
    if logs:
        fieldnames = list(logs[0].keys())

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(logs)

    renderer.close()

    print(f"Saved IK video: {video_path}")
    print(f"Saved IK CSV log: {csv_path}")
    print("Jacobian transpose IK completed successfully.")


if __name__ == "__main__":
    main()