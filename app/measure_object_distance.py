"""
measure_object_distance.py

목적:
- Panda 로봇팔 끝부분과 target_cube 사이 거리를 계산한다.
- body 이름이 모델 버전에 따라 다를 수 있으므로 end-effector 후보 이름을 순서대로 탐색한다.

실무 포인트:
- 로봇팔 제어, Pick & Place, 접근 판단에서는 거리 계산이 매우 중요하다.
- body 이름을 하드코딩하기 전에 반드시 list_body_names.py로 확인한다.
"""

from pathlib import Path

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
        "end-effector 후보 body를 찾지 못했습니다. "
        "먼저 python app/list_body_names.py로 body 이름을 확인하세요."
    )


def main():
    project_root = Path(__file__).resolve().parents[1]

    xml_path = (
        project_root
        / "mujoco_menagerie"
        / "franka_emika_panda"
        / "scene_with_cube.xml"
    )

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)

    # 모델 버전에 따라 이름이 다를 수 있으므로 후보를 여러 개 둔다.
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

    sphere_body_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        "target_sphere"
    )

    if cube_body_id < 0:
        raise RuntimeError("target_cube body를 찾지 못했습니다.")

    if sphere_body_id < 0:
        raise RuntimeError("target_sphere body를 찾지 못했습니다.")

    for _ in range(100):
        mujoco.mj_step(model, data)

    ee_pos = data.xpos[ee_body_id].copy()
    cube_pos = data.xpos[cube_body_id].copy()
    sphere_pos = data.xpos[sphere_body_id].copy()

    cube_distance = np.linalg.norm(ee_pos - cube_pos)
    sphere_distance = np.linalg.norm(ee_pos - sphere_pos)

    print("Selected end-effector body:", ee_name)
    print("End-effector position:", ee_pos)
    print("Cube position:", cube_pos)
    print("Sphere position:", sphere_pos)
    print("Distance to cube:", cube_distance)
    print("Distance to sphere:", sphere_distance)


if __name__ == "__main__":
    main()