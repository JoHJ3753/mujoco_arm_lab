"""
inspect_ee_target.py

목적:
- scene_with_cube.xml을 로드한다.
- Panda 로봇팔의 End-Effector 후보 body를 찾는다.
- target_cube 위치를 찾는다.
- End-Effector와 target_cube 사이의 거리 벡터와 거리를 출력한다.

실무 포인트:
- IK를 하기 전에 end-effector body 이름과 target body 이름을 반드시 확인해야 한다.
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
        "End-Effector 후보 body를 찾지 못했습니다. "
        "먼저 python app/list_body_names.py를 실행해서 body 이름을 확인하세요."
    )


def main():
    project_root = Path(__file__).resolve().parents[1]

    xml_path = (
        project_root
        / "mujoco_menagerie"
        / "franka_emika_panda"
        / "scene_with_cube.xml"
    )

    if not xml_path.exists():
        raise FileNotFoundError(f"XML 파일을 찾을 수 없습니다: {xml_path}")

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)

    # Panda 모델 버전에 따라 end-effector 이름이 다를 수 있으므로 후보를 여러 개 둔다.
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

    # 현재 qpos 기준으로 위치 계산
    mujoco.mj_forward(model, data)

    ee_pos = data.xpos[ee_body_id].copy()
    cube_pos = data.xpos[cube_body_id].copy()

    error_vec = cube_pos - ee_pos
    distance = np.linalg.norm(error_vec)

    print("Selected End-Effector body:", ee_name)
    print("End-Effector body id:", ee_body_id)
    print("Cube body id:", cube_body_id)
    print("End-Effector position:", ee_pos)
    print("Cube position:", cube_pos)
    print("Error vector cube - ee:", error_vec)
    print("Distance:", distance)


if __name__ == "__main__":
    main()