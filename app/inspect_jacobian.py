"""
inspect_jacobian.py

목적:
- End-Effector body에 대한 translational / rotational Jacobian을 계산한다.
- jacp, jacr의 shape을 확인한다.
- Jacobian이 qvel 자유도 nv 기준으로 만들어진다는 것을 이해한다.

실무 포인트:
- jacp shape은 (3, model.nv)이다.
- jacr shape도 (3, model.nv)이다.
- 위치만 제어할 때는 우선 jacp를 사용한다.
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

    raise RuntimeError("End-Effector 후보 body를 찾지 못했습니다.")


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

    ee_candidate_names = [
        "hand",
        "panda_hand",
        "link7",
        "attachment",
    ]

    ee_name, ee_body_id = find_first_existing_body(model, ee_candidate_names)

    # 현재 상태 기준으로 위치/속도/파생값 계산
    mujoco.mj_forward(model, data)

    # MuJoCo Jacobian 배열 준비
    # jacp: position Jacobian
    # jacr: rotation Jacobian
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))

    # body의 현재 위치를 기준점으로 사용
    point = data.xpos[ee_body_id].copy()

    mujoco.mj_jacBody(model, data, jacp, jacr, ee_body_id)

    print("Selected End-Effector body:", ee_name)
    print("model.nq:", model.nq)
    print("model.nv:", model.nv)
    print("jacp shape:", jacp.shape)
    print("jacr shape:", jacr.shape)
    print("End-Effector point:", point)

    print("\nFirst 3 rows of jacp:")
    print(jacp)

    print("\nFirst 3 rows of jacr:")
    print(jacr)


if __name__ == "__main__":
    main()