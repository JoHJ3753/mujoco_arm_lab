"""
list_body_names.py

목적:
- scene_with_cube.xml 안의 모든 body 이름을 출력한다.
- end-effector 후보 body 이름을 찾기 위해 사용한다.
"""

from pathlib import Path
import mujoco


def main():
    project_root = Path(__file__).resolve().parents[1]

    xml_path = (
        project_root
        / "mujoco_menagerie"
        / "franka_emika_panda"
        / "scene_with_cube.xml"
    )

    model = mujoco.MjModel.from_xml_path(str(xml_path))

    print("=== Body Names ===")

    for body_id in range(model.nbody):
        name = mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            body_id
        )
        print(f"body_id={body_id:2d}, name={name}")


if __name__ == "__main__":
    main()