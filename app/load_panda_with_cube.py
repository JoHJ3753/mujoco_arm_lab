"""
load_panda_with_cube.py

목적:
- custom_models/panda_with_object/scene_with_cube.xml 파일을 로드한다.
- target_cube와 target_sphere body가 모델에 포함되었는지 확인한다.
- viewer 없이 물리 step만 실행해 XML 구조가 정상인지 검증한다.
"""

from pathlib import Path
import mujoco


def get_body_id(model: mujoco.MjModel, body_name: str) -> int:
    body_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        body_name
    )
    return body_id


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

    print("XML path:", xml_path)

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)

    print("MuJoCo version:", mujoco.__version__)
    print("model.nbody:", model.nbody)
    print("model.njnt:", model.njnt)
    print("model.nq:", model.nq)
    print("model.nv:", model.nv)
    print("model.nu:", model.nu)
    print("model.ncam:", model.ncam)

    cube_body_id = get_body_id(model, "target_cube")
    sphere_body_id = get_body_id(model, "target_sphere")

    print("target_cube body id:", cube_body_id)
    print("target_sphere body id:", sphere_body_id)

    if cube_body_id < 0:
        raise RuntimeError("target_cube body를 찾지 못했습니다.")
    if sphere_body_id < 0:
        raise RuntimeError("target_sphere body를 찾지 못했습니다.")

    for _ in range(100):
        mujoco.mj_step(model, data)

    cube_pos = data.xpos[cube_body_id]
    sphere_pos = data.xpos[sphere_body_id]

    print("target_cube position:", cube_pos)
    print("target_sphere position:", sphere_pos)

    print("Panda with cube scene loaded successfully.")


if __name__ == "__main__":
    main()