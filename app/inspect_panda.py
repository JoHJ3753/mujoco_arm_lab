"""
inspect_panda.py

목적:
- MuJoCo Menagerie의 Franka Panda 모델을 로드한다.
- joint 이름, actuator 이름, camera 이름을 출력한다.
- qpos, qvel, ctrl 배열의 크기를 확인한다.

실무 포인트:
- 로봇을 제어하기 전에 반드시 joint와 actuator 구조를 먼저 확인해야 한다.
- actuator 이름과 joint 이름은 항상 1:1로 단순히 대응된다고 가정하면 안 된다.
"""

from pathlib import Path
import mujoco


def print_joint_info(model: mujoco.MjModel):
    print("\n=== Joint Info ===")
    print("Number of joints:", model.njnt)
    print("Number of qpos:", model.nq)
    print("Number of qvel:", model.nv)

    for joint_id in range(model.njnt):
        name = mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            joint_id
        )

        joint_type = model.jnt_type[joint_id]
        qpos_addr = model.jnt_qposadr[joint_id]
        qvel_addr = model.jnt_dofadr[joint_id]

        print(
            f"joint_id={joint_id:2d}, "
            f"name={name}, "
            f"type={joint_type}, "
            f"qpos_addr={qpos_addr}, "
            f"qvel_addr={qvel_addr}"
        )


def print_actuator_info(model: mujoco.MjModel):
    print("\n=== Actuator Info ===")
    print("Number of actuators:", model.nu)

    for actuator_id in range(model.nu):
        name = mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            actuator_id
        )

        ctrl_min = model.actuator_ctrlrange[actuator_id][0]
        ctrl_max = model.actuator_ctrlrange[actuator_id][1]

        print(
            f"actuator_id={actuator_id:2d}, "
            f"name={name}, "
            f"ctrl_range=[{ctrl_min:.3f}, {ctrl_max:.3f}]"
        )


def print_camera_info(model: mujoco.MjModel):
    print("\n=== Camera Info ===")
    print("Number of cameras:", model.ncam)

    for camera_id in range(model.ncam):
        name = mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_CAMERA,
            camera_id
        )
        print(f"camera_id={camera_id:2d}, name={name}")


def main():
    project_root = Path(__file__).resolve().parents[1]

    xml_path = (
        project_root
        / "mujoco_menagerie"
        / "franka_emika_panda"
        / "scene.xml"
    )

    if not xml_path.exists():
        raise FileNotFoundError(f"XML 파일을 찾을 수 없습니다: {xml_path}")

    print("XML path:", xml_path)

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)

    print("\n=== Basic Model Info ===")
    print("MuJoCo version:", mujoco.__version__)
    print("model.nbody:", model.nbody)
    print("model.njnt:", model.njnt)
    print("model.nq:", model.nq)
    print("model.nv:", model.nv)
    print("model.nu:", model.nu)
    print("data.qpos shape:", data.qpos.shape)
    print("data.qvel shape:", data.qvel.shape)
    print("data.ctrl shape:", data.ctrl.shape)

    print_joint_info(model)
    print_actuator_info(model)
    print_camera_info(model)


if __name__ == "__main__":
    main()