"""
load_panda.py

목적:
- MuJoCo Menagerie의 Franka Emika Panda scene.xml 파일을 로드한다.
- viewer 없이 모델 구조만 확인한다.
- Docker 환경에서 GUI 문제 없이 로봇팔 모델 로딩 여부를 검증한다.
"""

from pathlib import Path
import mujoco


def main():
    # 현재 Python 파일 기준이 아니라, 프로젝트 루트 기준으로 XML 경로를 잡는다.
    # /workspace/app/load_panda.py 에서 실행하더라도
    # /workspace/mujoco_menagerie/... 를 안정적으로 찾기 위한 방식이다.
    project_root = Path(__file__).resolve().parents[1]

    xml_path = (
        project_root
        / "mujoco_menagerie"
        / "franka_emika_panda"
        / "scene.xml"
    )

    print("XML path:", xml_path)

    if not xml_path.exists():
        raise FileNotFoundError(
            f"scene.xml을 찾을 수 없습니다: {xml_path}\n"
            "mujoco_menagerie 저장소가 /workspace 아래에 clone 되었는지 확인하세요."
        )

    # XML 파일을 MuJoCo 모델로 로드
    model = mujoco.MjModel.from_xml_path(str(xml_path))

    # 모델 상태 데이터 생성
    data = mujoco.MjData(model)

    print("MuJoCo version:", mujoco.__version__)
    print("Number of bodies:", model.nbody)
    print("Number of joints:", model.njnt)
    print("Number of actuators:", model.nu)
    print("Number of sensors:", model.nsensor)
    print("Initial simulation time:", data.time)

    # 물리 step 100회 실행
    for _ in range(100):
        mujoco.mj_step(model, data)

    print("Final simulation time:", data.time)
    print("qpos shape:", data.qpos.shape)
    print("qvel shape:", data.qvel.shape)
    print("ctrl shape:", data.ctrl.shape)

    print("Panda model loaded successfully.")


if __name__ == "__main__":
    main()