"""
render_panda.py

목적:
- MuJoCo Menagerie의 Franka Emika Panda 로봇팔 scene.xml을 로드한다.
- Docker 환경에서 GUI viewer 없이 Headless 방식으로 렌더링한다.
- 단일 PNG 이미지와 MP4 영상을 data 폴더에 저장한다.

실무 포인트:
- Windows 경로 C:\\... 를 직접 쓰지 않는다.
- 컨테이너 내부 기준 /workspace 경로를 사용한다.
- XML만 복사하지 않고 franka_emika_panda 폴더 전체를 유지해야 한다.
"""

from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np


def find_camera_name(model: mujoco.MjModel) -> str | None:
    """
    모델 안에 정의된 카메라 이름을 하나 찾아 반환한다.
    카메라가 없으면 None을 반환한다.
    """
    camera_names = []

    for camera_id in range(model.ncam):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_id)
        if name:
            camera_names.append(name)

    print("Available cameras:", camera_names)

    if camera_names:
        return camera_names[0]

    return None


def main():
    project_root = Path(__file__).resolve().parents[1]

    xml_path = (
        project_root
        / "mujoco_menagerie"
        / "franka_emika_panda"
        / "scene.xml"
    )

    output_dir = project_root / "data"
    output_dir.mkdir(exist_ok=True)

    image_path = output_dir / "panda_render.png"
    video_path = output_dir / "panda_motion.mp4"

    if not xml_path.exists():
        raise FileNotFoundError(
            f"scene.xml을 찾을 수 없습니다: {xml_path}\n"
            "mujoco_menagerie/franka_emika_panda 폴더가 있는지 확인하세요."
        )

    print("XML path:", xml_path)

    # MuJoCo 모델과 데이터 생성
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)

    print("MuJoCo version:", mujoco.__version__)
    print("Number of cameras:", model.ncam)
    print("Number of joints:", model.njnt)
    print("Number of actuators:", model.nu)

    # 모델 안에 정의된 카메라 이름 확인
    camera_name = find_camera_name(model)

    # 렌더러 생성
    # 640x480은 수업용으로 적당한 해상도이다.
    renderer = mujoco.Renderer(model, height=480, width=640)

    # 초기 상태에서 몇 step 진행하여 자세를 안정화한다.
    for _ in range(100):
        mujoco.mj_step(model, data)

    # 단일 이미지 렌더링
    if camera_name:
        renderer.update_scene(data, camera=camera_name)
    else:
        renderer.update_scene(data)

    image = renderer.render()

    imageio.imwrite(image_path, image)
    print(f"Saved image: {image_path}")

    # 영상 렌더링
    frames = []

    fps = 30
    duration_sec = 5
    total_frames = fps * duration_sec

    # 관절 제어 입력이 있는 경우, 부드럽게 움직이도록 ctrl 값을 조금씩 바꾼다.
    # 아직 정교한 제어가 아니라 "움직이는 렌더링 확인" 목적이다.
    for frame_idx in range(total_frames):
        t = frame_idx / fps

        if model.nu > 0:
            # data.ctrl 길이는 model.nu와 같다.
            # 너무 큰 값을 넣으면 로봇이 튀거나 비정상적으로 움직일 수 있으므로 작은 값 사용.
            ctrl = 0.2 * np.sin(t)

            # 모든 actuator에 같은 값을 넣기보다는 앞쪽 몇 개만 약하게 움직인다.
            data.ctrl[:] = 0.0
            data.ctrl[: min(3, model.nu)] = ctrl

        # 한 프레임 사이에 여러 물리 step을 실행하면 움직임이 더 자연스럽다.
        for _ in range(5):
            mujoco.mj_step(model, data)

        if camera_name:
            renderer.update_scene(data, camera=camera_name)
        else:
            renderer.update_scene(data)

        frame = renderer.render()
        frames.append(frame)

    imageio.mimsave(video_path, frames, fps=fps)
    print(f"Saved video: {video_path}")

    renderer.close()

    print("Headless rendering completed successfully.")


if __name__ == "__main__":
    main()