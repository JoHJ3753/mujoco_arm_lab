"""
render_panda_with_cube.py

목적:
- Panda 로봇팔 + cube/sphere 장면을 렌더링한다.
- front_camera 기준 RGB 이미지와 Depth 이미지를 저장한다.
- 물체가 카메라에 보이는지 확인한다.
"""

from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np


def normalize_depth_to_uint8(depth: np.ndarray) -> np.ndarray:
    finite_depth = depth[np.isfinite(depth)]

    if finite_depth.size == 0:
        return np.zeros_like(depth, dtype=np.uint8)

    min_depth = finite_depth.min()
    max_depth = finite_depth.max()

    if max_depth - min_depth < 1e-8:
        return np.zeros_like(depth, dtype=np.uint8)

    normalized = (depth - min_depth) / (max_depth - min_depth)
    normalized = np.clip(normalized, 0.0, 1.0)

    return (normalized * 255).astype(np.uint8)


def main():
    project_root = Path(__file__).resolve().parents[1]

    xml_path = (
        project_root
        / "mujoco_menagerie"
        / "franka_emika_panda"
        / "scene_with_cube.xml"
    )

    output_dir = project_root / "data" / "panda_with_cube"
    output_dir.mkdir(parents=True, exist_ok=True)

    rgb_path = output_dir / "panda_with_cube_rgb.png"
    depth_path = output_dir / "panda_with_cube_depth.png"
    depth_raw_path = output_dir / "panda_with_cube_depth_raw.npy"

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)

    renderer = mujoco.Renderer(model, height=480, width=640)

    for _ in range(100):
        mujoco.mj_step(model, data)

    camera_name = "front_camera"

    # RGB 렌더링
    renderer.disable_depth_rendering()
    renderer.update_scene(data, camera=camera_name)
    rgb = renderer.render()

    imageio.imwrite(rgb_path, rgb)

    # Depth 렌더링
    renderer.enable_depth_rendering()
    renderer.update_scene(data, camera=camera_name)
    depth = renderer.render()

    np.save(depth_raw_path, depth)

    depth_uint8 = normalize_depth_to_uint8(depth)
    imageio.imwrite(depth_path, depth_uint8)

    renderer.close()

    print(f"Saved RGB image: {rgb_path}")
    print(f"Saved depth image: {depth_path}")
    print(f"Saved raw depth: {depth_raw_path}")
    print("Panda with cube rendering completed successfully.")


if __name__ == "__main__":
    main()