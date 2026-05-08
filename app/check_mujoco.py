"""
check_mujoco.py

목적:
- Docker 컨테이너 안에서 MuJoCo v3.6.0이 정상 설치되었는지 확인한다.
- 아주 작은 XML 모델을 문자열로 만들고, MuJoCo가 이를 로드할 수 있는지 테스트한다.
- viewer를 띄우지 않고 물리 step만 수행하므로 Docker 환경에서도 안정적으로 실행된다.
"""

import mujoco


def main():
    # 아주 단순한 MuJoCo XML 모델
    # worldbody 안에 바닥 plane과 공 sphere를 하나 배치한다.
    xml = """
    <mujoco model="simple_test">
      <option gravity="0 0 -9.81"/>

      <worldbody>
        <geom name="floor" type="plane" size="2 2 0.1" rgba="0.8 0.8 0.8 1"/>

        <body name="ball" pos="0 0 1">
          <joint name="free_joint" type="free"/>
          <geom name="ball_geom" type="sphere" size="0.1" mass="1.0" rgba="1 0 0 1"/>
        </body>
      </worldbody>
    </mujoco>
    """

    # XML 문자열을 MuJoCo 모델로 변환
    model = mujoco.MjModel.from_xml_string(xml)

    # 모델 상태 데이터를 생성
    data = mujoco.MjData(model)

    print("MuJoCo version:", mujoco.__version__)
    print("Model name:", model.names.decode(errors="ignore")[:50])
    print("Number of bodies:", model.nbody)
    print("Number of joints:", model.njnt)
    print("Initial simulation time:", data.time)

    # 100 step 시뮬레이션 실행
    for _ in range(100):
        mujoco.mj_step(model, data)

    print("Final simulation time:", data.time)
    print("Ball position qpos:", data.qpos)


if __name__ == "__main__":
    main()