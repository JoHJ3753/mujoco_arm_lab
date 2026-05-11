"""
plot_panda_log.py

목적:
- control_panda_joint.py에서 저장한 CSV 로그를 읽는다.
- 시간에 따른 제어 입력과 qpos 변화를 그래프로 저장한다.

실무 포인트:
- 로봇 제어는 영상만 보는 것으로 충분하지 않다.
- 반드시 수치 로그와 그래프를 함께 확인해야 한다.
"""

from pathlib import Path
import csv

import matplotlib.pyplot as plt


def read_csv(csv_path: Path):
    rows = []

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            rows.append(row)

    return rows


def main():
    project_root = Path(__file__).resolve().parents[1]

    csv_path = project_root / "data" / "panda_joint_log.csv"
    output_path = project_root / "data" / "panda_joint_plot.png"

    if not csv_path.exists():
        raise FileNotFoundError(
            f"CSV 로그 파일을 찾을 수 없습니다: {csv_path}\n"
            "먼저 python app/control_panda_joint.py를 실행하세요."
        )

    rows = read_csv(csv_path)

    times = [float(row["time"]) for row in rows]
    commands = [float(row["command"]) for row in rows]

    # qpos_0, qpos_1, qpos_2만 우선 확인
    qpos_0 = [float(row.get("qpos_0", 0.0)) for row in rows]
    qpos_1 = [float(row.get("qpos_1", 0.0)) for row in rows]
    qpos_2 = [float(row.get("qpos_2", 0.0)) for row in rows]

    plt.figure(figsize=(12, 6))

    plt.plot(times, commands, label="command")
    plt.plot(times, qpos_0, label="qpos_0")
    plt.plot(times, qpos_1, label="qpos_1")
    plt.plot(times, qpos_2, label="qpos_2")

    plt.title("Panda Joint Control Log")
    plt.xlabel("Simulation Time")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)

    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"Saved plot: {output_path}")


if __name__ == "__main__":
    main()