"""
plot_position_control.py

목적:
- position_control_panda.py에서 저장한 CSV 로그를 읽는다.
- 시간에 따른 평균 절대 오차를 그래프로 저장한다.
- 목표 qpos와 현재 qpos를 비교하는 그래프를 저장한다.

실무 포인트:
- 제어 결과는 영상만 보면 안 된다.
- 오차가 줄어드는지 반드시 수치와 그래프로 확인해야 한다.
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

    csv_path = project_root / "data" / "panda_position_control_log.csv"
    error_plot_path = project_root / "data" / "panda_position_error_plot.png"
    qpos_plot_path = project_root / "data" / "panda_position_qpos_plot.png"

    if not csv_path.exists():
        raise FileNotFoundError(
            f"CSV 로그 파일을 찾을 수 없습니다: {csv_path}\n"
            "먼저 python app/position_control_panda.py를 실행하세요."
        )

    rows = read_csv(csv_path)

    times = [float(row["time"]) for row in rows]
    mean_abs_errors = [float(row["mean_abs_error"]) for row in rows]

    # 1. 평균 절대 오차 그래프
    plt.figure(figsize=(12, 6))
    plt.plot(times, mean_abs_errors, label="mean_abs_error")
    plt.title("Panda Position Control - Mean Absolute Error")
    plt.xlabel("Simulation Time")
    plt.ylabel("Mean Absolute Error")
    plt.legend()
    plt.grid(True)
    plt.savefig(error_plot_path, dpi=150)
    plt.close()

    # 2. qpos 0~2 목표값/현재값 비교 그래프
    plt.figure(figsize=(12, 6))

    for i in range(3):
        current_key = f"current_qpos_{i}"
        target_key = f"target_qpos_{i}"

        if current_key in rows[0] and target_key in rows[0]:
            current_values = [float(row[current_key]) for row in rows]
            target_values = [float(row[target_key]) for row in rows]

            plt.plot(times, current_values, label=f"current_qpos_{i}")
            plt.plot(times, target_values, linestyle="--", label=f"target_qpos_{i}")

    plt.title("Panda Position Control - Target vs Current Qpos")
    plt.xlabel("Simulation Time")
    plt.ylabel("Qpos")
    plt.legend()
    plt.grid(True)
    plt.savefig(qpos_plot_path, dpi=150)
    plt.close()

    print(f"Saved error plot: {error_plot_path}")
    print(f"Saved qpos plot: {qpos_plot_path}")


if __name__ == "__main__":
    main()