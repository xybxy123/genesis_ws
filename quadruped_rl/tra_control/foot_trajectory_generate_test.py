import argparse
import time
from collections import deque

import matplotlib.pyplot as plt
import numpy as np

from foot_trajectory_generate import generate_foot_trajectory


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Real-time foot trajectory visualizer")
    parser.add_argument(
        "--phase-step", type=float, default=0.01, help="Phase increment per frame"
    )
    parser.add_argument("--dt", type=float, default=0.02, help="Seconds per frame")
    parser.add_argument(
        "--max-points", type=int, default=300, help="Max points shown on curves"
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="Run time in seconds, 0 means endless",
    )
    parser.add_argument("--step-length", type=float, default=0.1)
    parser.add_argument("--step-height", type=float, default=0.04)
    parser.add_argument("--z-base", type=float, default=-0.2)
    parser.add_argument("--side-offset", type=float, default=0.0465)
    return parser


def setup_figure(params: dict):
    fig = plt.figure(figsize=(12, 8))

    ax3d = fig.add_subplot(221, projection="3d")
    ax_x = fig.add_subplot(222)
    ax_y = fig.add_subplot(223)
    ax_z = fig.add_subplot(224)

    (line3d,) = ax3d.plot([], [], [], "b-", linewidth=2, label="foot")
    (line_x,) = ax_x.plot([], [], "r-", linewidth=2, label="x")
    (line_y,) = ax_y.plot([], [], "g-", linewidth=2, label="y")
    (line_z,) = ax_z.plot([], [], "m-", linewidth=2, label="z")

    ax3d.set_title("3D Foot Trajectory (Real-time)")
    ax3d.set_xlabel("X")
    ax3d.set_ylabel("Y")
    ax3d.set_zlabel("Z")
    ax3d.legend()
    ax3d.grid(True)

    x_span = max(params["step_length"], 1e-3)
    y_span = max(abs(params["side_offset"]) + 0.03, 0.05)
    z_span = max(params["step_height"] + 0.03, 0.05)

    ax3d.set_xlim(-x_span / 2.0, x_span / 2.0)
    ax3d.set_ylim(-y_span, y_span)
    ax3d.set_zlim(params["z_base"] - 0.02, params["z_base"] + z_span)

    for axis in (ax_x, ax_y, ax_z):
        axis.set_xlim(0.0, 1.0)
        axis.legend()
        axis.grid(True)

    ax_x.set_title("X vs Phase")
    ax_y.set_title("Y vs Phase")
    ax_z.set_title("Z vs Phase")

    fig.tight_layout()
    return fig, (ax3d, ax_x, ax_y, ax_z), (line3d, line_x, line_y, line_z)


def main() -> None:
    args = build_parser().parse_args()

    params = {
        "step_length": float(args.step_length),
        "step_height": float(args.step_height),
        "z_base": float(args.z_base),
        "side_offset": float(args.side_offset),
    }

    max_points = max(10, int(args.max_points))
    phase_step = max(1e-6, float(args.phase_step))
    dt = max(1e-4, float(args.dt))
    duration = max(0.0, float(args.duration))

    phase_data = deque(maxlen=max_points)
    x_data = deque(maxlen=max_points)
    y_data = deque(maxlen=max_points)
    z_data = deque(maxlen=max_points)

    plt.ion()
    fig, (_, ax_x, ax_y, ax_z), (line3d, line_x, line_y, line_z) = setup_figure(params)

    print("Real-time plot running. Press Ctrl+C to stop.")

    start_time = time.time()
    phase = 0.0

    try:
        while True:
            if duration > 0.0 and (time.time() - start_time) >= duration:
                break

            x, y, z = generate_foot_trajectory(phase, params)

            phase_data.append(float(phase))
            x_data.append(float(x))
            y_data.append(float(y))
            z_data.append(float(z))

            px = np.asarray(phase_data)
            xx = np.asarray(x_data)
            yy = np.asarray(y_data)
            zz = np.asarray(z_data)

            line3d.set_data(xx, yy)
            line3d.set_3d_properties(zz)

            line_x.set_data(px, xx)
            line_y.set_data(px, yy)
            line_z.set_data(px, zz)

            for axis in (ax_x, ax_y, ax_z):
                axis.relim()
                axis.autoscale_view(scalex=False, scaley=True)

            fig.canvas.draw_idle()
            fig.canvas.flush_events()

            time.sleep(dt)
            phase = (phase + phase_step) % 1.0

    except KeyboardInterrupt:
        pass
    finally:
        print("Plot stopped.")
        plt.ioff()
        plt.close(fig)


if __name__ == "__main__":
    main()
