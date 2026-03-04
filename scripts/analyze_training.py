#!/usr/bin/env python3
import argparse
import re
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def analyze(log_path: Path, out_path: Path) -> None:
    data = []
    with open(log_path, "r") as f:
        for line in f:
            match = re.search(
                r"Training\s+\[\s+(\d+)/(\d+)]\s+.*total_loss:\s+([\d.]+).*dino_local_crops_loss:\s+([\d.]+)",
                line,
            )
            if match:
                iteration = int(match.group(1))
                total = int(match.group(2))
                total_loss = float(match.group(3))
                local_loss = float(match.group(4))
                data.append(
                    {
                        "iteration": iteration,
                        "total": total,
                        "total_loss": total_loss,
                        "local_loss": local_loss,
                    }
                )

    print(f"Found {len(data)} training entries in {log_path}")
    if len(data) == 0:
        return

    print(f"Range: iteration {data[0]['iteration']} to {data[-1]['iteration']}")
    print(f"Total iterations: {data[-1]['total']}")

    iterations = [d["iteration"] for d in data]
    total_losses = [d["total_loss"] for d in data]

    plt.figure(figsize=(14, 8))

    plt.subplot(2, 2, 1)
    plt.plot(iterations, total_losses, "b-", alpha=0.3)
    plt.xlabel("Iteration")
    plt.ylabel("Total Loss")
    plt.title("Total Loss over Training")
    plt.grid(True, alpha=0.3)

    if len(iterations) > 100:
        window = 50
        moving_avg = np.convolve(total_losses, np.ones(window) / window, mode="valid")
        plt.plot(iterations[window - 1 :], moving_avg, "r-", linewidth=2, label="Moving Avg")

    plt.subplot(2, 2, 2)
    recent_losses = total_losses[-200:] if len(total_losses) > 200 else total_losses
    plt.plot(iterations[-len(recent_losses) :], recent_losses, "g-")
    plt.xlabel("Iteration")
    plt.ylabel("Total Loss")
    plt.title("Recent Iterations (Zoomed)")
    plt.grid(True, alpha=0.3)

    stats = {
        "mean": np.mean(total_losses),
        "std": np.std(total_losses),
        "min": np.min(total_losses),
        "max": np.max(total_losses),
        "recent_mean": np.mean(total_losses[-100:]),
        "recent_std": np.std(total_losses[-100:]),
    }

    print("\nStatistics:")
    print(f"  Overall mean: {stats['mean']:.4f} ± {stats['std']:.4f}")
    print(f"  Overall range: [{stats['min']:.4f}, {stats['max']:.4f}]")
    print(f"  Recent 100 iterations mean: {stats['recent_mean']:.4f} ± {stats['recent_std']:.4f}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"\nSaved analysis plot to: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze DINOv2 training log and save loss curves.")
    parser.add_argument(
        "log_file",
        type=str,
        help="训练日志文件路径（log.txt）",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="输出图片路径（默认与 log.txt 同目录，名为 training_analysis.png）",
    )
    args = parser.parse_args()

    log_path = Path(args.log_file).expanduser().resolve()
    if args.out is None:
        out_path = log_path.parent / "training_analysis.png"
    else:
        out_path = Path(args.out).expanduser().resolve()

    analyze(log_path, out_path)


if __name__ == "__main__":
    main()
