#!/usr/bin/env python3
import argparse
import glob
import json
import os

import matplotlib.pyplot as plt


def plot_file(path, ax_pos, ax_speed, ax_load, ax_cur):
    with open(path) as f:
        data = json.load(f)
    entries = data["entries"]
    t = [e["timestamp"] for e in entries]
    t0 = t[0]
    t = [x - t0 for x in t]
    label = os.path.basename(path)

    ax_pos.plot(t, [e["position"] for e in entries], label=f"{label} pos")
    ax_pos.plot(t, [e["goal_position"] for e in entries], "--", label=f"{label} goal", alpha=0.6)
    ax_speed.plot(t, [e["speed"] for e in entries], label=label)
    ax_load.plot(t, [e["load"] for e in entries], label=label)
    if "present_current" in entries[0]:
        ax_cur.plot(t, [e.get("present_current", 0) for e in entries], label=label)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="*", help="JSON files (default: all in script dir)")
    args = parser.parse_args()

    files = args.files or sorted(glob.glob(os.path.join(here, "*.json")))
    if not files:
        print("No JSON files found")
        return

    for path in files:
        fig, (ax_pos, ax_speed, ax_load, ax_cur) = plt.subplots(
            4, 1, sharex=True, figsize=(12, 9)
        )
        plot_file(path, ax_pos, ax_speed, ax_load, ax_cur)

        ax_pos.set_ylabel("position [rad]")
        ax_speed.set_ylabel("speed [rad/s]")
        ax_load.set_ylabel("load")
        ax_cur.set_ylabel("present_current")
        ax_cur.set_xlabel("time [s]")
        for ax in (ax_pos, ax_speed, ax_load, ax_cur):
            ax.grid(True)
            ax.legend(fontsize=7, loc="upper right")
        fig.suptitle(os.path.basename(path))
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
