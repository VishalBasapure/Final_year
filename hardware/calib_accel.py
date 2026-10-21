import argparse, collections, json, sys, threading, time
import numpy as np
from scipy.optimize import least_squares
from logger import SerialLink, WifiLink


def fit(means):
    m = np.asarray(means, float)

    def resid(p):
        return np.linalg.norm((m - p[:3]) / p[3:], axis=1) - 1.0

    start = np.r_[np.zeros(3), np.ones(3)]
    sol = least_squares(resid, x0=start)
    return sol.x[:3], sol.x[3:], resid(start), sol.fun


def coverage_problems(means):
    if len(means) == 0:
        return ["all directions"]
    m = np.asarray(means)
    out = []
    for i, name in enumerate("XYZ"):
        if m[:, i].max() < 0.7:
            out.append(f"+{name} up")
        if m[:, i].min() > -0.7:
            out.append(f"-{name} up")
    return out


def main():
    ap = argparse.ArgumentParser()
    where = ap.add_mutually_exclusive_group(required=True)
    where.add_argument("--port")
    where.add_argument("--ip")
    ap.add_argument("--tcp-port", type=int, default=5000)
    ap.add_argument("--count", type=int, default=12)
    ap.add_argument("--hold", type=float, default=3.0)
    ap.add_argument("--out", default="accel_cal.json")
    args = ap.parse_args()

    try:
        link = SerialLink(args.port) if args.port else WifiLink(args.ip, args.tcp_port)
    except Exception as e:
        sys.exit(f"Could not connect ({e}).")

    link.write(b"S")
    print("Keep the sensor STILL for 2 s while the board starts up...")
    deadline = time.time() + 30
    while True:
        if time.time() > deadline:
            link.close()
            sys.exit("No GO from the board.")
        line = link.readline()
        if line.startswith("#"):
            print("  board:", line)
        if line == "# GO":
            break

    need = int(args.hold * 50)
    buf = collections.deque(maxlen=need)
    stop = threading.Event()

    def reader():
        while not stop.is_set():
            try:
                line = link.readline()
            except Exception:
                return
            p = line.split(",")
            if len(p) != 7:
                continue
            try:
                v = [float(x) for x in p]
            except ValueError:
                continue
            buf.append(v[1:4])

    threading.Thread(target=reader, daemon=True).start()

    means = []
    try:
        while len(means) < args.count or coverage_problems(means):
            k = len(means) + 1
            msg = f"\n[{k}] Put the sensor still in a NEW orientation"
            if len(means) >= args.count:
                msg += f" (still needed: {', '.join(coverage_problems(means))})"
            input(msg + ", then press Enter and let go... ")
            buf.clear()
            time.sleep(args.hold + 0.2)
            data = np.array(list(buf))
            if len(data) < need * 0.9:
                print("  not enough data - is the board streaming? try again.")
                continue
            mag = np.linalg.norm(data, axis=1)
            if mag.std() > 0.02:
                print(f"  it moved (noise {mag.std():.3f} g) - hold it still and try again.")
                continue
            mean = data.mean(axis=0)
            if any(np.linalg.norm(mean - m) < 0.25 for m in means):
                print("  same as an earlier orientation - turn it to a different one.")
                continue
            means.append(mean)
            print(f"  ok: x={mean[0]:+.3f} y={mean[1]:+.3f} z={mean[2]:+.3f}   |a|={np.linalg.norm(mean):.3f} g")
    except KeyboardInterrupt:
        print("\nStopped early - nothing saved.")
        stop.set()
        link.close()
        return

    stop.set()
    try:
        link.write(b"X")
    except Exception:
        pass
    link.close()

    offset, gain, before, after = fit(means)
    rms = lambda e: float(np.sqrt(np.mean(e ** 2))) * 100
    print("\n--- result ---")
    print(f"gravity error before: RMS {rms(before):.2f} %   worst {np.abs(before).max() * 100:.1f} %")
    print(f"gravity error after : RMS {rms(after):.2f} %   worst {np.abs(after).max() * 100:.1f} %")
    print("offset (g):", np.round(offset, 4))
    print("gain      :", np.round(gain, 4))
    if rms(after) > 0.6:
        print("WARNING: error after is still above 0.6 %. Some positions probably moved - run again.")
    with open(args.out, "w") as f:
        json.dump({"offset_g": offset.tolist(), "gain": gain.tolist(),
                   "n_orientations": len(means),
                   "rms_error_after_pct": rms(after)}, f, indent=2)
    print("saved:", args.out)a


if __name__ == "__main__":
    main()