"""
logger.py - record ONE labelled session from the ESP32-S3 + MPU6050.

Example:
    python logger.py --port COM8 --subject 1 --activity WALKING --seconds 120

Close the Arduino Serial Monitor first (only one program can use the port).
Needs:  pip install pyserial
"""
import argparse
import csv
import os
import sys
import time
from datetime import datetime

import serial

ACTIVITIES = ["WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS",
              "SITTING", "STANDING", "LAYING"]
GAP_MS = 30  # nominal gap is 20 ms; more than 30 ms means we lost a sample


def open_port(name):
    ser = serial.Serial()
    ser.port = name
    ser.baudrate = 115200
    ser.timeout = 1
    ser.dtr = False   # stops the ESP32-S3 from rebooting when the port opens
    ser.rts = False
    ser.open()
    time.sleep(1.0) 
    ser.reset_input_buffer()
    return ser


def read_line(ser):
    return ser.readline().decode(errors="ignore").strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", required=True, help="e.g. COM8")
    ap.add_argument("--subject", type=int, required=True)
    ap.add_argument("--activity", required=True, choices=ACTIVITIES)
    ap.add_argument("--seconds", type=int, default=120)
    ap.add_argument("--placement", default="waist",
                    help="tag for where the sensor is worn, used later for drift analysis")
    ap.add_argument("--outdir", default="data/raw")
    args = ap.parse_args()

    ser = open_port(args.port)
    print(f"Subject {args.subject} | {args.activity} | {args.seconds} s | placement={args.placement}")
    input("Put the sensor in position, then press Enter. Stay STILL for 2 s after that... ")
    print("Put the sensor DOWN and let go. Starting in 3 s...")
    for i in (3, 2, 1):
        print(f"  {i}...")
        time.sleep(1)
    ser.write(b"S")
    print("Calibrating gyro - hold STILL...")
    deadline = time.time() + 10
    while True:
        if time.time() > deadline:
            ser.close()
            sys.exit("No reply from the board. Is the logger sketch uploaded and the Serial Monitor closed?")
        line = read_line(ser)
        if line.startswith("#"):
            print("  board:", line)
        if line == "# GO":
            break

    print(f"GO! Do {args.activity} now.")
    rows = []
    last_t = None
    gaps = 0
    next_report = 10000
    last_rx = time.time()
    try:
        while True:
            line = read_line(ser)
            if not line:
                if time.time() - last_rx > 5:
                    print("No data for 5 s - stopping.")
                    break
                continue
            last_rx = time.time()
            if line.startswith("#"):
                print("  board:", line)
                continue
            parts = line.split(",")
            if len(parts) != 7:
                continue
            try:
                vals = [float(p) for p in parts]
            except ValueError:
                continue
            t = vals[0]
            if last_t is not None and t - last_t > GAP_MS:
                gaps += 1
            last_t = t
            rows.append(vals)
            if t >= next_report:
                print(f"  {int(t / 1000)} s ...")
                next_report += 10000
            if t >= args.seconds * 1000:
                break
    except KeyboardInterrupt:
        print("Stopped early.")

    ser.write(b"X")
    ser.close()

    if len(rows) < 2:
        sys.exit("Almost no data received - nothing saved.")

    os.makedirs(args.outdir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(args.outdir,
                        f"s{args.subject:02d}_{args.activity}_{args.placement}_{stamp}.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_ms", "ax_g", "ay_g", "az_g", "gx_rads", "gy_rads", "gz_rads"])
        w.writerows(rows)

    dur = (rows[-1][0] - rows[0][0]) / 1000.0
    rate = (len(rows) - 1) / dur if dur > 0 else 0.0
    mag = [(r[1] ** 2 + r[2] ** 2 + r[3] ** 2) ** 0.5 for r in rows]
    print("\n--- session summary ---")
    print(f"saved      : {path}")
    print(f"samples    : {len(rows)}  ({dur:.1f} s)")
    print(f"rate       : {rate:.1f} Hz   (target 50)")
    print(f"gaps >{GAP_MS}ms: {gaps}   (should be 0 or tiny)")
    print(f"mean |acc| : {sum(mag) / len(mag):.3f} g   (about 1.0 means units are right)")


if __name__ == "__main__":
    main()