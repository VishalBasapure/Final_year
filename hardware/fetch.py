"""Laptop side of the stored-mode logger (har_logger.ino).

Run from the hardware folder, with the board powered and on the same Wi-Fi.
Use the board's IP (shown in the Arduino Serial Monitor) or har-board.local.

  python fetch.py info --ip 192.168.1.3
  python fetch.py arm  --ip 192.168.1.3 --labels WALKING_UPSTAIRS,WALKING_DOWNSTAIRS --delay 20 --dur 12 --count 10
  python fetch.py list --ip 192.168.1.3
  python fetch.py get  --ip 192.168.1.3 --outdir data/s2 --trim-still --delete
  python fetch.py wipe --ip 192.168.1.3

arm   : tells the board to record by itself. After "ARMED" you can walk away.
        Takes cycle through --labels in order (UP, DOWN, UP, DOWN, ...).
        Each take = --settle s of waiting + ~2 s gyro calibration (stand still) + --dur s of recording.
get   : downloads every file, converts to the usual CSV format, checks it, optionally deletes it
        from the board (--delete). --trim-still cuts still seconds off the start/end of moving takes.
"""
import argparse
import datetime
import math
import os
import re
import socket
import struct
import sys

HDR = struct.Struct("<4sHHff3f24s")   # 52 bytes, must match the sketch
REC = struct.Struct("<H6h")            # 14 bytes
STATIC = {"SITTING", "STANDING", "LAYING"}
COLS = "t_ms,ax_g,ay_g,az_g,gx_rads,gy_rads,gz_rads"


class Board:
    def __init__(self, host, port=5000, timeout=10):
        try:
            self.s = socket.create_connection((host, port), timeout=timeout)
        except OSError as e:
            sys.exit(f"Could not reach {host}:{port} ({e}).\n"
                     "Check: board powered, laptop on the same Wi-Fi, IP typed correctly.")
        self.s.settimeout(timeout)
        self.buf = b""

    def _fill(self):
        try:
            data = self.s.recv(4096)
        except socket.timeout:
            raise TimeoutError("the board stopped answering")
        if not data:
            raise ConnectionError("the board closed the connection")
        self.buf += data

    def line(self):
        while b"\n" not in self.buf:
            self._fill()
        l, self.buf = self.buf.split(b"\n", 1)
        return l.decode(errors="ignore").strip()

    def exact(self, n):
        while len(self.buf) < n:
            self._fill()
        b, self.buf = self.buf[:n], self.buf[n:]
        return b

    def send(self, text):
        self.s.sendall(text.encode())

    def until(self, *prefixes):
        """Read lines until one starts with a given prefix. Returns (that line, all lines read)."""
        seen = []
        while True:
            l = self.line()
            seen.append(l)
            if any(l.startswith(p) for p in prefixes):
                return l, seen

    def close(self):
        try:
            self.s.close()
        except OSError:
            pass


def list_files(b):
    b.send("L\n")
    _, seen = b.until("# END")
    out = []
    for l in seen:
        m = re.match(r"# FILE (\S+) (\d+)", l)
        if m:
            out.append((m.group(1), int(m.group(2))))
    return sorted(out)


def download(b, name):
    b.send(f"G {name}\n")
    while True:
        l = b.line()
        if l.startswith("# ERROR"):
            raise RuntimeError(l)
        m = re.match(r"# SIZE (\d+)", l)
        if m:
            n = int(m.group(1))
            break
    data = b.exact(n)
    b.until("# END")
    return data


def parse(data):
    if len(data) < HDR.size:
        raise ValueError("file shorter than header")
    magic, hz, n, acc_lsb, gyr_lsb, gb0, gb1, gb2, label = HDR.unpack_from(data, 0)
    if magic != b"HAR1":
        raise ValueError("bad magic (not a HAR1 file)")
    if len(data) != HDR.size + n * REC.size:
        raise ValueError(f"size mismatch: header says {n} samples but file has "
                         f"{(len(data) - HDR.size) / REC.size:.1f}")
    label = label.split(b"\0")[0].decode()
    rows = []
    for i in range(n):
        t, ax, ay, az, gx, gy, gz = REC.unpack_from(data, HDR.size + i * REC.size)
        k = math.pi / 180.0 / gyr_lsb
        rows.append((t, ax / acc_lsb, ay / acc_lsb, az / acc_lsb,
                     gx * k - gb0, gy * k - gb1, gz * k - gb2))
    return label, hz, rows


def trim_still(rows, thr=0.05, margin_s=0.3, hz=50):
    """Cut still seconds off both ends. Returns (rows, cut_front_s, cut_back_s)."""
    n = len(rows)
    norm = [math.sqrt(r[1] ** 2 + r[2] ** 2 + r[3] ** 2) for r in rows]
    w = hz // 2
    moving = []
    for i in range(n):
        lo, hi = max(0, i - w // 2), min(n, i + w // 2 + 1)
        seg = norm[lo:hi]
        mu = sum(seg) / len(seg)
        sd = math.sqrt(sum((x - mu) ** 2 for x in seg) / len(seg))
        moving.append(sd > thr)
    if True not in moving:
        return rows, 0.0, 0.0
    first = moving.index(True)
    last = n - 1 - moving[::-1].index(True)
    m = int(margin_s * hz)
    a, z = max(0, first - m), min(n, last + m + 1)
    return rows[a:z], a / hz, (n - z) / hz


def summarize(label, rows, hz):
    n = len(rows)
    dur = (rows[-1][0] - rows[0][0]) / 1000.0 if n > 1 else 0.0
    rate = (n - 1) / dur if dur > 0 else 0.0
    mag = [math.sqrt(r[1] ** 2 + r[2] ** 2 + r[3] ** 2) for r in rows]
    mean = sum(mag) / len(mag) if mag else 0.0
    sd = math.sqrt(sum((x - mean) ** 2 for x in mag) / len(mag)) if mag else 0.0
    warns = []
    if not 0.9 < mean < 1.1:
        warns.append(f"UNITS: mean |acc| {mean:.2f} g")
    if abs(rate - hz) > 1.5:
        warns.append(f"RATE {rate:.1f} Hz")
    if label in STATIC and sd > 0.05:
        warns.append(f"moved a lot for a static activity (std {sd:.3f})")
    if label not in STATIC and sd < 0.1:
        warns.append(f"barely moved (std {sd:.3f}) - did you really do it?")
    if dur < 3:
        warns.append(f"short: {dur:.1f} s")
    return dur, rate, mean, sd, warns


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        f.write(COLS + "\n")
        t0 = rows[0][0]
        for r in rows:
            f.write(f"{r[0] - t0},{r[1]:.4f},{r[2]:.4f},{r[3]:.4f},{r[4]:.5f},{r[5]:.5f},{r[6]:.5f}\n")


def cmd_get(b, a):
    files = list_files(b)
    if not files:
        print("No files on the board.")
        return
    os.makedirs(a.outdir, exist_ok=True)
    now = datetime.datetime.now()
    date, hhmm = now.strftime("%Y%m%d"), now.strftime("%H%M")
    ok_names = []
    print(f"{len(files)} file(s) on the board.")
    for name, size in files:
        try:
            data = download(b, name)
            label, hz, rows = parse(data)
        except (RuntimeError, ValueError) as e:
            print(f"SKIP  {name}: {e}")
            continue
        note = ""
        if a.trim_still and label not in STATIC:
            rows2, cf, cb = trim_still(rows, hz=hz)
            if len(rows2) >= 3 * hz:
                rows, note = rows2, f" (trimmed {cf:.1f}s front, {cb:.1f}s back)"
            else:
                note = " (trim would leave under 3 s, kept whole take)"
        m = re.match(r"(\d+)_", name)
        idn = int(m.group(1)) % 100 if m else 0
        k = 0
        while True:
            out = os.path.join(a.outdir, f"s{a.subject:02d}_{label}_{a.place}_{date}_{hhmm}{(idn + k) % 100:02d}.csv")
            if not os.path.exists(out):
                break
            k += 1
        write_csv(out, rows)
        dur, rate, mean, sd, warns = summarize(label, rows, hz)
        print(("WARN  " if warns else "OK    ") +
              f"{name} -> {os.path.basename(out)}  {len(rows)} samples {dur:.1f}s {rate:.1f}Hz |acc|={mean:.3f}{note}")
        for w in warns:
            print("      -", w)
        ok_names.append(name)
    if a.delete and ok_names:
        for name in ok_names:
            b.send(f"D {name}\n")
            b.until("# DELETED", "# ERROR")
        print(f"Deleted {len(ok_names)} file(s) from the board.")
    elif ok_names:
        print("Files kept on the board. Re-run with --delete to free space after you have checked them.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["info", "arm", "list", "get", "wipe"])
    ap.add_argument("--ip", required=True)
    ap.add_argument("--tcp-port", type=int, default=5000)
    ap.add_argument("--labels", default="WALKING_UPSTAIRS,WALKING_DOWNSTAIRS")
    ap.add_argument("--delay", type=int, default=20, help="seconds before the first take")
    ap.add_argument("--dur", type=int, default=12, help="seconds per take (3-60)")
    ap.add_argument("--count", type=int, default=10, help="number of takes")
    ap.add_argument("--settle", type=int, default=2, help="seconds of waiting before each later take")
    ap.add_argument("--outdir", default="data/s2")
    ap.add_argument("--subject", type=int, default=1)
    ap.add_argument("--place", default="waist")
    ap.add_argument("--trim-still", action="store_true")
    ap.add_argument("--delete", action="store_true")
    a = ap.parse_args()

    b = Board(a.ip, a.tcp_port)
    try:
        if a.action == "info":
            b.send("I\n")
            l, _ = b.until("# INFO")
            print(l)
        elif a.action == "list":
            for name, size in list_files(b):
                print(f"{name:40s} {size:8d} bytes")
        elif a.action == "get":
            cmd_get(b, a)
        elif a.action == "wipe":
            if input("Delete ALL files on the board? type yes: ").strip().lower() == "yes":
                b.send("D *\n")
                l, _ = b.until("# DELETED", "# ERROR")
                print(l)
        elif a.action == "arm":
            labels = a.labels.replace(" ", "")
            b.send(f"A {labels} {a.delay} {a.dur} {a.count} {a.settle}\n")
            l, _ = b.until("# ARMED", "# ERROR")
            print(l)
            if l.startswith("# ARMED"):
                per = a.settle + 2 + a.dur
                print(f"First take starts in {a.delay} s. Each take is about {per} s "
                      f"(wait {a.settle}, stand still ~2, record {a.dur}). "
                      f"Whole job about {a.delay + per * a.count} s.")
                print("You can disconnect now. Collect the files later with: fetch.py get")
    except (ConnectionError, TimeoutError) as e:
        sys.exit(f"Lost the board: {e}")
    finally:
        b.close()


if __name__ == "__main__":
    main()