import argparse, csv, os, socket, sys, time
from datetime import datetime

ACTIVITIES = ["WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS",
              "SITTING", "STANDING", "LAYING"]
GAP_MS = 30


class SerialLink:
    def __init__(self, name):
        import serial
        ser = serial.Serial()
        ser.port = name
        ser.baudrate = 115200
        ser.timeout = 1
        ser.dtr = False
        ser.rts = False
        ser.open()
        time.sleep(1.0)
        ser.reset_input_buffer()
        self.ser = ser

    def readline(self):
        return self.ser.readline().decode(errors="ignore").strip()

    def write(self, data):
        self.ser.write(data)

    def close(self):
        self.ser.close()


class WifiLink:
    def __init__(self, ip, port):
        self.sock = socket.create_connection((ip, port), timeout=10)
        self.sock.settimeout(1.0)
        self.buf = b""

    def readline(self):
        while b"\n" not in self.buf:
            try:
                chunk = self.sock.recv(4096)
            except socket.timeout:
                return ""
            if not chunk:
                raise ConnectionError("the board closed the connection")
            self.buf += chunk
        line, self.buf = self.buf.split(b"\n", 1)
        return line.decode(errors="ignore").strip()

    def write(self, data):
        self.sock.sendall(data)

    def close(self):
        try:
            self.sock.close()
        except OSError:
            pass


def main():
    ap = argparse.ArgumentParser()
    where = ap.add_mutually_exclusive_group(required=True)
    where.add_argument("--port", help="USB serial port, e.g. COM8")
    where.add_argument("--ip", help="board IP address for WiFi mode")
    ap.add_argument("--tcp-port", type=int, default=5000)
    ap.add_argument("--subject", type=int, required=True)
    ap.add_argument("--activity", required=True, choices=ACTIVITIES)
    ap.add_argument("--seconds", type=int, default=120)
    ap.add_argument("--placement", default="waist")
    ap.add_argument("--outdir", default="data/raw")
    args = ap.parse_args()

    try:
        link = SerialLink(args.port) if args.port else WifiLink(args.ip, args.tcp_port)
    except Exception as e:
        if args.ip:
            sys.exit(f"Could not reach {args.ip}:{args.tcp_port} ({e}).\n"
                     "Check: board powered, same WiFi as this laptop, IP typed correctly.")
        sys.exit(f"Could not open {args.port} ({e}). Is the Serial Monitor closed?")

    print(f"Subject {args.subject} | {args.activity} | {args.seconds} s | placement={args.placement}")
    input("Put the sensor in position, then press Enter... ")
    print("Put the sensor DOWN and let go. Starting in 3 s...")
    for i in (3, 2, 1):
        print(f"  {i}...")
        time.sleep(1)

    link.write(b"S")
    print("Calibrating gyro - hold STILL...")
    deadline = time.time() + 30
    while True:
        if time.time() > deadline:
            link.close()
            sys.exit("No GO from the board. Sketch uploaded? Calibration kept failing?")
        line = link.readline()
        if line.startswith("#"):
            print("  board:", line)
        if line == "# GO":
            break

    print(f"GO! Do {args.activity} now.")
    rows, last_t, gaps = [], None, 0
    next_report, last_rx = 10000, time.time()
    try:
        while True:
            line = link.readline()
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
    except (ConnectionError, OSError) as e:
        print("Connection lost:", e)

    try:
        link.write(b"X")
    except Exception:
        pass
    link.close()

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
    m = sum(mag) / len(mag)
    if not 0.9 < m < 1.1:
        print("WARNING: units look wrong - do NOT use this file, re-record")
    print("\n--- session summary ---")
    print(f"saved      : {path}")
    print(f"samples    : {len(rows)}  ({dur:.1f} s)")
    print(f"rate       : {rate:.1f} Hz   (target 50)")
    print(f"gaps >{GAP_MS}ms: {gaps}   (should be 0 or tiny)")
    print(f"mean |acc| : {sum(mag) / len(mag):.3f} g   (about 1.0 means units are right)")


if __name__ == "__main__":
    main()