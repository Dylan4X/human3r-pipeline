import cv2
import socket
import struct
import time
import threading
import json
import math

SERVER_HOST = "127.0.0.1"
SERVER_PORT = 9999

CAM_ID = 0

DISPLAY_WIDTH = 640
DISPLAY_HEIGHT = 480

SEND_WIDTH = 320
SEND_HEIGHT = 240

TARGET_SEND_FPS = 5
JPEG_QUALITY = 70

latest_pose = {"persons": []}
latest_lock = threading.Lock()


def draw_text_block(img, lines, x=10, y=28, dy=26):
    for i, line in enumerate(lines):
        yy = y + i * dy
        cv2.putText(
            img,
            line,
            (x, yy),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )


def fmt_vec(name, v):
    if v is None:
        return f"{name}: None"
    return f"{name}: [{v[0]:.2f}, {v[1]:.2f}, {v[2]:.2f}]"


def draw_mini_stickman(img, person, x0=470, y0=60, w=140, h=220):
    root = person.get("root_world")
    head = person.get("head_world")
    lw = person.get("left_wrist_world")
    rw = person.get("right_wrist_world")
    la = person.get("left_ankle_world")
    ra = person.get("right_ankle_world")

    pts3 = {
        "root": root,
        "head": head,
        "lw": lw,
        "rw": rw,
        "la": la,
        "ra": ra,
    }

    valid = {k: v for k, v in pts3.items() if v is not None}
    if len(valid) < 2:
        cv2.rectangle(img, (x0, y0), (x0 + w, y0 + h), (100, 100, 100), 1)
        cv2.putText(img, "mini smpl: none", (x0, y0 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)
        return

    xs = [v[0] for v in valid.values()]
    ys = [v[1] for v in valid.values()]

    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)

    if abs(maxx - minx) < 1e-5:
        maxx += 1e-5
    if abs(maxy - miny) < 1e-5:
        maxy += 1e-5

    pad = 10
    sx = (w - 2 * pad) / (maxx - minx)
    sy = (h - 2 * pad) / (maxy - miny)
    s = min(sx, sy)

    def proj(p):
        if p is None:
            return None
        xx = x0 + pad + int((p[0] - minx) * s)
        yy = y0 + h - pad - int((p[1] - miny) * s)
        return (xx, yy)

    p2 = {k: proj(v) for k, v in pts3.items()}

    cv2.rectangle(img, (x0, y0), (x0 + w, y0 + h), (120, 120, 120), 1)
    cv2.putText(img, "mini smpl", (x0, y0 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 255), 1, cv2.LINE_AA)

    bones = [
        ("head", "root"),
        ("lw", "root"),
        ("rw", "root"),
        ("la", "root"),
        ("ra", "root"),
    ]

    for a, b in bones:
        if p2[a] is not None and p2[b] is not None:
            cv2.line(img, p2[a], p2[b], (0, 220, 255), 2, cv2.LINE_AA)

    for name, pt in p2.items():
        if pt is not None:
            cv2.circle(img, pt, 4, (0, 255, 0), -1, cv2.LINE_AA)
            cv2.putText(img, name, (pt[0] + 4, pt[1] - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)


sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect((SERVER_HOST, SERVER_PORT))
print(f"Connected to {SERVER_HOST}:{SERVER_PORT}")


def recv_loop():
    global latest_pose
    buf = b""
    while True:
        try:
            chunk = sock.recv(4096)
            if not chunk:
                print("server closed")
                break
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                try:
                    obj = json.loads(line.decode("utf-8"))
                    with latest_lock:
                        latest_pose = obj
                except Exception as e:
                    print("json parse error:", e)
        except Exception as e:
            print("recv error:", e)
            break


threading.Thread(target=recv_loop, daemon=True).start()

cap = cv2.VideoCapture(CAM_ID)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, DISPLAY_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DISPLAY_HEIGHT)

if not cap.isOpened():
    raise RuntimeError("Cannot open webcam")

frame_interval = 1.0 / TARGET_SEND_FPS
last_send = 0.0
sent = 0

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        now = time.time()
        if now - last_send >= frame_interval:
            last_send = now
            send_frame = cv2.resize(frame, (SEND_WIDTH, SEND_HEIGHT))
            ok, enc = cv2.imencode(
                ".jpg", send_frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
            )
            if ok:
                data = enc.tobytes()
                header = struct.pack("!I", len(data))
                sock.sendall(header + data)
                sent += 1

        vis = frame.copy()

        with latest_lock:
            pose = latest_pose.copy()

        lines = [
            f"sent={sent}",
            f"send_fps={TARGET_SEND_FPS}",
            f"send_size={SEND_WIDTH}x{SEND_HEIGHT}",
        ]

        persons = pose.get("persons", [])
        if len(persons) == 0:
            lines.append("person: none")
        else:
            p = persons[0]
            lines.append(f"person_id: {p.get('id', -1)}")
            lines.append(fmt_vec("root", p.get("root_world")))
            lines.append(fmt_vec("head", p.get("head_world")))
            lines.append(fmt_vec("lwrist", p.get("left_wrist_world")))
            lines.append(fmt_vec("rwrist", p.get("right_wrist_world")))
            if "server_latency_sec" in pose:
                lines.append(f"server_latency: {pose['server_latency_sec']:.2f}s")

            draw_mini_stickman(vis, p, x0=470, y0=60, w=140, h=220)

        draw_text_block(vis, lines, x=10, y=28, dy=26)

        cv2.imshow("Human3R Stream Visualization", vis)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    sock.close()