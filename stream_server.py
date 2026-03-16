#!/usr/bin/env python3
import os
import cv2
import json
import time
import shutil
import socket
import struct
import threading
from collections import deque

import numpy as np
import roma
import torch

from add_ckpt_path import add_path_to_dust3r
from infer_export import prepare_input, _find_joints_tensor, _jsonify

# -------------------------
# safe runtime config
# -------------------------
HOST = "127.0.0.1"
PORT = 9999

MODEL_PATH = "src/human3r_672S.pth"
DEVICE = "cuda"

SIZE = 256

USE_TTT3R = False
RESET_INTERVAL = 10000000

# 低延迟优先
WINDOW_SIZE = 6
TRIGGER_EVERY = 1
MAX_KEEP = 12

TMP_ROOT = "/amax/xuedingrong/projects/Human3R/stream_tmp"

# 这套索引目前仍是经验映射
JOINT_INDEX = {
    "root": 0,
    "head": 15,
    "left_wrist": 20,
    "right_wrist": 21,
    "left_ankle": 7,
    "right_ankle": 8,
}

# 只影响当前进程
torch.backends.cudnn.benchmark = True
torch.set_grad_enabled(False)


def recv_exact(conn, n):
    buf = b""
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return buf


def load_model():
    device = DEVICE
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA unavailable, falling back to CPU")
        device = "cpu"

    add_path_to_dust3r(MODEL_PATH)
    from src.dust3r.model import ARCroco3DStereo

    model = ARCroco3DStereo.from_pretrained(MODEL_PATH).to(device)
    model.eval()
    return model, device


def pick_joint(joints_world, pid, idx):
    if idx is None or idx < 0 or idx >= joints_world.shape[1]:
        return None
    return _jsonify(joints_world[pid][idx])


def safe_remove_dir(path):
    try:
        if os.path.exists(path):
            shutil.rmtree(path)
    except Exception:
        pass


def run_window_inference(model, device, frame_bgr_list, run_id):
    from src.dust3r.inference import inference_recurrent_lighter
    from src.dust3r.utils.camera import pose_encoding_to_camera
    from src.dust3r.utils.geometry import geotrf, matrix_cumprod
    from src.dust3r.post_process import estimate_focal_knowing_depth
    from src.dust3r.utils import SMPL_Layer

    workdir = os.path.join(TMP_ROOT, f"run_{run_id:06d}")
    safe_remove_dir(workdir)
    os.makedirs(workdir, exist_ok=True)

    img_paths = []
    for i, frame in enumerate(frame_bgr_list):
        p = os.path.join(workdir, f"{i:06d}.jpg")
        cv2.imwrite(p, frame)
        img_paths.append(p)

    img_mask = [True] * len(img_paths)
    img_res = getattr(model, "mhmr_img_res", None)

    views = prepare_input(
        img_paths=img_paths,
        img_mask=img_mask,
        size=SIZE,
        revisit=1,
        update=True,
        img_res=img_res,
        reset_interval=RESET_INTERVAL,
    )

    with torch.inference_mode():
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device == "cuda")):
            outputs, _ = inference_recurrent_lighter(
                views, model, device, use_ttt3r=USE_TTT3R
            )

    valid_length = len(outputs["pred"])
    outputs["pred"] = outputs["pred"][-valid_length:]
    outputs["views"] = outputs["views"][-valid_length:]

    reset_mask = torch.cat([view["reset"] for view in outputs["views"]], 0)
    shifted_reset_mask = torch.cat(
        [torch.tensor(False).unsqueeze(0), reset_mask[:-1]], dim=0
    )
    outputs["pred"] = [
        pred for pred, mask in zip(outputs["pred"], shifted_reset_mask) if not mask
    ]
    outputs["views"] = [
        view for view, mask in zip(outputs["views"], shifted_reset_mask) if not mask
    ]
    reset_mask = reset_mask[~shifted_reset_mask]

    if len(outputs["pred"]) == 0:
        safe_remove_dir(workdir)
        return {"frame_id": -1, "persons": []}

    pts3ds_self = torch.cat([o["pts3d_in_self_view"] for o in outputs["pred"]], 0)

    pr_poses = [
        pose_encoding_to_camera(pred["camera_pose"].clone()).cpu()
        for pred in outputs["pred"]
    ]
    if reset_mask.any():
        pr_poses_cat = torch.cat(pr_poses, 0)
        identity = torch.eye(4, device=pr_poses_cat.device)
        reset_poses = torch.where(
            reset_mask.unsqueeze(-1).unsqueeze(-1), pr_poses_cat, identity
        )
        cumulative_bases = matrix_cumprod(reset_poses)
        shifted_bases = torch.cat([identity.unsqueeze(0), cumulative_bases[:-1]], dim=0)
        pr_poses_cat = torch.einsum("bij,bjk->bik", shifted_bases, pr_poses_cat)
        pr_poses = list(pr_poses_cat.unsqueeze(1).unbind(0))

    if len(pr_poses) == 0:
        safe_remove_dir(workdir)
        return {"frame_id": -1, "persons": []}

    B, H, W, _ = pts3ds_self.shape
    pp = torch.tensor([W // 2, H // 2], device=pts3ds_self.device).float().repeat(B, 1)
    focal = estimate_focal_knowing_depth(pts3ds_self, pp, focal_mode="weiszfeld")

    intrinsics_tosave = torch.eye(3).unsqueeze(0).repeat(len(pr_poses), 1, 1)
    intrinsics_tosave[:, 0, 0] = focal.detach().cpu()
    intrinsics_tosave[:, 1, 1] = focal.detach().cpu()
    intrinsics_tosave[:, 0, 2] = pp[:, 0].cpu()
    intrinsics_tosave[:, 1, 2] = pp[:, 1].cpu()

    smpl_shape = [
        output.get("smpl_shape", torch.empty(1, 0, 10))[0]
        for output in outputs["pred"]
    ]
    smpl_rotvec = [
        roma.rotmat_to_rotvec(
            output.get("smpl_rotmat", torch.empty(1, 0, 53, 3, 3))[0]
        )
        for output in outputs["pred"]
    ]
    smpl_transl = [
        output.get("smpl_transl", torch.empty(1, 0, 3))[0]
        for output in outputs["pred"]
    ]
    smpl_expression = [
        output.get("smpl_expression", [None])[0]
        for output in outputs["pred"]
    ]
    smpl_id = [
        output.get("smpl_id", torch.empty(1, 0))[0]
        for output in outputs["pred"]
    ]

    latest = {"frame_id": len(pr_poses) - 1, "persons": []}

    f_id = len(pr_poses) - 1
    c2w = pr_poses[f_id]
    n_humans = smpl_shape[f_id].shape[0]

    if n_humans == 0:
        safe_remove_dir(workdir)
        return latest

    smpl_layer = SMPL_Layer(
        type="smplx",
        gender="neutral",
        num_betas=smpl_shape[f_id].shape[-1] if smpl_shape[f_id].numel() > 0 else 10,
        kid=False,
        person_center="head",
    )

    with torch.inference_mode():
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device == "cuda")):
            smpl_out = smpl_layer(
                smpl_rotvec[f_id],
                smpl_shape[f_id],
                smpl_transl[f_id],
                None,
                None,
                K=intrinsics_tosave[f_id].to(smpl_rotvec[f_id].device).expand(n_humans, -1, -1),
                expression=smpl_expression[f_id],
            )

    joints_local, joint_key = _find_joints_tensor(smpl_out)
    joints_world = geotrf(c2w, joints_local.unsqueeze(0))[0]

    for pid in range(n_humans):
        pid_val = int(smpl_id[f_id][pid].item()) if smpl_id[f_id].numel() > 0 else pid
        latest["persons"].append(
            {
                "id": pid_val,
                "joint_source_key": joint_key,
                "root_world": pick_joint(joints_world, pid, JOINT_INDEX["root"]),
                "head_world": pick_joint(joints_world, pid, JOINT_INDEX["head"]),
                "left_wrist_world": pick_joint(joints_world, pid, JOINT_INDEX["left_wrist"]),
                "right_wrist_world": pick_joint(joints_world, pid, JOINT_INDEX["right_wrist"]),
                "left_ankle_world": pick_joint(joints_world, pid, JOINT_INDEX["left_ankle"]),
                "right_ankle_world": pick_joint(joints_world, pid, JOINT_INDEX["right_ankle"]),
                "num_joints": int(joints_world.shape[1]),
            }
        )

    safe_remove_dir(workdir)
    return latest


class AsyncInferEngine:
    def __init__(self, model, device, conn):
        self.model = model
        self.device = device
        self.conn = conn

        self.lock = threading.Lock()
        self.cond = threading.Condition(self.lock)

        self.pending_frames = None
        self.pending_run_id = 0
        self.stopped = False

        self.worker = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker.start()

    def submit_latest(self, frame_bgr_list, run_id):
        with self.cond:
            self.pending_frames = frame_bgr_list
            self.pending_run_id = run_id
            self.cond.notify()

    def stop(self):
        with self.cond:
            self.stopped = True
            self.cond.notify_all()
        self.worker.join(timeout=1.0)

    def _worker_loop(self):
        while True:
            with self.cond:
                while self.pending_frames is None and not self.stopped:
                    self.cond.wait()

                if self.stopped:
                    return

                frames = self.pending_frames
                run_id = self.pending_run_id
                self.pending_frames = None

            t0 = time.time()
            try:
                latest = run_window_inference(self.model, self.device, frames, run_id)
                latest["server_run_id"] = run_id
                latest["server_latency_sec"] = round(time.time() - t0, 3)

                msg = (json.dumps(latest, ensure_ascii=False) + "\n").encode("utf-8")
                self.conn.sendall(msg)
                print(
                    f"[run {run_id}] done in {latest['server_latency_sec']:.2f}s, "
                    f"persons={len(latest['persons'])}"
                )
            except Exception as e:
                err = {"server_run_id": run_id, "error": str(e)}
                try:
                    self.conn.sendall((json.dumps(err, ensure_ascii=False) + "\n").encode("utf-8"))
                except Exception:
                    pass
                print(f"[run {run_id}] error: {e}")


def main():
    os.makedirs(TMP_ROOT, exist_ok=True)

    print("Loading model once...")
    model, device = load_model()
    print("Model loaded.")
    print(
        f"Listening on {HOST}:{PORT} | SIZE={SIZE}, "
        f"WINDOW_SIZE={WINDOW_SIZE}, TRIGGER_EVERY={TRIGGER_EVERY}"
    )

    frame_buffer = deque(maxlen=MAX_KEEP)
    recv_count = 0
    run_id = 0

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((HOST, PORT))
    server.listen(1)

    conn, addr = server.accept()
    print("Client connected:", addr)

    engine = AsyncInferEngine(model, device, conn)

    try:
        while True:
            header = recv_exact(conn, 4)
            if header is None:
                print("Client disconnected.")
                break

            msg_len = struct.unpack("!I", header)[0]
            payload = recv_exact(conn, msg_len)
            if payload is None:
                print("Client disconnected during payload.")
                break

            arr = np.frombuffer(payload, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is None:
                continue

            frame_buffer.append(frame)
            recv_count += 1

            if recv_count % 10 == 0:
                print(f"received frame={recv_count}, buffer={len(frame_buffer)}")

            if len(frame_buffer) >= WINDOW_SIZE and (recv_count % TRIGGER_EVERY == 0):
                run_id += 1
                engine.submit_latest(list(frame_buffer)[-WINDOW_SIZE:], run_id)

    finally:
        try:
            engine.stop()
        except Exception:
            pass
        conn.close()
        server.close()


if __name__ == "__main__":
    main()