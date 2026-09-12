
# #########################    driver_monitoring_with_model.py with count eye_alert & mouth_alert   #############################

"""
Driver Drowsiness Monitor  –  with Transformer Inference
=========================================================
• Webcam captures face landmarks via MediaPipe.
• Per-frame R1 (eye ratio) and Rm1 (mouth ratio) are recorded.
• Sliding windows (450 frames ≈ 15 s, step 90 frames ≈ 3 s) compute
  Perclose and Mclose on the fly.
• Every 3 minutes the accumulated feature sequence is fed to the
  trained DrowsinessTransformer (final_best_model_fold_5.pth).
• If drowsiness is predicted a beep is played (cross-platform).
• Press Esc to stop monitoring and exit.


Requirements
------------
    pip install opencv-python mediapipe torch numpy pandas
    # model file must be in the same directory:
    #   final_best_model_fold_5.pth
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision
from datetime import datetime
import csv
import os
import sys
import time
import math
import threading
import urllib.request

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# ─── Paths ────────────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "final_best_model_fold_5.pth")  # put the .pth next to this script

# ─── MediaPipe FaceLandmarker (new Tasks API – works on Python 3.13 / mediapipe 1.x) ──
FACE_LANDMARKER_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                           "face_landmarker.task")
FACE_LANDMARKER_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)


def ensure_face_landmarker_model(path: str = FACE_LANDMARKER_MODEL_PATH) -> str:
    """Downloads the FaceLandmarker .task model once, if not already present."""
    if not os.path.isfile(path):
        print("Downloading MediaPipe FaceLandmarker model (first run only) ...")
        urllib.request.urlretrieve(FACE_LANDMARKER_MODEL_URL, path)
        print(f"Model downloaded → {path}")
    return path


def create_face_landmarker(model_path: str) -> mp_vision.FaceLandmarker:
    """Builds a FaceLandmarker configured to mimic the old FaceMesh(refine_landmarks=True)."""
    base_options = mp_tasks.BaseOptions(model_asset_path=model_path)
    options = mp_vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        min_face_presence_confidence=0.5,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    return mp_vision.FaceLandmarker.create_from_options(options)

# ─── Landmark indices ─────────────────────────────────────────────────────────
LEFT_EYE_INDICES  = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246, 33]
RIGHT_EYE_INDICES = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398, 362]
MOUTH_INDICES     = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146, 61]

# ─── Alert thresholds (for per-frame on-screen indicators) ───────────────────
R1_THRESHOLD  = 0.68   # R1  < threshold → eye closing
RM1_THRESHOLD = 1.95   # Rm1 > threshold → mouth open / yawn

# ─── Sliding-window parameters ───────────────────────────────────────────────
WINDOW_SIZE = 450 //2         # frames (~15 s at 30 fps)
STEP_SIZE   = 90 //2          # frames (~3 s at 30 fps)

# ─── Report interval ─────────────────────────────────────────────────────────
REPORT_INTERVAL_SEC = 180  # 3 minutes

# ─── Transformer model (must match training configuration exactly) ────────────
INPUT_DIM       = 2
D_MODEL         = 64
NHEAD           = 4
NUM_LAYERS      = 3
DIM_FEEDFORWARD = 256
DROPOUT         = 0.3
NUM_CLASSES     = 2
MAX_LEN_PE      = 200      # positional-encoding buffer; covers any realistic sequence

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ══════════════════════════════════════════════════════════════════════════════
# Model definition  (identical to training code)
# ══════════════════════════════════════════════════════════════════════════════

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = MAX_LEN_PE):
        super().__init__()
        pe       = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))   # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1), :]


class DrowsinessTransformer(nn.Module):
    def __init__(
        self,
        input_dim: int       = INPUT_DIM,
        d_model: int         = D_MODEL,
        nhead: int           = NHEAD,
        num_layers: int      = NUM_LAYERS,
        dim_feedforward: int = DIM_FEEDFORWARD,
        dropout: float       = DROPOUT,
        num_classes: int     = NUM_CLASSES,
    ):
        super().__init__()
        self.input_proj   = nn.Linear(input_dim, d_model)
        self.pos_encoder  = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model        = d_model,
            nhead          = nhead,
            dim_feedforward= dim_feedforward,
            dropout        = dropout,
            activation     = "gelu",
            batch_first    = True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        x    = self.input_proj(x)
        x    = self.pos_encoder(x)

        mask = (
            torch.arange(seq_len, device=x.device)
            .expand(batch_size, seq_len) >= lengths.unsqueeze(1)
        )
        memory     = self.transformer_encoder(x, src_key_padding_mask=mask)
        valid_mask = (~mask).float().unsqueeze(-1)
        pooled     = (memory * valid_mask).sum(dim=1) / valid_mask.sum(dim=1).clamp(min=1e-9)
        pooled     = self.dropout(pooled)
        return self.classifier(pooled)


# ══════════════════════════════════════════════════════════════════════════════
# Load trained model
# ══════════════════════════════════════════════════════════════════════════════

def load_model(path: str) -> DrowsinessTransformer:
    model = DrowsinessTransformer()
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Model file not found: {path}\n"
            "Place 'final_best_model_fold_5.pth' in the same directory as this script."
        )
    state = torch.load(path, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    print(f"✅ Model loaded from '{path}'  (device: {DEVICE})")
    return model


# ══════════════════════════════════════════════════════════════════════════════
# Feature computation  –  sliding windows over the raw frame buffer
# ══════════════════════════════════════════════════════════════════════════════

def compute_features_from_buffer(
    eye_buffer: list[float],
    mouth_buffer: list[float],
) -> np.ndarray:
    """
    Applies the same sliding-window logic used during training.
    Returns a float32 array of shape (num_windows, 2) where columns are
    [Perclose, Mclose].  Returns None when there are not enough frames.
    """
    eye   = np.array(eye_buffer,   dtype=np.float32)
    mouth = np.array(mouth_buffer, dtype=np.float32)

    n_frames = len(eye)
    if n_frames < WINDOW_SIZE // 3:
        return None

    features   = []
    start_idx  = 0
    while start_idx < n_frames:
        end_idx      = min(start_idx + WINDOW_SIZE, n_frames)
        w_eye        = eye[start_idx:end_idx]
        w_mouth      = mouth[start_idx:end_idx]
        n            = len(w_eye)
        if n < WINDOW_SIZE // 3:
            break
        perclose = float((w_eye   < R1_THRESHOLD).sum())  / n
        mclose   = float((w_mouth > RM1_THRESHOLD).sum()) / n
        features.append([perclose, mclose])
        start_idx += STEP_SIZE

    if not features:
        return None
    return np.array(features, dtype=np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# Inference  –  feed feature sequence to the transformer
# ══════════════════════════════════════════════════════════════════════════════

def predict_drowsiness(
    model: DrowsinessTransformer,
    features: np.ndarray,          # shape (num_windows, 2)
) -> tuple[int, float]:
    """
    Returns (predicted_class, confidence_of_sleepy).
      0 → alert / not drowsy
      1 → drowsy
    """
    seq_len = features.shape[0]

    seq_tensor    = torch.from_numpy(features).unsqueeze(0).to(DEVICE)   # (1, T, 2)
    length_tensor = torch.tensor([seq_len], dtype=torch.long).to(DEVICE)

    with torch.no_grad():
        logits = model(seq_tensor, length_tensor)           # (1, 2)
        probs  = torch.softmax(logits, dim=-1)[0]

    pred       = int(probs.argmax().item())
    confidence = float(probs[1].item())                     # P(drowsy)
    return pred, confidence


# ══════════════════════════════════════════════════════════════════════════════
# Cross-platform beep
# ══════════════════════════════════════════════════════════════════════════════

def beep_alert(n: int = 3):
    """Play n short beeps in a background thread (non-blocking)."""
    def _beep():
        for _ in range(n):
            if sys.platform == "win32":
                import winsound
                winsound.Beep(880, 300)
                time.sleep(0.15)
            elif sys.platform == "darwin":
                os.system("afplay /System/Library/Sounds/Funk.aiff 2>/dev/null || "
                          "say 'Alert' 2>/dev/null")
                time.sleep(0.5)
            else:
                # Linux: try paplay / aplay / print bell char
                played = False
                for cmd in [
                    "paplay /usr/share/sounds/freedesktop/stereo/alarm-clock-elapsed.oga",
                    "aplay -q /usr/share/sounds/alsa/Front_Left.wav",
                    "play -q -n synth 0.3 sine 880 2>/dev/null",
                ]:
                    if os.system(cmd + " 2>/dev/null") == 0:
                        played = True
                        break
                if not played:
                    print("\a", end="", flush=True)   # terminal bell fallback
                time.sleep(0.4)

    threading.Thread(target=_beep, daemon=True).start()


# ══════════════════════════════════════════════════════════════════════════════
# Polygon area helper
# ══════════════════════════════════════════════════════════════════════════════

def polygon_area(landmarks, indices: list[int], w: int, h: int) -> float:
    """landmarks: list of NormalizedLandmark (from FaceLandmarker), indexable directly."""
    pts = [(landmarks[i].x * w, landmarks[i].y * h) for i in indices]
    n, area = len(pts), 0.0
    for i in range(n):
        j = (i + 1) % n
        area += pts[i][0] * pts[j][1]
        area -= pts[j][0] * pts[i][1]
    return abs(area) / 2.0


def draw_face_mesh_manual(frame, landmarks, w: int, h: int) -> None:
    """Lightweight replacement for mp_drawing.draw_landmarks (no legacy `solutions` needed)."""
    # all points, faint gray dots
    for lm in landmarks:
        cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 1, (80, 80, 80), -1)

    def _poly(indices, color):
        pts = np.array([(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in indices],
                        dtype=np.int32)
        cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=1)

    _poly(LEFT_EYE_INDICES,  (0, 220, 100))
    _poly(RIGHT_EYE_INDICES, (0, 220, 100))
    _poly(MOUTH_INDICES,     (0, 200, 255))


# ══════════════════════════════════════════════════════════════════════════════
# Utility – draw rounded rectangle
# ══════════════════════════════════════════════════════════════════════════════

def draw_rounded_rect(img, x1, y1, x2, y2, r, color, thickness=-1):
    cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, thickness)
    cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), color, thickness)
    for cx, cy in [(x1+r, y1+r), (x2-r, y1+r), (x1+r, y2-r), (x2-r, y2-r)]:
        cv2.ellipse(img, (cx, cy), (r, r), 0, 0, 360, color, thickness)


# ══════════════════════════════════════════════════════════════════════════════
# Main monitoring loop
# ══════════════════════════════════════════════════════════════════════════════

def run_monitoring():
    # ── Load model ────────────────────────────────────────────────────────────
    model = load_model(MODEL_PATH)

    # ── Open webcam ───────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("❌ Webcam could not be opened!")
        return

    model_path = ensure_face_landmarker_model()

    print("\n📷 Webcam active")
    print("🔹 SPACE → save baseline, then start monitoring")
    print("🔹 ESC   → stop & exit\n")

    # ── State ─────────────────────────────────────────────────────────────────
    normal_eye_area   = None
    normal_mouth_area = None
    baseline_saved    = False
    monitoring        = False
    monitor_start     = None          # time when current monitoring session began
    last_report_time  = None          # time of last 3-minute inference report

    # Continuous raw-ratio buffers (grow until ESC)
    eye_ratio_buffer   = []
    mouth_ratio_buffer = []

    # Latest inference result
    last_prediction    = None          # None | "ALERT" | "OK"
    last_confidence    = 0.0
    last_report_ts     = "–"
    next_report_in     = REPORT_INTERVAL_SEC

    # ── Frame counters ─────────────────────────────────────────────────────────
    total_frames       = 0
    eye_alert_frames   = 0
    mouth_alert_frames = 0

    # CSV logging
    csv_filename = None
    csv_file     = None
    writer       = None

    # ── Overlay layout constants ───────────────────────────────────────────────
    PANEL_X, PANEL_Y = 10, 10
    PANEL_W, PANEL_H = 300, 255   # increased height to fit frame counter rows

    with create_face_landmarker(model_path) as face_landmarker:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rgb       = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int(time.time() * 1000)
            results   = face_landmarker.detect_for_video(mp_image, timestamp_ms)
            h, w, _   = frame.shape
            face_ok   = False

            R1  = None
            Rm1 = None

            # ── Face processing ───────────────────────────────────────────────
            if results.face_landmarks:
                face_ok = True
                for fl in results.face_landmarks:
                    draw_face_mesh_manual(frame, fl, w, h)

                    left_area  = polygon_area(fl, LEFT_EYE_INDICES,  w, h)
                    right_area = polygon_area(fl, RIGHT_EYE_INDICES, w, h)
                    mouth_area = polygon_area(fl, MOUTH_INDICES,      w, h)

                    if baseline_saved and normal_eye_area and normal_mouth_area:
                        R1  = max(left_area, right_area) / normal_eye_area
                        Rm1 = mouth_area / normal_mouth_area

                        # ── Increment frame counters ──────────────────────────
                        total_frames += 1
                        if R1  < R1_THRESHOLD:  eye_alert_frames   += 1
                        if Rm1 > RM1_THRESHOLD: mouth_alert_frames += 1

                        # Accumulate buffers for inference
                        eye_ratio_buffer.append(R1)
                        mouth_ratio_buffer.append(Rm1)

                        # Log to CSV
                        if monitoring and writer:
                            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                            writer.writerow([
                                ts,
                                f"{left_area:.2f}", f"{right_area:.2f}", f"{mouth_area:.2f}",
                                f"{R1:.4f}", f"{Rm1:.4f}",
                                "ALERT_EYE"   if R1  < R1_THRESHOLD  else "",
                                "ALERT_MOUTH" if Rm1 > RM1_THRESHOLD else "",
                            ])
                            csv_file.flush()

            # ── 3-minute inference trigger ────────────────────────────────────
            if monitoring and last_report_time is not None:
                elapsed_since_report = time.time() - last_report_time
                next_report_in = max(0.0, REPORT_INTERVAL_SEC - elapsed_since_report)

                if elapsed_since_report >= REPORT_INTERVAL_SEC:
                    features = compute_features_from_buffer(
                        eye_ratio_buffer, mouth_ratio_buffer
                    )
                    last_report_time = time.time()
                    last_report_ts   = datetime.now().strftime("%H:%M:%S")

                    if features is not None:
                        pred, conf = predict_drowsiness(model, features)
                        last_confidence = conf
                        if pred == 1:
                            last_prediction = "DROWSY"
                            print(f"🔴 [{last_report_ts}]  DROWSINESS DETECTED  "
                                  f"(P(drowsy)={conf:.2%})")
                            beep_alert(n=3)
                        else:
                            last_prediction = "ALERT"
                            print(f"🟢 [{last_report_ts}]  Driver is ALERT  "
                                  f"(P(drowsy)={conf:.2%})")
                    else:
                        last_prediction = "INSUF"
                        print(f"⚠️  [{last_report_ts}]  Not enough data for inference.")
                    next_report_in = REPORT_INTERVAL_SEC

            # ── Dark semi-transparent side panel ─────────────────────────────
            overlay = frame.copy()
            draw_rounded_rect(overlay, PANEL_X, PANEL_Y,
                              PANEL_X + PANEL_W, PANEL_Y + PANEL_H,
                              8, (20, 20, 20), -1)
            cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

            def txt(s, y, color=(220,220,220), scale=0.55, bold=1):
                cv2.putText(frame, s, (PANEL_X + 10, PANEL_Y + y),
                            cv2.FONT_HERSHEY_SIMPLEX, scale, color, bold, cv2.LINE_AA)

            # ── Panel content ─────────────────────────────────────────────────
            txt("DRIVER MONITOR", 24, (0,200,255), 0.62, 2)

            if R1 is not None and Rm1 is not None:
                r1c  = (0,80,255) if R1  < R1_THRESHOLD  else (80,255,80)
                rm1c = (0,80,255) if Rm1 > RM1_THRESHOLD else (80,255,200)
                txt(f"Eye  R1 : {R1:.3f}", 52,  r1c)
                txt(f"Mouth Rm1: {Rm1:.3f}", 76, rm1c)

                per = sum(1 for v in eye_ratio_buffer[-WINDOW_SIZE:]
                          if v < R1_THRESHOLD) / max(1, min(len(eye_ratio_buffer), WINDOW_SIZE))
                mcl = sum(1 for v in mouth_ratio_buffer[-WINDOW_SIZE:]
                          if v > RM1_THRESHOLD) / max(1, min(len(mouth_ratio_buffer), WINDOW_SIZE))
                txt(f"Perclose : {per:.3f}", 100, (180,180,180))
                txt(f"Mclose   : {mcl:.3f}", 122, (180,180,180))
            else:
                txt("Eye  R1 : –", 52)
                txt("Mouth Rm1: –", 76)
                txt("Perclose : –", 100)
                txt("Mclose   : –", 122)

            # Separator 1 – after ratios
            cv2.line(frame, (PANEL_X+6, PANEL_Y+132), (PANEL_X+PANEL_W-6, PANEL_Y+132),
                     (80,80,80), 1)

            # ── Frame counters ─────────────────────────────────────────────────
            eye_pct   = (eye_alert_frames   / total_frames * 100) if total_frames else 0.0
            mouth_pct = (mouth_alert_frames / total_frames * 100) if total_frames else 0.0
            txt(f"Frames  total : {total_frames}",                          148, (180,180,180), 0.48)
            txt(f"Eye   alerts  : {eye_alert_frames} ({eye_pct:.1f}%)",     164, (80,255,80),   0.48)
            txt(f"Mouth alerts  : {mouth_alert_frames} ({mouth_pct:.1f}%)", 180, (80,220,255),  0.48)

            # Separator 2 – before inference status
            cv2.line(frame, (PANEL_X+6, PANEL_Y+192), (PANEL_X+PANEL_W-6, PANEL_Y+192),
                     (80,80,80), 1)

            # Last inference result
            if last_prediction == "DROWSY":
                res_color = (30,30,255)
                res_label = f"DROWSY  {last_confidence:.0%}"
                icon_color= (0,0,255)
            elif last_prediction == "ALERT":
                res_color = (30,220,80)
                res_label = f"ALERT   {last_confidence:.0%}"
                icon_color= (0,200,60)
            elif last_prediction == "INSUF":
                res_color = (0,200,255)
                res_label = "Insufficient data"
                icon_color= (0,200,200)
            else:
                res_color = (120,120,120)
                res_label = "Pending first report"
                icon_color= (100,100,100)

            txt(f"Status: {res_label}", 212, res_color, 0.58, 2)
            txt(f"Last   : {last_report_ts}", 232, (150,150,150), 0.48)

            # ── Next report countdown bar ─────────────────────────────────────
            if monitoring and last_report_time is not None:
                bar_total = PANEL_W - 20
                bar_filled = int((1 - next_report_in / REPORT_INTERVAL_SEC) * bar_total)
                bar_filled = max(0, min(bar_filled, bar_total))
                BY = PANEL_Y + PANEL_H + 8
                cv2.rectangle(frame, (PANEL_X+10, BY), (PANEL_X+10+bar_total, BY+10),
                              (40,40,40), -1)
                cv2.rectangle(frame, (PANEL_X+10, BY), (PANEL_X+10+bar_filled, BY+10),
                              (0,200,255), -1)
                mins = int(next_report_in) // 60
                secs = int(next_report_in) % 60
                cv2.putText(frame, f"Next report: {mins:01d}m {secs:02d}s",
                            (PANEL_X+10, BY+26),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180,180,180), 1, cv2.LINE_AA)

            # ── Per-frame alert banners ───────────────────────────────────────
            banner_y = h - 80
            if R1 is not None and R1 < R1_THRESHOLD:
                cv2.rectangle(frame, (0, banner_y), (w, banner_y+28), (0,0,180), -1)
                cv2.putText(frame, "⚠  DROWSINESS: Eyes Closing",
                            (10, banner_y+20), cv2.FONT_HERSHEY_SIMPLEX,
                            0.65, (255,255,255), 2, cv2.LINE_AA)
                banner_y -= 32

            if Rm1 is not None and Rm1 > RM1_THRESHOLD:
                cv2.rectangle(frame, (0, banner_y), (w, banner_y+28), (0,100,200), -1)
                cv2.putText(frame, "⚠  YAWN: Mouth Open",
                            (10, banner_y+20), cv2.FONT_HERSHEY_SIMPLEX,
                            0.65, (255,255,255), 2, cv2.LINE_AA)

            # ── Status footer ─────────────────────────────────────────────────
            if not baseline_saved:
                foot = "SPACE: save baseline & start monitoring"
                fc   = (0, 200, 255)
            elif not monitoring:
                foot = "Baseline saved  |  SPACE: start  |  ESC: quit"
                fc   = (180, 255, 180)
            else:
                foot = "MONITORING  |  ESC: stop & exit"
                fc   = (0, 255, 200)

            cv2.rectangle(frame, (0, h-22), (w, h), (20,20,20), -1)
            cv2.putText(frame, foot, (8, h-7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, fc, 1, cv2.LINE_AA)

            # ── "DROWSY" full-frame flash ──────────────────────────────────────
            if last_prediction == "DROWSY":
                flash = frame.copy()
                cv2.rectangle(flash, (0,0), (w,h), (0,0,200), -1)
                cv2.addWeighted(flash, 0.12, frame, 0.88, 0, frame)
                cv2.putText(frame, "DROWSY DRIVER DETECTED",
                            (w//2 - 200, h//2),
                            cv2.FONT_HERSHEY_DUPLEX, 0.9, (0,30,255), 2, cv2.LINE_AA)

            cv2.imshow("Driver Drowsiness Monitor", frame)
            key = cv2.waitKey(10) & 0xFF

            # ── ESC → quit ────────────────────────────────────────────────────
            if key == 27:
                print("\n🛑 Monitoring stopped.")

                # ── Write frame counter summary to CSV ────────────────────────
                if csv_file and writer:
                    eye_pct   = (eye_alert_frames   / total_frames * 100) if total_frames else 0.0
                    mouth_pct = (mouth_alert_frames / total_frames * 100) if total_frames else 0.0
                    writer.writerow([])   # blank separator row
                    writer.writerow(["── SUMMARY ──"])
                    writer.writerow(["metric", "count", "percentage"])
                    writer.writerow(["total_frames",       total_frames,       "100.0%"])
                    writer.writerow(["eye_alert_frames",   eye_alert_frames,   f"{eye_pct:.1f}%"])
                    writer.writerow(["mouth_alert_frames", mouth_alert_frames, f"{mouth_pct:.1f}%"])
                    csv_file.close()
                    print(f"💾 Raw CSV + summary saved → {csv_filename}")
                break

            # ── SPACE → baseline or start ─────────────────────────────────────
            elif key == 32:
                if not face_ok:
                    print("⚠️  No face detected – sit in front of the camera.\n")
                    continue

                if not baseline_saved:
                    # Need current areas – re-process the last frame synchronously
                    if results.face_landmarks:
                        for fl in results.face_landmarks:
                            la = polygon_area(fl, LEFT_EYE_INDICES,  w, h)
                            ra = polygon_area(fl, RIGHT_EYE_INDICES, w, h)
                            ma = polygon_area(fl, MOUTH_INDICES,      w, h)
                            normal_eye_area   = max(la, ra)
                            normal_mouth_area = ma

                    ts_str       = datetime.now().strftime("%Y%m%d_%H%M%S")
                    img_filename = f"baseline_{ts_str}.jpg"
                    txt_filename = f"baseline_{ts_str}.txt"
                    cv2.imwrite(img_filename, frame)

                    with open(txt_filename, "w", encoding="utf-8") as f:
                        f.write(f"Timestamp        : {datetime.now()}\n")
                        f.write(f"Normal Eye Ref   : {normal_eye_area:.2f} px²\n")
                        f.write(f"Normal Mouth Ref : {normal_mouth_area:.2f} px²\n")

                    csv_filename = f"driver_ratios_{ts_str}.csv"
                    csv_file = open(csv_filename, "w", newline="", encoding="utf-8")
                    writer   = csv.writer(csv_file)
                    writer.writerow([
                        "timestamp",
                        "left_eye_area", "right_eye_area", "mouth_area",
                        "R1_eye_ratio",  "Rm1_mouth_ratio",
                        "eye_alert",     "mouth_alert",
                    ])

                    baseline_saved = True
                    print(f"\n✅ Baseline saved!")
                    print(f"   Eye ref  : {normal_eye_area:.2f}  |  Mouth ref : {normal_mouth_area:.2f}")
                    print(f"   CSV      : {csv_filename}")
                    print("\n▶  Monitoring starts in 3 s …")
                    time.sleep(3)

                if not monitoring:
                    monitoring       = True
                    monitor_start    = time.time()
                    last_report_time = time.time()
                    last_prediction  = None
                    print(f"🟢 Monitoring started  (report every {REPORT_INTERVAL_SEC//60} min)  |  ESC to stop")

    cap.release()
    cv2.destroyAllWindows()


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    _ = mp_vision.FaceLandmarker   # sanity check that the Tasks API is available
    print(f"MediaPipe Version : {mp.__version__}")
    print(f"PyTorch   Version : {torch.__version__}")
    print(f"Inference device  : {DEVICE}")
    run_monitoring()
