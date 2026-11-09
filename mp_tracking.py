"""MediaPipe hand tracking and blendshape face tracking, wire-compatible with the
OpenSeeFace fork that ships with VTube Studio (OpenSeeFaceTracker_v_1_20_2).

Modes (same flags VTube Studio passes to its tracker):
    --hands 1       OpenSeeFace face tracking + MediaPipe hands
    --hands 2       MediaPipe hands only
    --mediapipe 1   MediaPipe face (52 ARKit-style blendshapes + 478 points) instead of OpenSeeFace

Packet layouts (little endian, every packet is sent to ports target-6 .. target+6):
    hands (712 bytes):  d time, i num_hands, i camera,
                        up to 2 x (i index, i handedness [0 = "Left", 1 = other], f score),
                        up to 2 x 21 x (i landmark id, f x, f y, f z), zero padded
    face (6032 bytes):  d time, i num_faces, i camera, i width, i height,
                        16 f facial transformation matrix (row major),
                        52 f blendshape scores, 478 x (f x, f y, f z), zero padded

With a GPU backend selected, the MediaPipe networks run on it through mp_gpu.py
(converted by tools/convert_mediapipe_models.py); otherwise the mediapipe
package is used, which only runs on the CPU on Windows.
"""
import os
import struct
import sys

import cv2
import numpy as np

HAND_PACKET_SIZE = 712
FACE_PACKET_SIZE = 6032
FACE_MODEL = "face_landmarker_v2_with_blendshapes.task"
HAND_MODEL = "hand_landmarker.task"
PORT_SPREAD = range(-6, 7)


def _model_base(model_dir):
    if model_dir:
        return model_dir
    base = os.path.dirname(sys.executable) if getattr(sys, "frozen", False) else os.path.dirname(os.path.abspath(__file__))
    for c in (os.path.join(base, "models"), os.path.join(base, "..", "models"), "models"):
        if os.path.isdir(c):
            return c
    return "models"


def find_model(name, model_dir):
    candidates = []
    if model_dir:
        candidates.append(os.path.join(model_dir, name))
    base = os.path.dirname(sys.executable) if getattr(sys, "frozen", False) else os.path.dirname(os.path.abspath(__file__))
    candidates += [os.path.join(base, "models", name), os.path.join(base, "..", "models", name), os.path.join("models", name)]
    for c in candidates:
        if os.path.exists(c):
            return c
    raise FileNotFoundError(f"MediaPipe model {name} not found (looked in: {', '.join(candidates)}). "
                            "Run `uv run python tools/fetch_mediapipe_models.py`.")


class MediaPipeTracking:
    def __init__(self, model_dir=None, hands=0, face=False, hand_conf=0.5, face_conf=0.3, log=False, engine="auto"):
        """engine: "gpu" runs the converted models on the selected OpenSeeFace backend (mp_gpu.py),
        "mediapipe" uses the mediapipe package (CPU only on Windows), "auto" picks gpu unless the
        backend is cpu or the converted models are missing."""
        self.log = log
        self.track_hands = hands in (1, 2)
        self.only_hands = hands == 2
        self.track_face = bool(face) and not self.only_hands
        self.last_ts = -1
        self.hand_landmarker = None
        self.face_landmarker = None
        self.engine = self._pick_engine(engine, model_dir)
        if self.engine == "gpu":
            import mp_gpu
            base = _model_base(model_dir)
            if self.track_hands:
                self.hand_landmarker = mp_gpu.HandLandmarkerGPU(base, 2, hand_conf, hand_conf, hand_conf)
            if self.track_face:
                self.face_landmarker = mp_gpu.FaceLandmarkerGPU(base, face_conf, face_conf, face_conf)
        else:
            import mediapipe as mp
            from mediapipe.tasks.python import BaseOptions
            from mediapipe.tasks.python import vision
            self.mp = mp
            if self.track_hands:
                with open(find_model(HAND_MODEL, model_dir), "rb") as fh:
                    opts = vision.HandLandmarkerOptions(
                        base_options=BaseOptions(model_asset_buffer=fh.read()),
                        num_hands=2,
                        min_hand_detection_confidence=hand_conf,
                        min_hand_presence_confidence=hand_conf,
                        min_tracking_confidence=hand_conf,
                        running_mode=vision.RunningMode.VIDEO)
                self.hand_landmarker = vision.HandLandmarker.create_from_options(opts)
            if self.track_face:
                with open(find_model(FACE_MODEL, model_dir), "rb") as fh:
                    opts = vision.FaceLandmarkerOptions(
                        base_options=BaseOptions(model_asset_buffer=fh.read()),
                        min_face_detection_confidence=face_conf,
                        min_face_presence_confidence=face_conf,
                        min_tracking_confidence=face_conf,
                        output_face_blendshapes=True,
                        output_facial_transformation_matrixes=True,
                        running_mode=vision.RunningMode.VIDEO)
                self.face_landmarker = vision.FaceLandmarker.create_from_options(opts)
        # hands and face are independent; run them side by side (the backends release the GIL)
        self.pool = None
        if self.hand_landmarker is not None and self.face_landmarker is not None:
            from concurrent.futures import ThreadPoolExecutor
            self.pool = ThreadPoolExecutor(max_workers=1)

    @staticmethod
    def _pick_engine(engine, model_dir):
        if engine in ("gpu", "mediapipe"):
            return engine
        import backends
        have_models = os.path.exists(os.path.join(_model_base(model_dir), "portable", "mp_hand_landmarks.onnx"))
        # torch is excluded: see mp_gpu._session (GPU driver resets on Arc)
        return "gpu" if backends.resolve_backend() not in ("cpu", "torch") and have_models else "mediapipe"

    @property
    def replaces_osf_face(self):
        return self.only_hands or self.track_face

    def _timestamp(self, now):
        # detect_for_video needs strictly increasing timestamps
        ts = max(int(round(now * 1000)), self.last_ts + 1)
        self.last_ts = ts
        return ts

    def process(self, frame, now, camera, width, height):
        """Run the enabled MediaPipe models on a BGR frame and return the packets to send."""
        packets = []
        if self.hand_landmarker is None and self.face_landmarker is None:
            return packets
        # The VTS tracker feeds MediaPipe a horizontally mirrored image
        rgb = np.ascontiguousarray(frame[:, ::-1, ::-1])  # mirror + BGR->RGB in one single-threaded copy
        image = rgb if self.engine == "gpu" else self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)
        ts = self._timestamp(now)
        face_future = None
        if self.pool is not None:
            face_future = self.pool.submit(self.face_landmarker.detect_for_video, image, ts)
        if self.hand_landmarker is not None:
            packets.append(hand_packet(self.hand_landmarker.detect_for_video(image, ts), now, camera))
            if self.log:
                print(f"Hand tracking finished at time {now}")
        if self.face_landmarker is not None:
            result = face_future.result() if face_future is not None else self.face_landmarker.detect_for_video(image, ts)
            packets.append(face_packet(result, now, camera, width, height))
        return packets

    def close(self):
        for lm in (self.hand_landmarker, self.face_landmarker):
            if lm is not None:
                lm.close()


def _pad(data, size):
    if len(data) > size:
        raise ValueError(f"packet too large: {len(data)} > {size}")
    return bytes(data) + bytes(size - len(data))


def hand_packet(result, now, camera):
    handedness = result.handedness or []
    data = bytearray(struct.pack("<dii", now, len(handedness), camera))
    for hand in handedness[:2]:
        cat = hand[0]
        data += struct.pack("<iif", cat.index, 0 if cat.category_name == "Left" else 1, cat.score)
    for hand in (result.hand_landmarks or [])[:2]:
        for idx, mark in enumerate(hand):
            data += struct.pack("<ifff", idx, mark.x, mark.y, mark.z)
    return _pad(data, HAND_PACKET_SIZE)


def face_packet(result, now, camera, width, height):
    matrices = result.facial_transformation_matrixes or []
    data = bytearray(struct.pack("<diiii", now, len(matrices), camera, int(width), int(height)))
    if matrices:
        data += np.asarray(matrices[0], dtype="<f4").reshape(4, 4).tobytes()
        data += struct.pack(f"<{len(result.face_blendshapes[0])}f", *(b.score for b in result.face_blendshapes[0]))
        data += np.array([(p.x, p.y, p.z) for p in result.face_landmarks[0]], dtype="<f4").tobytes()
    return _pad(data, FACE_PACKET_SIZE)
