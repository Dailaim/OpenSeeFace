"""MediaPipe hand and face landmarker pipelines running on the OpenSeeFace inference backends.

The MediaPipe Python package only runs on the CPU on Windows. This module runs the
same networks (converted to ONNX by tools/convert_mediapipe_models.py) through
backends.create_session, so they use OpenVINO (Intel oneAPI), MIGraphX (ROCm),
DirectML or PyTorch XPU/ROCm, and reimplements the graph logic MediaPipe does in
C++: SSD anchor decoding, weighted NMS, rotated ROI crops, landmark projection,
frame-to-frame ROI tracking and the face geometry (pose matrix) pipeline.

Results are returned in the same shape as mediapipe.tasks results (only the
attributes mp_tracking.py uses), so the packet code is shared.
"""
import math
import os
from types import SimpleNamespace

import cv2
import numpy as np

import backends

# --------------------------------------------------------------------------- common


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(np.asarray(x, np.float64), -100.0, 100.0)))


def _norm_rad(a):
    return a - 2 * math.pi * math.floor((a + math.pi) / (2 * math.pi))


class Rect:
    """Normalized rotated rect (MediaPipe NormalizedRect)."""
    __slots__ = ("cx", "cy", "w", "h", "rot")

    def __init__(self, cx, cy, w, h, rot=0.0):
        self.cx, self.cy, self.w, self.h, self.rot = cx, cy, w, h, rot

    def transform(self, W, H, scale=1.0, shift_x=0.0, shift_y=0.0, square_long=True):
        w, h, r = self.w, self.h, self.rot
        cx, cy = self.cx, self.cy
        if r == 0.0:
            cx += w * shift_x
            cy += h * shift_y
        else:
            cx += (W * w * shift_x * math.cos(r) - H * h * shift_y * math.sin(r)) / W
            cy += (W * w * shift_x * math.sin(r) + H * h * shift_y * math.cos(r)) / H
        if square_long:
            long_side = max(w * W, h * H)
            w, h = long_side / W, long_side / H
        return Rect(cx, cy, w * scale, h * scale, r)

    def iou(self, o):
        ax0, ay0, ax1, ay1 = self.cx - self.w / 2, self.cy - self.h / 2, self.cx + self.w / 2, self.cy + self.h / 2
        bx0, by0, bx1, by1 = o.cx - o.w / 2, o.cy - o.h / 2, o.cx + o.w / 2, o.cy + o.h / 2
        iw, ih = max(0.0, min(ax1, bx1) - max(ax0, bx0)), max(0.0, min(ay1, by1) - max(ay0, by0))
        inter = iw * ih
        union = self.w * self.h + o.w * o.h - inter
        return inter / union if union > 0 else 0.0


class MirroredBGR:
    """A camera frame (BGR) that the pipeline treats as its horizontally mirrored RGB version.

    Mirroring + BGR->RGB on a full 2560x1440 frame costs ~5 ms of CPU per frame, while the
    networks only ever see 128-256 px crops of it. crop_rect/letterbox fold the mirror into
    their affine transform / small resized image instead, so the full frame is never copied.
    """
    __slots__ = ("bgr", "shape")

    def __init__(self, bgr):
        self.bgr = bgr
        self.shape = bgr.shape


def crop_rect(image, rect, size):
    """Rotated ROI crop -> (size, size, 3) RGB uint8 (ImageToTensorCalculator)."""
    H, W = image.shape[:2]
    sw, sh = rect.w * W, rect.h * H
    cx, cy = rect.cx * W, rect.cy * H
    c, s = math.cos(rect.rot), math.sin(rect.rot)
    m = np.array([[c * sw / size, -s * sh / size, cx - 0.5 * c * sw + 0.5 * s * sh],
                  [s * sw / size, c * sh / size, cy - 0.5 * s * sw - 0.5 * c * sh]], np.float32)
    if isinstance(image, MirroredBGR):
        # sample x' = (W - 1) - x in the original frame, then BGR -> RGB on the small crop only
        m[0] = (-m[0, 0], -m[0, 1], (W - 1) - m[0, 2])
        out = cv2.warpAffine(image.bgr, m, (size, size), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE)
        return out[:, :, ::-1]
    return cv2.warpAffine(image, m, (size, size), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE)


def letterbox(image, size):
    """Keep-aspect resize with zero padding -> tensor, plus the (S, pad_x, pad_y) to undo it in normalized coords."""
    H, W = image.shape[:2]
    S = max(W, H)
    scale = size / S
    nw, nh = max(1, round(W * scale)), max(1, round(H * scale))
    ox, oy = (size - nw) // 2, (size - nh) // 2
    if isinstance(image, MirroredBGR):
        small = cv2.resize(image.bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)[:, ::-1, ::-1]
    else:
        small = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    out = np.zeros((size, size, 3), small.dtype)
    out[oy:oy + nh, ox:ox + nw] = small
    return out, (S, ox / scale, oy / scale)


def unletterbox(xy, lb, W, H):
    S, pad_x, pad_y = lb
    out = xy.copy()
    out[..., 0] = (xy[..., 0] * S - pad_x) / W
    out[..., 1] = (xy[..., 1] * S - pad_y) / H
    return out


def ssd_anchors(input_size, strides=(8, 16, 16, 16)):
    """SsdAnchorsCalculator with fixed_anchor_size, aspect ratio 1 + interpolated scale (2 anchors per layer)."""
    anchors = []
    layer = 0
    while layer < len(strides):
        count = 0
        last = layer
        while last < len(strides) and strides[last] == strides[layer]:
            count += 2
            last += 1
        grid = int(math.ceil(input_size / strides[layer]))
        for y in range(grid):
            for x in range(grid):
                for _ in range(count):
                    anchors.append(((x + 0.5) / grid, (y + 0.5) / grid))
        layer = last
    return np.asarray(anchors, np.float32)


def decode_detections(raw_boxes, raw_scores, anchors, input_size, num_keypoints, min_score):
    """TensorsToDetectionsCalculator (reverse_output_order, sigmoid scores, clip 100). Returns list of (score, box[4] xyxy, kps[n,2])."""
    scores = _sigmoid(raw_scores.reshape(-1))
    idx = np.nonzero(scores >= min_score)[0]
    if idx.size == 0:
        return []
    raw = raw_boxes.reshape(len(anchors), -1)[idx] / input_size
    a = anchors[idx]
    cx, cy = raw[:, 0] + a[:, 0], raw[:, 1] + a[:, 1]
    w, h = raw[:, 2], raw[:, 3]
    boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1)
    kps = raw[:, 4:4 + 2 * num_keypoints].reshape(-1, num_keypoints, 2) + a[:, None, :]
    return list(zip(scores[idx], boxes, kps))


def weighted_nms(dets, threshold):
    """NonMaxSuppressionCalculator, WEIGHTED algorithm with IoU overlap."""
    dets = sorted(dets, key=lambda d: -d[0])
    out = []
    while dets:
        top = dets[0]
        b = top[1]
        area = (b[2] - b[0]) * (b[3] - b[1])
        keep, rest = [], []
        for d in dets:
            c = d[1]
            iw = max(0.0, min(b[2], c[2]) - max(b[0], c[0]))
            ih = max(0.0, min(b[3], c[3]) - max(b[1], c[1]))
            inter = iw * ih
            union = area + (c[2] - c[0]) * (c[3] - c[1]) - inter
            (keep if union > 0 and inter / union > threshold else rest).append(d)
        if not keep:
            keep, rest = [top], dets[1:]
        ws = np.array([d[0] for d in keep], np.float32)
        box = np.sum([d[1] * w for d, w in zip(keep, ws)], 0) / ws.sum()
        kps = np.sum([d[2] * w for d, w in zip(keep, ws)], 0) / ws.sum()
        out.append((top[0], box, kps))
        dets = rest
    return out


def rect_from_points(p0, p1, box, W, H, target_angle):
    """DetectionsToRectsCalculator: box (xyxy normalized) + rotation from keypoint p0 -> p1."""
    rot = _norm_rad(target_angle - math.atan2(-(p1[1] - p0[1]) * H, (p1[0] - p0[0]) * W))
    return Rect((box[0] + box[2]) / 2, (box[1] + box[3]) / 2, box[2] - box[0], box[3] - box[1], rot)


def project(landmarks, rect):
    """LandmarkProjectionCalculator: ROI-normalized landmarks -> image-normalized."""
    x, y = landmarks[:, 0] - 0.5, landmarks[:, 1] - 0.5
    c, s = math.cos(rect.rot), math.sin(rect.rot)
    out = np.empty_like(landmarks)
    out[:, 0] = (c * x - s * y) * rect.w + rect.cx
    out[:, 1] = (s * x + c * y) * rect.h + rect.cy
    out[:, 2] = landmarks[:, 2] * rect.w
    return out


class OneEuro:
    """Landmark smoothing (MediaPipe OneEuroFilter, values scaled by 1 / object size)."""

    def __init__(self, min_cutoff, beta, d_cutoff=1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.x = self.dx = self.t = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def reset(self):
        self.x = self.dx = self.t = None

    def __call__(self, value, t, scale):
        v = value * scale
        if self.x is None or self.x.shape != v.shape or t <= self.t:
            self.x, self.dx, self.t = v, np.zeros_like(v), t
            return value
        dt = t - self.t
        self.t = t
        dx = (v - self.x) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        self.dx = a_d * dx + (1 - a_d) * self.dx
        cutoff = self.min_cutoff + self.beta * np.abs(self.dx)
        a = 1.0 / (1.0 + (1.0 / (2 * math.pi * cutoff)) / dt)
        self.x = a * v + (1 - a) * self.x
        return self.x / scale


def _session(model_dir, name, batch=1):
    if backends.resolve_backend() == "torch":
        # On Intel Arc (torch 2.15 XPU) one of these converted graphs triggers a device-side
        # assert that resets the GPU driver (and once rebooted the PC). Never run them there.
        raise RuntimeError("The MediaPipe models are not supported on the torch backend; use openvino/rocm/dml or --mp-engine mediapipe")
    return backends.create_session(os.path.join(model_dir, "portable", name), threads=1, batch=batch)


def _run(session, tensor):
    return session.run([], {session.get_inputs()[0].name: tensor[None].astype(np.float32)})


def _lm_objects(points):
    return [SimpleNamespace(x=float(p[0]), y=float(p[1]), z=float(p[2])) for p in points]


# --------------------------------------------------------------------------- hands


class HandLandmarkerGPU:
    PALM_SUBSET = [0, 1, 2, 3, 5, 6, 9, 10, 13, 14, 17, 18]

    def __init__(self, model_dir, num_hands=2, min_detection=0.5, min_presence=0.5, min_tracking=0.5, smooth=False):
        self.detector = _session(model_dir, "mp_hand_detector.onnx")
        self.landmarker = _session(model_dir, "mp_hand_landmarks.onnx")
        self.anchors = ssd_anchors(192)
        self.num_hands = num_hands
        self.min_detection, self.min_presence, self.min_tracking = min_detection, min_presence, min_tracking
        self.tracked = []  # rects for the next frame
        self.smooth = smooth
        self.filters = {}

    def _palms(self, image):
        H, W = image.shape[:2]
        tensor, lb = letterbox(image, 192)
        boxes, scores = _run(self.detector, tensor.astype(np.float32) / 255.0)
        dets = weighted_nms(decode_detections(boxes[0], scores[0], self.anchors, 192, 7, self.min_detection), 0.3)
        rects = []
        for score, box, kps in dets[: self.num_hands]:
            box = unletterbox(box.reshape(2, 2), lb, W, H).reshape(4)
            kps = unletterbox(kps, lb, W, H)
            rect = rect_from_points(kps[0], kps[2], box, W, H, math.pi / 2)
            rects.append(rect.transform(W, H, scale=2.6, shift_y=-0.5))
        return rects

    @classmethod
    def _rect_from_landmarks(cls, lms, W, H):
        """HandLandmarksToRectCalculator + RectTransformation(scale 2.0, shift_y -0.1)."""
        pts = lms[cls.PALM_SUBSET, :2] * (W, H)
        x0, y0 = pts[0]
        x1 = ((pts[4, 0] + pts[8, 0]) / 2 + pts[6, 0]) / 2
        y1 = ((pts[4, 1] + pts[8, 1]) / 2 + pts[6, 1]) / 2
        rot = _norm_rad(math.pi / 2 - math.atan2(-(y1 - y0), x1 - x0))
        rev = _norm_rad(-rot)
        center = (pts.max(0) + pts.min(0)) / 2
        d = pts - center
        c, s = math.cos(rev), math.sin(rev)
        proj = np.stack([d[:, 0] * c - d[:, 1] * s, d[:, 0] * s + d[:, 1] * c], 1)
        pmin, pmax = proj.min(0), proj.max(0)
        pc = (pmin + pmax) / 2
        c2, s2 = math.cos(rot), math.sin(rot)
        cx = pc[0] * c2 - pc[1] * s2 + center[0]
        cy = pc[0] * s2 + pc[1] * c2 + center[1]
        size = pmax - pmin
        return Rect(cx / W, cy / H, size[0] / W, size[1] / H, rot).transform(W, H, scale=2.0, shift_y=-0.1)

    def detect_for_video(self, image, timestamp_ms):
        """image: RGB uint8 (already mirrored like the VTS tracker does)."""
        H, W = image.shape[:2]
        rects = list(self.tracked)
        if len(rects) < self.num_hands:
            for r in self._palms(image):
                # tracked hands supersede overlapping detections (AssociationNormRectCalculator)
                if all(r.iou(t) < 0.5 for t in rects):
                    rects.append(r)
        rects = rects[: self.num_hands]
        found = []  # (presence, pts, right_prob, rect_for_next_frame or None)
        for rect in rects:
            out = _run(self.landmarker, crop_rect(image, rect, 224).astype(np.float32) / 255.0)
            lm = out[0].reshape(21, 3) / 224.0
            presence = float(np.asarray(out[1]).reshape(-1)[0])
            right = float(np.asarray(out[2]).reshape(-1)[0])
            if presence < self.min_presence:
                continue
            pts = project(lm, rect)
            nxt = self._rect_from_landmarks(pts, W, H) if presence >= self.min_tracking else None
            found.append((presence, pts, right, nxt))
        # Two ROIs can converge onto the same hand (a fresh palm detection next to a tracked
        # hand). Keep the most confident one when the landmarks overlap; MediaPipe does this
        # with its own rect association, comparing landmarks is stricter.
        found.sort(key=lambda f: -f[0])
        kept = []
        for f in found:
            size = float(np.ptp(f[1][:, :2] * (W, H), 0).max()) or 1.0
            if all(float(np.linalg.norm((f[1][:, :2] - k[1][:, :2]) * (W, H), axis=1).mean()) > 0.08 * size for k in kept):
                kept.append(f)
        hands, handedness = [], []
        for presence, pts, right, _ in kept:
            hands.append(pts)
            is_right = right > 0.5
            handedness.append([SimpleNamespace(index=0 if is_right else 1, category_name="Right" if is_right else "Left",
                                               score=right if is_right else 1.0 - right, display_name="")])
        self.tracked = [k[3] for k in kept if k[3] is not None]
        if self.smooth:
            hands = self._smooth(hands, timestamp_ms / 1000.0)
        return SimpleNamespace(hand_landmarks=[_lm_objects(h) for h in hands], handedness=handedness)

    def _smooth(self, hands, t):
        out = []
        for i, pts in enumerate(hands):
            f = self.filters.setdefault(i, OneEuro(min_cutoff=0.05, beta=80.0, d_cutoff=1.0))
            span = np.ptp(pts[:, :2], 0)
            scale = 1.0 / max(float((span[0] + span[1]) / 2), 1e-3)
            out.append(f(pts, t, scale))
        for i in list(self.filters):
            if i >= len(hands):
                self.filters[i].reset()
        return out

    def close(self):
        pass


# --------------------------------------------------------------------------- face

BLENDSHAPE_NAMES = [
    "_neutral", "browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft", "browOuterUpRight", "cheekPuff",
    "cheekSquintLeft", "cheekSquintRight", "eyeBlinkLeft", "eyeBlinkRight", "eyeLookDownLeft", "eyeLookDownRight",
    "eyeLookInLeft", "eyeLookInRight", "eyeLookOutLeft", "eyeLookOutRight", "eyeLookUpLeft", "eyeLookUpRight",
    "eyeSquintLeft", "eyeSquintRight", "eyeWideLeft", "eyeWideRight", "jawForward", "jawLeft", "jawOpen", "jawRight",
    "mouthClose", "mouthDimpleLeft", "mouthDimpleRight", "mouthFrownLeft", "mouthFrownRight", "mouthFunnel",
    "mouthLeft", "mouthLowerDownLeft", "mouthLowerDownRight", "mouthPressLeft", "mouthPressRight", "mouthPucker",
    "mouthRight", "mouthRollLower", "mouthRollUpper", "mouthShrugLower", "mouthShrugUpper", "mouthSmileLeft",
    "mouthSmileRight", "mouthStretchLeft", "mouthStretchRight", "mouthUpperUpLeft", "mouthUpperUpRight",
    "noseSneerLeft", "noseSneerRight"]

# face_blendshapes_graph.cc kLandmarksSubsetIdxs
BLENDSHAPE_SUBSET = [
    0, 1, 4, 5, 6, 7, 8, 10, 13, 14, 17, 21, 33, 37, 39, 40, 46, 52, 53, 54, 55, 58, 61, 63, 65, 66, 67, 70, 78, 80,
    81, 82, 84, 87, 88, 91, 93, 95, 103, 105, 107, 109, 127, 132, 133, 136, 144, 145, 146, 148, 149, 150, 152, 153,
    154, 155, 157, 158, 159, 160, 161, 162, 163, 168, 172, 173, 176, 178, 181, 185, 191, 195, 197, 234, 246, 249,
    251, 263, 267, 269, 270, 276, 282, 283, 284, 285, 288, 291, 293, 295, 296, 297, 300, 308, 310, 311, 312, 314,
    317, 318, 321, 323, 324, 332, 334, 336, 338, 356, 361, 362, 365, 373, 374, 375, 377, 378, 379, 380, 381, 382,
    384, 385, 386, 387, 388, 389, 390, 397, 398, 400, 402, 405, 409, 415, 454, 466, 468, 469, 470, 471, 472, 473,
    474, 475, 476, 477]


class FaceGeometry:
    """face_geometry GeometryPipeline: screen landmarks -> 4x4 pose matrix (metric, cm)."""

    def __init__(self, npz_path, vertical_fov=63.0, near=1.0):
        data = np.load(npz_path)
        self.canonical = data["canonical"].T.astype(np.float64)  # 3 x 468
        self.weights = data["weights"].astype(np.float64)
        self.sqrt_w = np.sqrt(self.weights)
        self.fov = math.radians(vertical_fov)
        self.near = near

    def _solve(self, src, dst):
        """Weighted orthogonal Procrustes (procrustes_solver.cc)."""
        sw = self.sqrt_w
        ws, wt = src * sw, dst * sw
        total = float((sw * sw).sum())
        com = (ws * sw).sum(1, keepdims=True) / total
        cws = ws - com * sw
        design = wt @ cws.T
        u, _, vt = np.linalg.svd(design)
        if np.linalg.det(u) * np.linalg.det(vt) < 0:
            u[:, 2] *= -1
        rot = u @ vt
        scale = float(((rot @ cws) * wt).sum() / (cws * ws).sum())
        rs = scale * rot
        trans = ((wt - rs @ ws) * sw).sum(1) / total
        m = np.eye(4)
        m[:3, :3] = rs
        m[:3, 3] = trans
        return m

    def matrix(self, landmarks, W, H):
        h_near = 2 * self.near * math.tan(0.5 * self.fov)
        w_near = W * h_near / H
        left, bottom = -0.5 * w_near, -0.5 * h_near
        screen = landmarks[:468].T.astype(np.float64).copy()  # 3 x N
        screen[1] = 1.0 - screen[1]
        screen *= np.array([[w_near], [h_near], [w_near]])
        screen[0] += left
        screen[1] += bottom
        depth_offset = screen[2].mean()

        def rescale_z(lm, scale):
            lm[2] = (lm[2] - depth_offset + self.near) / scale

        def unproject(lm):
            lm[0] = lm[0] * lm[2] / self.near
            lm[1] = lm[1] * lm[2] / self.near

        inter = screen.copy()
        inter[2] *= -1
        s1 = np.linalg.norm(self._solve(self.canonical, inter)[:3, 0])
        inter = screen.copy()
        rescale_z(inter, s1)
        unproject(inter)
        inter[2] *= -1
        s2 = np.linalg.norm(self._solve(self.canonical, inter)[:3, 0])
        metric = screen
        rescale_z(metric, s1 * s2)
        unproject(metric)
        metric[2] *= -1
        return self._solve(self.canonical, metric)


class FaceLandmarkerGPU:
    def __init__(self, model_dir, min_detection=0.5, min_presence=0.5, min_tracking=0.5, smooth=False):
        self.detector = _session(model_dir, "mp_face_detector.onnx")
        self.landmarker = _session(model_dir, "mp_face_landmarks.onnx")
        self.blendshapes = _session(model_dir, "mp_face_blendshapes.onnx")
        self.geometry = FaceGeometry(os.path.join(model_dir, "portable", "mp_face_geometry.npz"))
        self.anchors = ssd_anchors(128)
        self.min_detection, self.min_presence, self.min_tracking = min_detection, min_presence, min_tracking
        self.tracked = None
        self.smooth = smooth
        self.filter = OneEuro(min_cutoff=0.05, beta=80.0, d_cutoff=1.0)

    def _detect(self, image):
        H, W = image.shape[:2]
        tensor, lb = letterbox(image, 128)
        boxes, scores = _run(self.detector, tensor.astype(np.float32) / 127.5 - 1.0)
        dets = weighted_nms(decode_detections(boxes[0], scores[0], self.anchors, 128, 6, self.min_detection), 0.3)
        if not dets:
            return None
        score, box, kps = dets[0]
        box = unletterbox(box.reshape(2, 2), lb, W, H).reshape(4)
        kps = unletterbox(kps, lb, W, H)
        return rect_from_points(kps[0], kps[1], box, W, H, 0.0).transform(W, H, scale=1.5)

    @staticmethod
    def _rect_from_landmarks(lms, W, H):
        box = np.concatenate([lms[:, :2].min(0), lms[:, :2].max(0)])
        return rect_from_points(lms[33], lms[263], box, W, H, 0.0).transform(W, H, scale=1.5)

    def detect_for_video(self, image, timestamp_ms):
        H, W = image.shape[:2]
        empty = SimpleNamespace(face_landmarks=[], face_blendshapes=[], facial_transformation_matrixes=[])
        rect = self.tracked or self._detect(image)
        if rect is None:
            return empty
        out = _run(self.landmarker, crop_rect(image, rect, 256).astype(np.float32) / 255.0)
        lm = np.asarray(out[0]).reshape(478, 3) / 256.0
        presence = float(_sigmoid(np.asarray(out[1]).reshape(-1)[0]))
        if presence < self.min_presence:
            self.tracked = None
            self.filter.reset()
            return empty
        pts = project(lm, rect)
        self.tracked = self._rect_from_landmarks(pts, W, H) if presence >= self.min_tracking else None
        if self.smooth:
            span = np.ptp(pts[:, :2], 0)
            pts = self.filter(pts, timestamp_ms / 1000.0, 1.0 / max(float((span[0] + span[1]) / 2), 1e-3))
        sub = pts[BLENDSHAPE_SUBSET, :2] * (W, H)
        scores = np.asarray(_run(self.blendshapes, sub.astype(np.float32))[0]).reshape(-1)
        matrix = self.geometry.matrix(pts, W, H)
        return SimpleNamespace(
            face_landmarks=[_lm_objects(pts)],
            face_blendshapes=[[SimpleNamespace(index=i, category_name=n, score=float(s)) for i, (n, s) in enumerate(zip(BLENDSHAPE_NAMES, scores))]],
            facial_transformation_matrixes=[matrix.astype(np.float32)])

    def close(self):
        pass
