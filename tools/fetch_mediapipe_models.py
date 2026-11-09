"""Download the MediaPipe models used by --hands / --mediapipe into models/.

These are Google's official float16 v1 models; they are byte-identical to the
copies VTube Studio ships with its tracker.
"""
import hashlib
import os
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = {
    "face_landmarker_v2_with_blendshapes.task": (
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
        "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff"),
    "hand_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
        "fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1"),
}


def main():
    dst_dir = os.path.join(ROOT, "models")
    for name, (url, sha) in MODELS.items():
        dst = os.path.join(dst_dir, name)
        if os.path.exists(dst) and hashlib.sha256(open(dst, "rb").read()).hexdigest() == sha:
            print(f"{name}: ok")
            continue
        data = urllib.request.urlopen(url, timeout=60).read()
        got = hashlib.sha256(data).hexdigest()
        if got != sha:
            raise SystemExit(f"{name}: checksum mismatch ({got})")
        with open(dst, "wb") as f:
            f.write(data)
        print(f"{name}: downloaded")


if __name__ == "__main__":
    main()
