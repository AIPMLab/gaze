import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
import mediapipe as mp
from PIL import Image
from others.dataset_ETH_Gazehub import ETHXGaze_Gazehub_Dataset

# =========================
# Config
# =========================
OUT_ROOT = "./ETHX_with_eye"
EYE_SIZE = (60, 36)
NOISE_STD = 10

mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)

# MediaPipe eye landmark indices
LEFT_EYE_IDXS  = [33, 133, 159, 145]
RIGHT_EYE_IDXS = [362, 263, 386, 374]


def make_noise_eye():
    eye = np.random.normal(
        loc=127, scale=NOISE_STD,
        size=(EYE_SIZE[1], EYE_SIZE[0], 3)
    )
    return np.clip(eye, 0, 255).astype(np.uint8)

def crop_eye_from_landmarks(img, lmks, idxs):
    h, w = img.shape[:2]
    pts = np.array(
        [(int(lmks[i].x * w), int(lmks[i].y * h)) for i in idxs]
    )

    x1, y1 = pts.min(axis=0)
    x2, y2 = pts.max(axis=0)

    pad = int(0.4 * max(x2 - x1, y2 - y1))
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad)
    y2 = min(h, y2 + pad)

    if x2 <= x1 or y2 <= y1:
        return None

    eye = img[y1:y2, x1:x2]
    return cv2.resize(eye, EYE_SIZE)

def extract_ethx_eye(split="test", test_subjects=None):
    dataset = ETHXGaze_Gazehub_Dataset(
        root="/home/NWE/miniconda3/envs/worksacpe_gaze/dataset/ETHGaze-gazehub",
        T_sample=1,
        stride=1,
        img_resize=(256, 256),
        eye_resize=EYE_SIZE,
        transform=None,
        pad_short=False,
        enableINFO=True,
        split=split,
        test_subjects=test_subjects,
    )

    os.makedirs(OUT_ROOT, exist_ok=True)
    img_root = os.path.join(OUT_ROOT, "Image")
    lbl_root = os.path.join(OUT_ROOT, "Label")
    os.makedirs(img_root, exist_ok=True)
    os.makedirs(lbl_root, exist_ok=True)

    label_path = os.path.join(lbl_root, f"{split}.label")
    fout = open(label_path, "w")
    fout.write("face left right gaze head origin cam_index frame_index normmat\n")

    for subject, data in tqdm(dataset.subject_data.items(), desc=f"Processing {split}"):
        subj_dir = os.path.join(img_root, subject)
        for sub in ["face", "left", "right"]:
            os.makedirs(os.path.join(subj_dir, sub), exist_ok=True)

        for i in range(data["n"]):
            face_rel = os.path.join(dataset.root, "Image/train", data["face_paths"][i])
            face_img = Image.open(face_rel).convert("RGB")
            face_img = np.array(face_img)        # PIL → numpy, RGB
            rgb = face_img                        # MediaPipe 直接用
            res = mp_face_mesh.process(rgb)

            left_eye = make_noise_eye()
            right_eye = make_noise_eye()

            if res.multi_face_landmarks:
                lmks = res.multi_face_landmarks[0].landmark
                le = crop_eye_from_landmarks(face_img, lmks, LEFT_EYE_IDXS)
                re = crop_eye_from_landmarks(face_img, lmks, RIGHT_EYE_IDXS)

                if le is not None:
                    left_eye = le
                if re is not None:
                    right_eye = re

            fname = os.path.basename(face_rel)
            face_bgr  = cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR)
            left_bgr  = cv2.cvtColor(left_eye, cv2.COLOR_RGB2BGR)
            right_bgr = cv2.cvtColor(right_eye, cv2.COLOR_RGB2BGR)

            cv2.imwrite(os.path.join(subj_dir, "face", fname), face_bgr)
            cv2.imwrite(os.path.join(subj_dir, "left", fname), left_bgr)
            cv2.imwrite(os.path.join(subj_dir, "right", fname), right_bgr)
            # -------- label --------
            gaze = data["labels"]["gaze2d"][i]
            head = data["labels"]["head2d"][i]
            origin = data["labels"]["origin"][i]
            rmat = data["labels"]["rmat"][i]
            frame_idx = data["labels"]["frame_idx"][i] if "frame_idx" in data["labels"] else 0

            fout.write(
                f"{subject}/face/{fname} "
                f"{subject}/left/{fname} "
                f"{subject}/right/{fname} "
                f"{gaze[0].item()},{gaze[1].item()} "
                f"{head[0].item()},{head[1].item()} "
                f"{subject}/face/{fname} "
                f"0 {frame_idx} "
                + ",".join([str(x.item()) for x in rmat.flatten()])
                + "\n"
            )

    fout.close()
    print("✓ Eye crop extraction finished.")


if __name__ == "__main__":
    extract_ethx_eye(
        split="test",
        test_subjects=["subject0008", "subject0012"]
    )
