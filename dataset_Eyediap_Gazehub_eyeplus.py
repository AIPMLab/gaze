import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import cv2
import os
import random
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset

class Eyediap_Gazehub_Dataset(Dataset):
    def __init__(
        self,
        root,
        T_sample: int = 32,
        stride: int = 1,
        img_resize=(224, 224),
        eye_resize=(60, 36),
        transform=None,
        pad_short: bool = False,
        enableINFO: bool = True,
        split: str = "train",              
        test_subjects: list = None         
    ):
        super().__init__()
        self.root = root
        self.img_root = os.path.join(root, "Image")
        self.label_root = os.path.join(root, "Label")
        # self.label_root = os.path.join(root, "ClusterLabel")

        self.T_sample = T_sample
        self.stride = stride
        self.img_resize = img_resize
        self.eye_resize = eye_resize
        self.transform = transform
        self.pad_short = pad_short
        self.enableINFO= enableINFO

        if self.enableINFO:
            print("[Data] Using EyeDiap-preprocess by Gazehub")
            print(f"[Data] T_sample: {self.T_sample}")

        label_files = sorted([f for f in os.listdir(self.label_root) if f.endswith(".label")])
        
        if not label_files:
            raise RuntimeError(f"No .label files found in {self.label_root}")

        self.subject_data = {}

        for lf in label_files:
            subject = lf.replace(".label", "")
            label_path = os.path.join(self.label_root, lf)

            with open(label_path, "r") as fh:
                lines = [l.strip() for l in fh.readlines() if l.strip()]

            if lines[0].lower().startswith("face"):
                lines = lines[1:]

            parsed = []
            for line in lines:
                cols = line.split()
                face_rel = cols[0].replace("\\", "/")
                left_rel = cols[1].replace("\\", "/")
                right_rel = cols[2].replace("\\", "/")
                def parse_floats(s):
                    return [float(x) for x in s.split(",")]

                parsed.append({
                    "face_path": face_rel,
                    "left_path": left_rel,
                    "right_path": right_rel,
                    "gaze3d": torch.tensor(parse_floats(cols[4])),
                    "head3d": torch.tensor(parse_floats(cols[5])),
                    "gaze2d": torch.tensor(parse_floats(cols[6])),
                    "head2d": torch.tensor(parse_floats(cols[7])),
                    "rvec": torch.tensor(parse_floats(cols[8])),
                    "svec": torch.tensor(parse_floats(cols[9])),
                    "origin": torch.tensor(parse_floats(cols[10])),
                })

            def frame_id(p):
                try: return int(os.path.basename(p).split(".")[0])
                except: return 0

            parsed.sort(key=lambda x: frame_id(x["face_path"]))

            face_paths = [e["face_path"] for e in parsed]

            labels = {
                k: torch.stack([e[k] for e in parsed], dim=0)
                for k in ["gaze3d", "gaze2d", "head3d", "head2d", "rvec", "svec", "origin"]
            }

            left_paths = [e["left_path"] for e in parsed]
            right_paths = [e["right_path"] for e in parsed]

            self.subject_data[subject] = {
                "face_paths": face_paths,
                "left_paths": left_paths,
                "right_paths": right_paths,
                "labels": labels,
                "n": len(face_paths),
            }
            
        if test_subjects is None:
            test_subjects = []

        test_subjects = set(test_subjects)

        if split == "train":
            selected_subjects = [
                s for s in self.subject_data.keys()
                if s not in test_subjects
            ]
        elif split == "test":
            selected_subjects = [
                s for s in self.subject_data.keys()
                if s in test_subjects
            ]
        else:
            raise ValueError(f"Unknown split: {split}")
        
        if self.enableINFO: print(f"[Data] **{split}** Using {len(selected_subjects)} subjects")
    
        self.index = []
        for subject in selected_subjects:
            info = self.subject_data[subject]

            N = info["n"]
            if N >= T_sample:
                for s in range(0, N - T_sample + 1, stride):
                    self.index.append((subject, s))
            elif pad_short:
                self.index.append((subject, 0))

        if not self.index:
            raise RuntimeError("No sliding windows created.")
    
    def __len__(self):
        return len(self.index)

    def load_face(self, path):
        img = Image.open(path).convert("RGB")
        img = img.resize(self.img_resize)
        img = torch.from_numpy(np.array(img)).permute(2,0,1).float() / 255.0
        return img

    def load_eye(self, path):
        img = Image.open(path).convert("L")
        img = img.resize(self.eye_resize)
        tensor = torch.from_numpy(np.array(img)).float() / 255.0
        tensor = tensor.unsqueeze(0).repeat(3, 1, 1)
        return tensor
    
    def __getitem__(self, idx):
        subject, start = self.index[idx]
        info = self.subject_data[subject]
        N = info["n"]

        indices = (list(range(start, start+self.T_sample))
                   if N >= self.T_sample
                   else list(range(N)) + [N-1]*(self.T_sample-N))

        # full-face sequence
        faces, left_eyes, right_eyes, eyes_concat = [], [], [], []

        for i in indices:
            face_path = os.path.join(self.img_root, info["face_paths"][i])
            left_path = os.path.join(self.img_root, info["left_paths"][i])
            right_path = os.path.join(self.img_root, info["right_paths"][i])

            face = self.load_face(face_path)                # [3,224,224]
            le   = self.load_eye(left_path)                 # [1,36,60]
            re   = self.load_eye(right_path)                # [1,36,60]

            # 左右拼接
            eye_lr = torch.cat([le, re], dim=2)

            faces.append(face)
            left_eyes.append(le)
            right_eyes.append(re)
            eyes_concat.append(eye_lr)

        faces = torch.stack(faces, dim=0)
        left_eyes = torch.stack(left_eyes, dim=0)
        right_eyes = torch.stack(right_eyes, dim=0)
        eyes_concat = torch.stack(eyes_concat, dim=0)

        # labels
        labels = {}
        for k,v in info["labels"].items():
            if N >= self.T_sample:
                labels[k] = v[start:start+self.T_sample].clone()
            else:
                labels[k] = torch.cat([v, v[-1:].repeat(self.T_sample-N,1)], dim=0)

        return {
            "face": faces,                 # [T,3,224,224]
            "left_eye": left_eyes,         # [T,1,36,60]
            "right_eye": right_eyes,       # [T,1,36,60]
            "eyes_concat": eyes_concat,    # NEW [T,3,36,120]
            "labels": labels
        }


if __name__ == "__main__":
    print("[INFO] Checking dataset: Eyediap_eyeplus")
    dataset_test = Eyediap_Gazehub_Dataset(
        root="/home/NWE/miniconda3/envs/worksacpe_gaze/dataset/Eyediap-gazehub",
        T_sample = 4,
        stride = 1,
        img_resize=(256, 256),
        eye_resize=(60, 36),
        transform=None,
        pad_short = False,
        enableINFO = True,
        test_subjects=["p4"],
        split="test"
    )
    
    print(f"Test length: {len(dataset_test)} samples")
    dataset_train = Eyediap_Gazehub_Dataset(
        root="/home/NWE/miniconda3/envs/worksacpe_gaze/dataset/Eyediap-gazehub",
        T_sample = 4,
        stride = 1,
        img_resize=(256, 256),
        eye_resize=(60, 36),
        transform=None,
        pad_short = False,
        enableINFO = True,
        test_subjects=["p4"],
        split="train"
    )
    print(f"Train length: {len(dataset_train)} samples")
    
    # 取一个样本测试
    sample = dataset_train[0]
    for i in sample: print(i," ")
    print("Face:", sample["face"].shape)
    print("left_eye:", sample["left_eye"].shape)
    print("right_eye:", sample["right_eye"].shape)
    print("eyes_concat:", sample["eyes_concat"].shape)
    print("labels:", sample["labels"])