import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

class RTGene_Gazehub_Dataset(Dataset):
    def __init__(
        self,
        root,
        split="train",                 
        T_sample: int = 4,
        stride: int = 1,
        img_resize=(256, 256),
        eye_resize=(60, 36),
        transform=None,
        pad_short: bool = False,
        enableINFO: bool = True,
        test_subject: int = 1   # 1 / 2 / 3
    ):
        super().__init__()

        self.root = root
        self.img_root = os.path.join(root, "rawdata")
        self.label_root = os.path.join(root, "Label")

        self.T_sample = T_sample
        self.stride = stride
        self.img_resize = img_resize
        self.eye_resize = eye_resize
        self.transform = transform
        self.pad_short = pad_short
        self.enableINFO = enableINFO
        self.test_subject = int(test_subject)
        assert self.test_subject in [1,2,3]

        train_dir = os.path.join(self.label_root, "test")
        test_dir  = os.path.join(self.label_root, "test")

        fold_name = f"train{self.test_subject}.label"

        if self.enableINFO:
            print("[Data] Using dataset: dataset_RTGene (Gazehub preprocess)")
            print(f"[Data] Split: {split}, T_sample: {self.T_sample}")
            
        if split == "train":
            # 使用 train/ 下除 test_subject 外的 trainX.label
            label_files = [
                os.path.join(train_dir, f)
                for f in os.listdir(train_dir)
                if f.endswith(".label") and f != fold_name
            ]

        elif split == "test":
            # 使用 test/ 下对应的 trainX.label
            label_files = [
                os.path.join(test_dir, fold_name)
            ]
        else:
            raise ValueError(f"Unsupported split: {split}")

        if len(label_files) == 0:
            raise RuntimeError(f"No .label files found in {self.label_root}")

        self.subject_data = {}
        
        for label_path in label_files:
            subject = os.path.basename(label_path).replace(".label", "")

            with open(label_path, "r") as fh:
                lines = [l.strip() for l in fh.readlines() if l.strip()]

            # Skip header if exists
            if lines[0].lower().startswith("face"):
                lines = lines[1:]

            parsed = []

            def parse_floats(s):
                return [float(x) for x in s.split(",")]

            for line in lines:
                cols = line.split()
                # Format:
                # Face, Left, Right, Origin, 3DGaze(3), 3DHead(3), 2DGaze(2), 2DHead(2)

                parsed.append({
                    "face_path": cols[0].replace("\\", "/"),
                    "left_path": cols[1].replace("\\", "/"),
                    "right_path": cols[2].replace("\\", "/"),
                    "origin_str": cols[3],  # string only

                    "gaze3d": torch.tensor(parse_floats(cols[4]), dtype=torch.float32),
                    "head3d": torch.tensor(parse_floats(cols[5]), dtype=torch.float32),
                    "gaze2d": torch.tensor(parse_floats(cols[6]), dtype=torch.float32),
                    "head2d": torch.tensor(parse_floats(cols[7]), dtype=torch.float32),
                })

            # Sort by frame number (same as MPII)
            def frame_id(p):
                try:
                    return int(os.path.basename(p).split("_")[-1].split(".")[0])
                except:
                    return 0

            parsed.sort(key=lambda x: frame_id(x["face_path"]))

            face_paths = [e["face_path"] for e in parsed]

            labels = {
                "gaze3d": torch.stack([e["gaze3d"] for e in parsed]),
                "gaze2d": torch.stack([e["gaze2d"] for e in parsed]),
                "head3d": torch.stack([e["head3d"] for e in parsed]),
                "head2d": torch.stack([e["head2d"] for e in parsed]),
                "origin" : torch.tensor([0.0]),  # dummy to keep structure consistent
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

        self.index = []
        for subject, info in self.subject_data.items():
            N = info["n"]
            if N >= T_sample:
                for s in range(0, N - T_sample + 1, stride):
                    self.index.append((subject, s))
            elif pad_short:
                self.index.append((subject, 0))

        if not self.index:
            raise RuntimeError("No sliding windows created.")
        
        if enableINFO:
            print(f"[RT-GENE] Fold test_subject = {self.test_subject}")
            print(f"[RT-GENE] Loaded {len(self.subject_data)} subjects")


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

        # Sliding window indices
        if N >= self.T_sample:
            indices = range(start, start + self.T_sample)
        else:
            indices = list(range(N)) + [N-1] * (self.T_sample - N)

        faces, left_eyes, right_eyes, eyes_concat = [], [], [], []

        for i in indices:
            face_path = os.path.join(self.img_root, info["face_paths"][i])
            left_path = os.path.join(self.img_root, info["left_paths"][i])
            right_path = os.path.join(self.img_root, info["right_paths"][i])

            face = self.load_face(face_path)
            le   = self.load_eye(left_path)
            re   = self.load_eye(right_path)

            eye_lr = torch.cat([le, re], dim=2)        # [1,36,120]
            eye_lr = eye_lr.repeat(3,1,1)              # [3,36,120]

            faces.append(face)
            left_eyes.append(le)
            right_eyes.append(re)
            eyes_concat.append(eye_lr)

        faces = torch.stack(faces)
        left_eyes = torch.stack(left_eyes)
        right_eyes = torch.stack(right_eyes)
        eyes_concat = torch.stack(eyes_concat)

        # label slicing
        labels = {}
        for k,v in info["labels"].items():
            if N >= self.T_sample:
                labels[k] = v[start:start+self.T_sample].clone()
            else:
                labels[k] = torch.cat(
                    [v, v[-1:].repeat(self.T_sample-N,1)],
                    dim=0
                )

        return {
            "face": faces,
            "left_eye": left_eyes,
            "right_eye": right_eyes,
            "eyes_concat": eyes_concat,
            "labels": labels
        }

if __name__ == "__main__":
    print("[INFO] Checking dataset: RT-Gene")
    dataset_test = RTGene_Gazehub_Dataset(
        root="/home/mon3tr/miniconda3/envs/hybridGaze/dataset/RTGene",
        split="test", 
        T_sample=4, # test
        stride=1,
        img_resize=(224,224),
        pad_short=False,
        test_subject=1
    )
    print("num windows (dataset length):", len(dataset_test))

    dataset = RTGene_Gazehub_Dataset(
        root="/home/mon3tr/miniconda3/envs/hybridGaze/dataset/RTGene",
        split="train",
        T_sample=4, # test
        stride=1,
        img_resize=(224,224),
        pad_short=False,
        test_subject=1
    )
    print("num windows (dataset length):", len(dataset))
    sample = dataset[0]
    print(sample)